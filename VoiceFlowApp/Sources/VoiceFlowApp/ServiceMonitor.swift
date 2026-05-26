import Foundation
import Combine

/// Polls the running Parakeet daemon and llama-server for liveness, PID, uptime,
/// memory usage, and request count. Drives the Models page dashboard.
///
/// Uses `ps` for the heavy lifting (PID/RSS/etag-style start time) — Apple's
/// `sysctl` API works but `ps` is one fork per refresh and the data is small.
final class ServiceMonitor: ObservableObject {
    struct Service {
        var name: String
        var endpoint: String?       // e.g. http://127.0.0.1:8080 (nil for daemons)
        var pid: Int?
        var rssBytes: UInt64?       // resident memory
        var startedAt: Date?
        var requestCount: Int?      // since process start, parsed from log file
        var isAlive: Bool { pid != nil }
        var uptimeText: String {
            guard let s = startedAt else { return "—" }
            let dt = Date().timeIntervalSince(s)
            return Self.formatDuration(dt)
        }
        var memoryText: String {
            guard let rss = rssBytes else { return "—" }
            let mb = Double(rss) / 1_048_576.0
            if mb >= 1024 { return String(format: "%.2f GB", mb / 1024.0) }
            return String(format: "%.0f MB", mb)
        }

        static func formatDuration(_ s: TimeInterval) -> String {
            let total = Int(s)
            if total < 60 { return "\(total)s" }
            let m = total / 60, h = m / 60
            if h >= 1 { return "\(h)h \(m % 60)m" }
            return "\(m)m \(total % 60)s"
        }
    }

    @Published var parakeet = Service(name: "Speech service")
    @Published var llamaServer = Service(name: "Language model service", endpoint: "http://127.0.0.1:8080")

    private var timer: Timer?

    func start() {
        refresh()
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.refresh()
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    func refresh() {
        // Parakeet — PID file at /tmp/voiceflow_parakeet_daemon.pid; binary contains "parakeet_asr_daemon"
        var p = parakeet
        if let pid = readPidFile("/tmp/voiceflow_parakeet_daemon.pid"), processAlive(pid) {
            p.pid = pid
            (p.rssBytes, p.startedAt) = procStats(pid)
            p.requestCount = countLogLines("/tmp/voiceflow_parakeet_daemon.log", needle: "Transcribed in")
        } else {
            // Fall back to scanning by name
            if let pid = pgrep("parakeet_asr_daemon") {
                p.pid = pid
                (p.rssBytes, p.startedAt) = procStats(pid)
                p.requestCount = countLogLines("/tmp/voiceflow_parakeet_daemon.log", needle: "Transcribed in")
            } else {
                p.pid = nil; p.rssBytes = nil; p.startedAt = nil; p.requestCount = nil
            }
        }
        parakeet = p

        // llama-server (Bonsai) — pgrep by binary path
        var l = llamaServer
        if let pid = pgrep("llama-server.*Bonsai") ?? pgrep("PrismML-llama.cpp/build/bin/llama-server") {
            l.pid = pid
            (l.rssBytes, l.startedAt) = procStats(pid)
            l.requestCount = countLogLines("/tmp/llama_server_bonsai.log", needle: "POST /v1")
        } else {
            l.pid = nil; l.rssBytes = nil; l.startedAt = nil; l.requestCount = nil
        }
        llamaServer = l
    }

    // MARK: - process helpers

    private func readPidFile(_ path: String) -> Int? {
        guard let data = try? String(contentsOfFile: path, encoding: .utf8),
              let pid = Int(data.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return nil
        }
        return pid
    }

    private func processAlive(_ pid: Int) -> Bool {
        // kill(pid, 0) checks existence without sending a signal.
        kill(pid_t(pid), 0) == 0
    }

    private func pgrep(_ pattern: String) -> Int? {
        let pipe = Pipe()
        let task = Process()
        task.launchPath = "/usr/bin/pgrep"
        task.arguments = ["-f", pattern]
        task.standardOutput = pipe
        task.standardError = Pipe()  // discard
        do {
            try task.run()
            task.waitUntilExit()
            let out = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let str = String(data: out, encoding: .utf8) else { return nil }
            // Take the first PID line — pgrep can return multiple
            if let firstLine = str.split(separator: "\n").first,
               let pid = Int(firstLine) {
                return pid
            }
        } catch {
            return nil
        }
        return nil
    }

    /// Returns (RSS bytes, start date) for a PID by parsing `ps -o rss=,lstart= -p <pid>`.
    private func procStats(_ pid: Int) -> (UInt64?, Date?) {
        let pipe = Pipe()
        let task = Process()
        task.launchPath = "/bin/ps"
        task.arguments = ["-o", "rss=,lstart=", "-p", String(pid)]
        task.standardOutput = pipe
        task.standardError = Pipe()
        do {
            try task.run()
            task.waitUntilExit()
            let raw = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let str = String(data: raw, encoding: .utf8) else { return (nil, nil) }
            let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
            // First whitespace-separated field is RSS in KB; remainder is the lstart format
            //   like "Tue May  6 11:19:48 2026"
            guard let firstSpace = trimmed.firstIndex(where: { $0 == " " }) else {
                return (nil, nil)
            }
            let rssStr = String(trimmed[..<firstSpace])
            let dateStr = String(trimmed[trimmed.index(after: firstSpace)...])
                .trimmingCharacters(in: .whitespaces)

            let rssKB = UInt64(rssStr)
            let rssBytes = rssKB.map { $0 * 1024 }

            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = "EEE MMM d HH:mm:ss yyyy"
            let date = formatter.date(from: dateStr)

            return (rssBytes, date)
        } catch {
            return (nil, nil)
        }
    }

    /// Cheap line count for a log file matching a needle. Reads the entire file —
    /// fine for VoiceFlow's small log files (<10MB) and we only refresh every 3s.
    private func countLogLines(_ path: String, needle: String) -> Int? {
        guard let str = try? String(contentsOfFile: path, encoding: .utf8) else {
            return nil
        }
        var count = 0
        var iter = str.startIndex
        while let r = str.range(of: needle, range: iter..<str.endIndex) {
            count += 1
            iter = r.upperBound
        }
        return count
    }
}
