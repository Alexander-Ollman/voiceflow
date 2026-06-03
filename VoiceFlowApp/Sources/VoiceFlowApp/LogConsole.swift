import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Log model

enum VFLogLevel: String {
    case info
    case warning
    case error

    var color: Color {
        switch self {
        case .info:    return .white.opacity(0.78)
        case .warning: return Color(red: 1.0, green: 0.78, blue: 0.33)
        case .error:   return Color(red: 1.0, green: 0.45, blue: 0.42)
        }
    }
}

enum VFLogSource: String, CaseIterable, Identifiable {
    case app      = "App"
    case parakeet = "Parakeet"
    case llama    = "LLM Server"
    case ffi      = "Pipeline"

    var id: String { rawValue }

    /// Short tag shown at the start of each row.
    var short: String {
        switch self {
        case .app:      return "app"
        case .parakeet: return "parakeet"
        case .llama:    return "llm"
        case .ffi:      return "pipeline"
        }
    }

    var tint: Color {
        switch self {
        case .app:      return .white.opacity(0.45)
        case .parakeet: return Color(red: 0.45, green: 0.78, blue: 1.0).opacity(0.85)
        case .llama:    return Color(red: 0.72, green: 0.55, blue: 1.0).opacity(0.9)
        case .ffi:      return Color(red: 0.55, green: 0.85, blue: 0.6).opacity(0.85)
        }
    }
}

struct VFLogLine: Identifiable {
    let id: UInt64
    let date: Date
    let source: VFLogSource
    let level: VFLogLevel
    let text: String
}

// MARK: - Log store

/// Central diagnostics buffer.
///
/// 1. Redirects the process `stdout`/`stderr` into an in-memory ring buffer so
///    the app's many `print()` calls — which are otherwise invisible in a
///    bundled `.app` launched from Finder — are captured. Captured output is
///    teed back to the original fd (no-op when that was /dev/null) and appended
///    to `~/Library/Logs/VoiceFlow/voiceflow.log`.
/// 2. Tails the child-process log files (Parakeet daemon, llama-server, Rust
///    FFI) and merges their new lines into the same timeline.
///
/// Install as early as possible via `LogStore.shared.start()`.
final class LogStore: ObservableObject {
    static let shared = LogStore()

    @Published private(set) var lines: [VFLogLine] = []

    private let maxLines = 10_000
    private var nextID: UInt64 = 0
    private var started = false

    /// Serializes all parsing/file work off the main thread.
    private let queue = DispatchQueue(label: "com.era-laboratories.voiceflow.logstore")

    // stdout/stderr capture
    private var stdoutPipe: Pipe?
    private var savedStdout: Int32 = -1
    private var remainder = Data()

    // persistent app log
    private(set) var logFileURL: URL?
    private var fileHandle: FileHandle?
    private static let fileFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return f
    }()

    // child-process log tailing
    private var tailOffsets: [VFLogSource: UInt64] = [:]
    private var tailTimer: Timer?
    private let childPaths: [(VFLogSource, String)] = [
        (.parakeet, "/tmp/voiceflow_parakeet_daemon.log"),
        (.llama,    "/tmp/llama_server_bonsai.log"),
        (.ffi,      "/tmp/voiceflow_debug.log"),
    ]

    private init() {}

    func start() {
        guard !started else { return }
        started = true
        setupFile()
        installStdoutCapture()
        initChildTails()
        startTailTimer()
    }

    func clear() {
        lines.removeAll()
    }

    // MARK: Persistent file

    private func setupFile() {
        let fm = FileManager.default
        guard let lib = try? fm.url(for: .libraryDirectory, in: .userDomainMask,
                                    appropriateFor: nil, create: false) else { return }
        let dir = lib.appendingPathComponent("Logs/VoiceFlow", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("voiceflow.log")

        // Rotate if the file has grown beyond ~5 MB so it can't balloon forever.
        if let attrs = try? fm.attributesOfItem(atPath: url.path),
           let size = attrs[.size] as? UInt64, size > 5_000_000 {
            try? fm.removeItem(at: url)
        }
        if !fm.fileExists(atPath: url.path) {
            fm.createFile(atPath: url.path, contents: nil)
        }
        logFileURL = url
        fileHandle = try? FileHandle(forWritingTo: url)
        _ = try? fileHandle?.seekToEnd()
    }

    // MARK: stdout/stderr capture

    private func installStdoutCapture() {
        // Unbuffered so print() lines surface immediately, even when stdout is
        // not a TTY (the bundled-app case).
        setvbuf(stdout, nil, _IONBF, 0)

        savedStdout = dup(STDOUT_FILENO)

        let pipe = Pipe()
        stdoutPipe = pipe
        let wfd = pipe.fileHandleForWriting.fileDescriptor
        dup2(wfd, STDOUT_FILENO)
        dup2(wfd, STDERR_FILENO)

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self else { return }
            let data = handle.availableData
            guard !data.isEmpty else { return }
            // Tee back to the original console (no-op if it was /dev/null).
            if self.savedStdout >= 0 {
                data.withUnsafeBytes { raw in
                    if let base = raw.baseAddress {
                        _ = write(self.savedStdout, base, data.count)
                    }
                }
            }
            self.queue.async { self.ingestAppData(data) }
        }
    }

    private func ingestAppData(_ data: Data) {
        remainder.append(data)
        while let nl = remainder.firstIndex(of: 0x0A) {
            let lineData = remainder.subdata(in: remainder.startIndex..<nl)
            remainder.removeSubrange(remainder.startIndex...nl)
            appendLine(source: .app, raw: String(decoding: lineData, as: UTF8.self))
        }
    }

    // MARK: Child-process tailing

    private func initChildTails() {
        for (src, path) in childPaths {
            let attrs = try? FileManager.default.attributesOfItem(atPath: path)
            tailOffsets[src] = (attrs?[.size] as? UInt64) ?? 0
        }
    }

    private func startTailTimer() {
        let t = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.queue.async { self?.pollChildLogs() }
        }
        RunLoop.main.add(t, forMode: .common)
        tailTimer = t
    }

    private func pollChildLogs() {
        for (src, path) in childPaths {
            guard let fh = FileHandle(forReadingAtPath: path) else { continue }
            defer { try? fh.close() }
            let end = (try? fh.seekToEnd()) ?? 0
            let start = tailOffsets[src] ?? 0
            // File truncated/rotated since last poll → start over from the top.
            let from = end < start ? 0 : start
            try? fh.seek(toOffset: from)
            let data = (try? fh.readToEnd()) ?? Data()
            tailOffsets[src] = (try? fh.offset()) ?? end
            guard !data.isEmpty else { continue }
            let chunk = String(decoding: data, as: UTF8.self)
            for raw in chunk.split(separator: "\n", omittingEmptySubsequences: true) {
                appendLine(source: src, raw: String(raw))
            }
        }
    }

    // MARK: Append + classify

    /// Must be called on `queue`.
    private func appendLine(source: VFLogSource, raw: String) {
        let text = raw.trimmingCharacters(in: CharacterSet(charactersIn: "\r\n"))
        if text.isEmpty { return }
        let level = Self.classify(text)
        let date = Date()
        let id = nextID
        nextID &+= 1

        // Persist app-sourced lines (child logs already persist to their own files).
        if source == .app, let fh = fileHandle {
            let entry = "[\(Self.fileFormatter.string(from: date))] \(text)\n"
            if let d = entry.data(using: .utf8) { try? fh.write(contentsOf: d) }
        }

        let line = VFLogLine(id: id, date: date, source: source, level: level, text: text)
        DispatchQueue.main.async {
            self.lines.append(line)
            if self.lines.count > self.maxLines {
                self.lines.removeFirst(self.lines.count - self.maxLines)
            }
        }
    }

    private static let errorKeys = ["error", "fatal", "panic", "exception",
                                    "traceback", "sigsegv", "sigabrt",
                                    "failed", "failure", "❌", "‼️"]
    private static let warnKeys = ["warning", "warn", "⚠️", "deprecat"]

    static func classify(_ text: String) -> VFLogLevel {
        let l = text.lowercased()
        if errorKeys.contains(where: l.contains) { return .error }
        if warnKeys.contains(where: l.contains) { return .warning }
        return .info
    }
}

// MARK: - Log console view

struct LogConsoleView: View {
    @ObservedObject private var store = LogStore.shared
    @Environment(\.dismiss) private var dismiss

    enum LevelFilter: String, CaseIterable, Identifiable {
        case all = "All", warnings = "Warnings+", errors = "Errors"
        var id: String { rawValue }
    }

    @State private var levelFilter: LevelFilter = .all
    @State private var sourceFilter: VFLogSource? = nil
    @State private var search = ""
    @State private var autoScroll = true
    /// -1 sentinel = "All".
    @State private var exportCount = 500
    private let exportOptions = [100, 500, 1000, 5000, -1]

    private static let rowFormatter: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "HH:mm:ss.SSS"; return f
    }()
    private static let stampFormatter: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"; return f
    }()
    private static let fileStamp: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd-HHmmss"; return f
    }()

    private var filtered: [VFLogLine] {
        store.lines.filter { line in
            switch levelFilter {
            case .all:      break
            case .warnings: if line.level == .info { return false }
            case .errors:   if line.level != .error { return false }
            }
            if let src = sourceFilter, line.source != src { return false }
            if !search.isEmpty, !line.text.localizedCaseInsensitiveContains(search) { return false }
            return true
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Color.white.opacity(0.08))
            controls
            logArea
            Divider().overlay(Color.white.opacity(0.08))
            exportBar
        }
        .frame(width: 880, height: 640)
        .background(VisualEffectBackground(material: .hudWindow, blending: .behindWindow).ignoresSafeArea())
        .preferredColorScheme(.dark)
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Diagnostics")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(.white)
                Text("Live logs from the app and speech/LLM engines — export to help diagnose issues")
                    .font(.system(size: 12))
                    .foregroundColor(.white.opacity(0.55))
            }
            Spacer()
            Button("Done") { dismiss() }
                .buttonStyle(VFOutlinePillStyle())
        }
        .padding(.horizontal, 20)
        .padding(.top, 18)
        .padding(.bottom, 14)
    }

    // MARK: Controls

    private var controls: some View {
        HStack(spacing: 10) {
            Picker("", selection: $levelFilter) {
                ForEach(LevelFilter.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(width: 220)
            .labelsHidden()

            Menu {
                Button("All sources") { sourceFilter = nil }
                Divider()
                ForEach(VFLogSource.allCases) { src in
                    Button(src.rawValue) { sourceFilter = src }
                }
            } label: {
                Label(sourceFilter?.rawValue ?? "All sources", systemImage: "line.3.horizontal.decrease.circle")
                    .font(.system(size: 12))
            }
            .menuStyle(.borderlessButton)
            .frame(width: 150)

            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.45))
                TextField("Filter", text: $search)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Capsule().fill(Color.white.opacity(0.06)))

            Spacer()

            Toggle(isOn: $autoScroll) {
                Text("Auto-scroll").font(.system(size: 12))
            }
            .toggleStyle(.switch)
            .controlSize(.small)

            Button { store.clear() } label: {
                Label("Clear", systemImage: "trash")
            }
            .buttonStyle(VFGhostPillStyle())
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    // MARK: Log area

    private var logArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    if filtered.isEmpty {
                        Text(store.lines.isEmpty
                             ? "Waiting for log output…"
                             : "No lines match the current filter.")
                            .font(.system(size: 12))
                            .foregroundColor(.white.opacity(0.4))
                            .padding(.top, 24)
                    } else {
                        ForEach(filtered) { row($0) }
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color.black.opacity(0.22))
            .onChange(of: store.lines.count) {
                if autoScroll {
                    withAnimation(.linear(duration: 0.1)) { proxy.scrollTo("bottom", anchor: .bottom) }
                }
            }
            .onAppear {
                if autoScroll { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }

    private func row(_ line: VFLogLine) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(Self.rowFormatter.string(from: line.date))
                .font(.system(size: 10.5, design: .monospaced))
                .foregroundColor(.white.opacity(0.32))
            Text(line.source.short)
                .font(.system(size: 10.5, weight: .semibold, design: .monospaced))
                .foregroundColor(line.source.tint)
                .frame(width: 66, alignment: .leading)
            Text(line.text)
                .font(.system(size: 11.5, design: .monospaced))
                .foregroundColor(line.level.color)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 1)
    }

    // MARK: Export bar

    private var exportBar: some View {
        HStack(spacing: 10) {
            Text("Last")
                .font(.system(size: 12))
                .foregroundColor(.white.opacity(0.6))
            Picker("", selection: $exportCount) {
                ForEach(exportOptions, id: \.self) { n in
                    Text(n == -1 ? "All" : "\(n)").tag(n)
                }
            }
            .labelsHidden()
            .frame(width: 90)
            Text("lines")
                .font(.system(size: 12))
                .foregroundColor(.white.opacity(0.6))

            Button { exportRecent(copy: false) } label: {
                Label("Export…", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(VFOutlinePillStyle())
            Button { exportRecent(copy: true) } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
            .buttonStyle(VFGhostPillStyle())

            Spacer()

            Button { exportProblems(copy: false) } label: {
                Label("Errors & Warnings", systemImage: "exclamationmark.triangle")
            }
            .buttonStyle(VFOutlinePillStyle())
            .tint(.orange)
            Button { exportProblems(copy: true) } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
            .buttonStyle(VFGhostPillStyle())
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    // MARK: Export actions

    private func exportRecent(copy: Bool) {
        let f = filtered
        let slice = (exportCount == -1 || exportCount >= f.count) ? f : Array(f.suffix(exportCount))
        deliver(render(slice), namePrefix: "voiceflow-logs", copy: copy)
    }

    private func exportProblems(copy: Bool) {
        let slice = store.lines.filter { $0.level != .info }
        deliver(render(slice), namePrefix: "voiceflow-errors", copy: copy)
    }

    private func render(_ ls: [VFLogLine]) -> String {
        let body = ls.map {
            "[\(Self.stampFormatter.string(from: $0.date))] [\($0.source.rawValue)] [\($0.level.rawValue.uppercased())] \($0.text)"
        }.joined(separator: "\n")
        let header = "VoiceFlow diagnostics export — \(ls.count) line(s)\n" +
                     "Generated \(Self.stampFormatter.string(from: Date()))\n" +
                     String(repeating: "-", count: 48) + "\n"
        return header + body + "\n"
    }

    private func deliver(_ text: String, namePrefix: String, copy: Bool) {
        if copy {
            let pb = NSPasteboard.general
            pb.clearContents()
            pb.setString(text, forType: .string)
            return
        }
        let panel = NSSavePanel()
        panel.title = "Export Logs"
        panel.nameFieldStringValue = "\(namePrefix)-\(Self.fileStamp.string(from: Date())).txt"
        panel.allowedContentTypes = [.plainText]
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
