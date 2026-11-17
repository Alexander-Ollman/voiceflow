import Foundation
import Darwin

/// Manages the lifecycle of a local llama-server (PrismML fork) serving Bonsai-8B.
/// Spawns on app launch with VoiceFlow's chosen flags, monitors liveness, restarts
/// if it dies, and shuts down cleanly on app termination.
///
/// We need PrismML's fork specifically — Bonsai-8B uses Q1_0 quantization (ggml type
/// 41) which upstream llama.cpp doesn't yet support.
///
/// Context size is set to 16384 (vs the previous default of 4096) to comfortably
/// fit base prompt + persona + transcript + headroom for a few hundred output
/// tokens without `truncated = 1` cutoffs.
@MainActor
final class LlamaServerManager: ObservableObject {

    enum State: Equatable {
        case stopped
        case launching
        case ready
        case crashed(String)
        case missingBinary(String)  // path checked
    }

    @Published var state: State = .stopped
    @Published var lastError: String?

    private var childWatchdog: ChildProcessWatchdog?
    private var lifecycleID = UUID()
    private var process: Process?
    private var watchdogTimer: Timer?

    // MARK: - Config

    /// Default context size — large enough for the base prompt (~3000 tokens) +
    /// persona (~300) + long transcript (~500) + comfortable output budget (~12000).
    static let contextSize: Int = 16384
    static let port: Int = 8080
    static let host: String = "127.0.0.1"
    static let logPath: String = "/tmp/llama_server_bonsai.log"

    /// Hugging Face filename — matches SetupHelper.bonsaiFilename.
    private var bonsaiPath: URL { SetupHelper.bonsaiPath }

    /// Search paths for the PrismML llama-server binary, in priority order.
    private static var binarySearchPaths: [String] {
        let home = NSHomeDirectory()
        return [
            // 1. Bundled at Contents/MacOS/llama-server (production path).
            Bundle.main.bundleURL
                .appendingPathComponent("Contents/MacOS/llama-server").path,
            // 2. Explicit env override (dev iteration on a fresh PrismML build).
            ProcessInfo.processInfo.environment["VOICEFLOW_LLAMA_SERVER"] ?? "",
            // 3. PrismML build at the conventional dev path
            "\(home)/PrismML-llama.cpp/build/bin/llama-server",
            // 4. Sibling-checkout style
            "\(home)/dev/PrismML-llama.cpp/build/bin/llama-server",
        ].filter { !$0.isEmpty }
    }

    /// Find the llama-server binary if any of the search paths point to a real file.
    static func locateBinary() -> URL? {
        for path in binarySearchPaths {
            if FileManager.default.isExecutableFile(atPath: path) {
                return URL(fileURLWithPath: path)
            }
        }
        return nil
    }

    static var binaryAvailable: Bool {
        locateBinary() != nil
    }

    // MARK: - Lifecycle

    /// Start the server if not already running. Idempotent — if a server is already
    /// listening on our port (e.g. user started it manually), we adopt rather than
    /// spawn a duplicate.
    func start() async {
        let attempt = lifecycleID
        if state == .ready, let p = process, p.isRunning {
            return
        }

        // Something already on our port: only reuse it if it is a VoiceFlow
        // server that another running VoiceFlow owns. A VoiceFlow server whose
        // app has exited is an orphan: stop it and start our own. Anything else
        // is another program; never format with someone else's model.
        if let listener = await Self.portListener() {
            guard lifecycleID == attempt else { return }
            switch Self.classify(listener) {
            case .sharedVoiceFlow:
                NSLog("[LlamaServer] Reusing VoiceFlow server PID %d owned by running VoiceFlow PID %d",
                      listener, Self.parentPID(of: listener) ?? -1)
                state = .ready
                return
            case .orphanedVoiceFlow:
                NSLog("[LlamaServer] Stopping orphaned VoiceFlow server PID %d on :%d", listener, Self.port)
                await Self.terminateOrphan(listener)
                guard lifecycleID == attempt else { return }
            case .foreign(let name):
                lastError = "Port \(Self.port) is in use by \(name) (PID \(listener)). Quit it, then restart VoiceFlow's language model."
                NSLog("[LlamaServer] %@", lastError ?? "")
                state = .crashed(lastError!)
                return
            }
        }

        guard lifecycleID == attempt else { return }
        guard FileManager.default.fileExists(atPath: bonsaiPath.path) else {
            lastError = "Bonsai GGUF not found at \(bonsaiPath.path). Run Setup first."
            state = .crashed(lastError!)
            return
        }

        guard let binary = Self.locateBinary() else {
            let searched = Self.binarySearchPaths.joined(separator: "\n  - ")
            lastError = """
            Language model server binary not found. Searched:
              - \(searched)

            This usually means VoiceFlow.app is corrupted or was extracted
            incorrectly. Re-download and reinstall from the official DMG.
            """
            state = .missingBinary(searched)
            NSLog("[LlamaServer] %@", lastError ?? "")
            return
        }

        state = .launching
        let p = Process()
        p.executableURL = binary
        p.arguments = [
            "-m", bonsaiPath.path,
            "--port", String(Self.port),
            "--host", Self.host,
            "-ngl", "99",
            "-c", String(Self.contextSize),
            "--no-mmproj",
            // Q8_0 KV cache: halves the per-token KV memory cost relative to
            // FP16 with no measurable quality drop on non-reasoning chat
            // models. At 16K context that's ~1.1 GB of headroom recovered.
            //   FP16:  16384 tok × 144 KB = 2.3 GB
            //   Q8_0:  16384 tok ×  72 KB = 1.15 GB
            // The cached prompt prefix (~4,370 tokens) still sits warm; only
            // the storage format of each cached K/V vector changes.
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
        ]

        // Truncate the log file at startup so the request counter starts fresh.
        FileManager.default.createFile(atPath: Self.logPath, contents: nil)
        let logHandle = FileHandle(forWritingAtPath: Self.logPath) ?? FileHandle.nullDevice
        p.standardOutput = logHandle
        p.standardError = logHandle

        // Watch for crashes — if the process exits before we set .stopped, mark it.
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in
                guard let self = self else { return }
                if self.state != .stopped {
                    let code = proc.terminationStatus
                    self.state = .crashed("Server exited with code \(code)")
                    NSLog("[LlamaServer] Server crashed with exit code %d", code)
                }
            }
        }

        do {
            try p.run()
            childWatchdog = ChildProcessWatchdog.start(childPID: p.processIdentifier, role: "llama-server")
            process = p
            NSLog("[LlamaServer] Spawned PID %d (model %@, ctx %d)",
                  p.processIdentifier, bonsaiPath.lastPathComponent, Self.contextSize)
        } catch {
            state = .crashed("Failed to spawn: \(error.localizedDescription)")
            return
        }

        // Wait for /v1/models to respond — typically <1s on warm cache, <5s cold.
        let readyDeadline = Date().addingTimeInterval(45)
        while Date() < readyDeadline {
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard lifecycleID == attempt else { return }
            if await isPortServing() {
                guard lifecycleID == attempt else { return }
                state = .ready
                return
            }
            if let p = process, !p.isRunning {
                state = .crashed("Server exited before becoming ready")
                return
            }
        }
        state = .crashed("Server didn't respond on port \(Self.port) within 45s")
    }

    func stop() async {
        lifecycleID = UUID()
        watchdogTimer?.invalidate()
        watchdogTimer = nil
        if await ManagedProcessShutdown.stop(process) {
            childWatchdog?.stop()
            childWatchdog = nil
        } else {
            // Leave the watchdog armed: it terminates the server once VoiceFlow
            // exits. Disarming here is how a live helper becomes an orphan.
            NSLog("[LlamaServer] Owned server did not exit within shutdown deadline; watchdog stays armed")
        }
        process = nil
        state = .stopped
    }

    // MARK: - Port ownership

    enum PortListenerKind: Equatable {
        case sharedVoiceFlow          // a VoiceFlow server whose VoiceFlow is still running
        case orphanedVoiceFlow        // a VoiceFlow server whose VoiceFlow has exited
        case foreign(String)          // anything else
    }

    /// PID listening on our TCP port, if any.
    static func portListener() async -> pid_t? {
        await Task.detached(priority: .userInitiated) { () -> pid_t? in
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
            task.arguments = ["-nP", "-t", "-iTCP:\(port)", "-sTCP:LISTEN"]
            let out = Pipe()
            task.standardOutput = out
            task.standardError = FileHandle.nullDevice
            guard (try? task.run()) != nil else { return nil }
            task.waitUntilExit()
            let text = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            return text.split(whereSeparator: \.isNewline).compactMap { pid_t($0) }.first
        }.value
    }

    static func executablePath(of pid: pid_t) -> String? {
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN) * 4)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        return length > 0 ? String(cString: buffer) : nil
    }

    static func parentPID(of pid: pid_t) -> pid_t? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.size)
        guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else { return nil }
        return pid_t(info.pbi_ppid)
    }

    /// A VoiceFlow server is the llama-server bundled inside a VoiceFlow app.
    /// It is shared when its parent is a running VoiceFlow, orphaned when its
    /// parent is gone (re-parented to launchd).
    static func classify(_ pid: pid_t) -> PortListenerKind {
        let path = executablePath(of: pid) ?? ""
        guard path.hasSuffix(".app/Contents/MacOS/llama-server"), path.contains("VoiceFlow") else {
            return .foreign(path.isEmpty ? "another program" : (path as NSString).lastPathComponent)
        }
        guard let parent = parentPID(of: pid), parent > 1,
              let parentPath = executablePath(of: parent),
              parentPath.hasSuffix(".app/Contents/MacOS/VoiceFlow") else {
            return .orphanedVoiceFlow
        }
        return .sharedVoiceFlow
    }

    /// Stop an orphaned VoiceFlow server we did not spawn: TERM, then KILL.
    static func terminateOrphan(_ pid: pid_t) async {
        kill(pid, SIGTERM)
        for _ in 0..<30 {
            if kill(pid, 0) != 0 { return }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        kill(pid, SIGKILL)
        for _ in 0..<20 {
            if kill(pid, 0) != 0 { return }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    /// Quick health probe — hit /v1/models and check for HTTP 200.
    private func isPortServing() async -> Bool {
        guard let url = URL(string: "http://\(Self.host):\(Self.port)/v1/models") else {
            return false
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1.5
        do {
            let (_, response) = try await URLSession.shared.data(for: req)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }
}
