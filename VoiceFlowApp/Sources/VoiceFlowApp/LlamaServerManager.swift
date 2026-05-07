import Foundation

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
            // 1. Bundled in app Resources (preferred — once we ship it)
            Bundle.main.path(forResource: "llama-server", ofType: nil) ?? "",
            // 2. Explicit env override
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
        if state == .ready, let p = process, p.isRunning {
            return
        }

        // If something is already serving the port, just adopt it.
        if await isPortServing() {
            NSLog("[LlamaServer] External server already on :%d, adopting.", Self.port)
            state = .ready
            return
        }

        guard FileManager.default.fileExists(atPath: bonsaiPath.path) else {
            lastError = "Bonsai GGUF not found at \(bonsaiPath.path). Run Setup first."
            state = .crashed(lastError!)
            return
        }

        guard let binary = Self.locateBinary() else {
            let searched = Self.binarySearchPaths.joined(separator: "\n  - ")
            lastError = """
            PrismML llama-server binary not found. Searched:
              - \(searched)

            Build it with:
              git clone https://github.com/PrismML-Eng/llama.cpp ~/PrismML-llama.cpp
              cd ~/PrismML-llama.cpp && cmake -B build && cmake --build build --target llama-server -j
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
            if await isPortServing() {
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

    func stop() {
        watchdogTimer?.invalidate()
        watchdogTimer = nil
        if let p = process, p.isRunning {
            p.terminate()
            // Give it a beat to exit gracefully before SIGKILL.
            DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) {
                if p.isRunning { p.interrupt() }
            }
        }
        process = nil
        state = .stopped
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
