import Foundation

/// Real Qwen3-ASR engine that manages a Python daemon process and communicates
/// over a Unix domain socket using length-prefixed JSON (4-byte big-endian + JSON).
@MainActor
final class Qwen3ASREngine: ObservableObject {
    enum EngineState: Equatable {
        case unloaded
        case starting
        case loadingModel
        case ready
        case error(String)
    }

    @Published var state: EngineState = .unloaded
    @Published var modelId: String = "qwen3-asr-0.6b"

    private static let socketPath = "/tmp/voiceflow_qwen3_asr_daemon.sock"
    private static let pidFile = "/tmp/voiceflow_qwen3_asr_daemon.pid"
    private static let daemonStartTimeout: TimeInterval = 10
    private static let daemonShutdownTimeout: TimeInterval = 2

    private var daemonProcess: Process?

    // MARK: - Public API

    /// Load the model via the Python daemon.
    /// Ensures the daemon is running, then sends a preload command.
    func loadModel(from modelDir: String) async {
        state = .starting

        do {
            try await ensureDaemonRunning()

            state = .loadingModel
            let response = try await sendCommand([
                "command": "preload",
                "model_path": modelDir,
            ])

            if let status = response["status"] as? String, status == "ok" {
                state = .ready
            } else {
                let msg = response["message"] as? String ?? "Unknown preload error"
                state = .error(msg)
            }
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    /// Process audio samples and return transcribed text.
    /// Encodes Float32 PCM as 16-bit WAV, base64-encodes, sends to daemon.
    func process(audio: [Float]) async -> String? {
        guard case .ready = state else { return nil }

        do {
            let wavData = encodeAsWAV(audio: audio, sampleRate: 16000)
            let b64 = wavData.base64EncodedString()

            let response = try await sendCommand([
                "command": "transcribe",
                "audio": b64,
            ])

            if let status = response["status"] as? String, status == "ok",
               let text = response["text"] as? String {
                return text
            } else {
                let msg = response["message"] as? String ?? "Transcription failed"
                NSLog("[Qwen3ASR] Transcription error: \(msg)")
                return nil
            }
        } catch {
            NSLog("[Qwen3ASR] Process error: \(error)")
            return nil
        }
    }

    /// Unload the model from the daemon (keeps daemon alive).
    func unload() {
        state = .unloaded
        Task.detached {
            _ = try? await self.sendCommand(["command": "unload"])
        }
    }

    // MARK: - VLM Support

    /// Load a VLM model via the Python daemon.
    func loadVlmModel(from modelDir: String, modelType: String) async -> Bool {
        do {
            try await ensureDaemonRunning()

            let response = try await sendCommand([
                "command": "preload_vlm",
                "model_path": modelDir,
                "model_type": modelType,
            ])

            if let status = response["status"] as? String, status == "ok" {
                NSLog("[Qwen3ASR] VLM loaded from: \(modelDir)")
                return true
            } else {
                let msg = response["message"] as? String ?? "Unknown VLM load error"
                NSLog("[Qwen3ASR] VLM load failed: \(msg)")
                return false
            }
        } catch {
            NSLog("[Qwen3ASR] VLM load error: \(error)")
            return false
        }
    }

    /// Analyze an image using the loaded VLM.
    /// Returns a description string, or nil on failure.
    func analyzeImage(imageData: Data, prompt: String? = nil) async -> String? {
        do {
            let b64 = imageData.base64EncodedString()
            var command: [String: Any] = [
                "command": "analyze_image",
                "image": b64,
            ]
            if let prompt = prompt {
                command["prompt"] = prompt
            }

            let response = try await sendCommand(command)

            if let status = response["status"] as? String, status == "ok",
               let description = response["description"] as? String {
                return description
            } else {
                let msg = response["message"] as? String ?? "VLM analysis failed"
                NSLog("[Qwen3ASR] VLM analysis error: \(msg)")
                return nil
            }
        } catch {
            NSLog("[Qwen3ASR] VLM analysis error: \(error)")
            return nil
        }
    }

    /// Unload the VLM from the daemon (keeps daemon and ASR model alive).
    func unloadVlm() {
        Task.detached {
            _ = try? await self.sendCommand(["command": "unload_vlm"])
        }
    }

    /// Shut down the Python daemon process.
    func stopDaemon() {
        // Send shutdown command
        let shutdownDone = DispatchSemaphore(value: 0)
        Task.detached {
            _ = try? await self.sendCommand(["command": "shutdown"])
            shutdownDone.signal()
        }

        // Wait briefly for graceful shutdown
        let deadline = DispatchTime.now() + Self.daemonShutdownTimeout
        _ = shutdownDone.wait(timeout: deadline)

        // Force-kill if still alive
        if let process = daemonProcess, process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
        daemonProcess = nil

        // Don't delete socket here — the daemon's cleanup() handles it
        // and checks PID ownership to avoid races with a new daemon.

        state = .unloaded
    }

    // MARK: - Daemon Lifecycle

    /// Ensure the daemon process is running and accepting connections.
    private func ensureDaemonRunning() async throws {
        // Check if daemon is already alive
        if await isDaemonAlive() {
            return
        }

        // Spawn the daemon
        try spawnDaemon()

        // Wait for daemon to become responsive
        let deadline = Date().addingTimeInterval(Self.daemonStartTimeout)
        while Date() < deadline {
            try await Task.sleep(nanoseconds: 200_000_000) // 200ms
            if await isDaemonAlive() {
                return
            }
        }

        throw Qwen3ASRError.daemonStartTimeout
    }

    /// Check if the daemon is alive by sending a check command.
    private nonisolated func isDaemonAlive() async -> Bool {
        guard FileManager.default.fileExists(atPath: Self.socketPath) else {
            return false
        }
        do {
            let response = try await sendCommand(["command": "check"])
            return (response["status"] as? String) == "ok"
        } catch {
            return false
        }
    }

    /// Spawn the Python daemon process.
    private func spawnDaemon() throws {
        let pythonPath = findPython()
        let scriptPath = try findDaemonScript()

        NSLog("[Qwen3ASR] Spawning daemon: \(pythonPath) \(scriptPath)")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [scriptPath]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "CUDA_VISIBLE_DEVICES": "",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        ]) { _, new in new }

        // Redirect stdout/stderr to log file
        let logFile = FileHandle(forWritingAtPath: "/tmp/voiceflow_daemon.log")
            ?? FileHandle.nullDevice
        process.standardOutput = logFile
        process.standardError = logFile

        try process.run()
        daemonProcess = process
        NSLog("[Qwen3ASR] Daemon spawned with PID \(process.processIdentifier)")
    }

    /// Find a suitable Python 3 interpreter.
    private func findPython() -> String {
        // 1. Check VOICEFLOW_PYTHON env var
        if let envPython = ProcessInfo.processInfo.environment["VOICEFLOW_PYTHON"],
           FileManager.default.isExecutableFile(atPath: envPython) {
            return envPython
        }

        // 2. VoiceFlow venv (has torch, qwen-asr, etc.)
        let home = NSHomeDirectory()
        let venvPython = "\(home)/.local/share/voiceflow/python-env/bin/python3"
        if FileManager.default.isExecutableFile(atPath: venvPython) {
            return venvPython
        }

        // 3. Homebrew default
        let homebrewPython = "/opt/homebrew/bin/python3"
        if FileManager.default.isExecutableFile(atPath: homebrewPython) {
            return homebrewPython
        }

        // 4. System / PATH fallback
        return "/usr/bin/env"
    }

    /// Find the daemon script, preferring the app bundle Resources.
    private func findDaemonScript() throws -> String {
        // 1. App bundle Resources
        if let bundlePath = Bundle.main.path(
            forResource: "qwen3_asr_daemon", ofType: "py"
        ) {
            return bundlePath
        }

        // 2. Development fallback: relative to executable
        let executableDir = Bundle.main.executableURL?
            .deletingLastPathComponent().path ?? ""
        let devPaths = [
            // When running from VoiceFlowApp/.build/release/
            executableDir + "/../../../scripts/qwen3_asr_daemon.py",
            // From repo root
            executableDir + "/../../../../scripts/qwen3_asr_daemon.py",
        ]
        for path in devPaths {
            let resolved = (path as NSString).standardizingPath
            if FileManager.default.fileExists(atPath: resolved) {
                return resolved
            }
        }

        // 3. Absolute fallback for common dev layout
        let home = NSHomeDirectory()
        let repoPath = "\(home)/voiceflow/scripts/qwen3_asr_daemon.py"
        if FileManager.default.fileExists(atPath: repoPath) {
            return repoPath
        }

        throw Qwen3ASRError.daemonScriptNotFound
    }

    // MARK: - Unix Socket IPC

    /// Send a command to the daemon over the Unix socket and return the response.
    @discardableResult
    private nonisolated func sendCommand(
        _ command: [String: Any]
    ) async throws -> [String: Any] {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try Self.sendCommandSync(command)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Synchronous socket send/receive (called on background thread).
    private nonisolated static func sendCommandSync(
        _ command: [String: Any]
    ) throws -> [String: Any] {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw Qwen3ASRError.socketError("Failed to create socket")
        }
        defer { close(fd) }

        // Set send/recv timeout (120 seconds - model loading and transcription can be slow)
        var timeout = timeval(tv_sec: 120, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

        // Connect to Unix socket
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        let sunPathSize = MemoryLayout.size(ofValue: addr.sun_path)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: sunPathSize) { dest in
                pathBytes.withUnsafeBufferPointer { src in
                    let count = min(src.count, sunPathSize)
                    dest.update(from: src.baseAddress!, count: count)
                }
            }
        }

        let connectResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.connect(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connectResult == 0 else {
            throw Qwen3ASRError.socketError("Failed to connect: \(String(cString: strerror(errno)))")
        }

        // Send length-prefixed JSON
        let jsonData = try JSONSerialization.data(
            withJSONObject: command, options: []
        )
        var length = UInt32(jsonData.count).bigEndian
        let lengthData = Data(bytes: &length, count: 4)

        try sendAll(fd: fd, data: lengthData)
        try sendAll(fd: fd, data: jsonData)

        // Receive length-prefixed JSON response
        let respLengthData = try recvExact(fd: fd, count: 4)
        let respLength = respLengthData.withUnsafeBytes {
            UInt32(bigEndian: $0.load(as: UInt32.self))
        }
        guard respLength > 0, respLength < 100 * 1024 * 1024 else {
            throw Qwen3ASRError.socketError("Invalid response length: \(respLength)")
        }

        let respData = try recvExact(fd: fd, count: Int(respLength))
        guard let response = try JSONSerialization.jsonObject(
            with: respData, options: []
        ) as? [String: Any] else {
            throw Qwen3ASRError.socketError("Invalid JSON response")
        }

        return response
    }

    /// Send all bytes over a file descriptor.
    private nonisolated static func sendAll(fd: Int32, data: Data) throws {
        try data.withUnsafeBytes { buffer in
            var sent = 0
            let total = buffer.count
            let baseAddress = buffer.baseAddress!
            while sent < total {
                let n = Darwin.send(
                    fd,
                    baseAddress.advanced(by: sent),
                    total - sent,
                    0
                )
                guard n > 0 else {
                    throw Qwen3ASRError.socketError("Send failed: \(String(cString: strerror(errno)))")
                }
                sent += n
            }
        }
    }

    /// Receive exactly `count` bytes from a file descriptor.
    private nonisolated static func recvExact(fd: Int32, count: Int) throws -> Data {
        var data = Data(count: count)
        var received = 0
        try data.withUnsafeMutableBytes { buffer in
            let baseAddress = buffer.baseAddress!
            while received < count {
                let n = Darwin.recv(
                    fd,
                    baseAddress.advanced(by: received),
                    count - received,
                    0
                )
                guard n > 0 else {
                    throw Qwen3ASRError.socketError("Recv failed: \(String(cString: strerror(errno)))")
                }
                received += n
            }
        }
        return data
    }

    // MARK: - WAV Encoding

    /// Encode Float32 PCM samples as a 16-bit WAV file in memory.
    private func encodeAsWAV(audio: [Float], sampleRate: Int) -> Data {
        let numSamples = audio.count
        let bitsPerSample: Int = 16
        let numChannels: Int = 1
        let byteRate = sampleRate * numChannels * (bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = numSamples * (bitsPerSample / 8)
        let fileSize = 36 + dataSize

        var data = Data(capacity: 44 + dataSize)

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(fileSize).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt sub-chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })    // sub-chunk size
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })     // PCM format
        data.append(contentsOf: withUnsafeBytes(of: UInt16(numChannels).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(blockAlign).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(bitsPerSample).littleEndian) { Array($0) })

        // data sub-chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })

        // Convert Float32 to Int16 samples
        for sample in audio {
            let clamped = max(-1.0, min(1.0, sample))
            let intSample = Int16(clamped * 32767.0)
            data.append(contentsOf: withUnsafeBytes(of: intSample.littleEndian) { Array($0) })
        }

        return data
    }
}

// MARK: - Errors

enum Qwen3ASRError: LocalizedError {
    case daemonStartTimeout
    case daemonScriptNotFound
    case socketError(String)

    var errorDescription: String? {
        switch self {
        case .daemonStartTimeout:
            return "Qwen3-ASR daemon failed to start within timeout"
        case .daemonScriptNotFound:
            return "qwen3_asr_daemon.py not found in bundle or development paths"
        case .socketError(let msg):
            return "Socket error: \(msg)"
        }
    }
}
