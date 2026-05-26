import Foundation

/// Parakeet-MLX engine: spawns a Python daemon (parakeet_asr_daemon.py) and
/// communicates over a Unix domain socket using the same length-prefixed JSON
/// protocol as Qwen3ASREngine. Batch transcription only — no streaming partials.
@MainActor
final class ParakeetASREngine: ObservableObject {
    enum EngineState: Equatable {
        case unloaded
        case starting
        case loadingModel
        case ready
        case error(String)
    }

    @Published var state: EngineState = .unloaded
    @Published var modelId: String = "mlx-community/parakeet-tdt-0.6b-v2"

    private static let daemonStartTimeout: TimeInterval = 10
    private static let daemonShutdownTimeout: TimeInterval = 2

    private var daemonProcess: Process?

    // MARK: - Public API

    func loadModel(modelId: String? = nil) async {
        let target = modelId ?? self.modelId
        state = .starting
        do {
            try await ensureDaemonRunning()
            state = .loadingModel
            let response = try await sendCommand([
                "command": "preload",
                "model_id": target,
            ])
            if (response["status"] as? String) == "ok" {
                self.modelId = target
                state = .ready
            } else {
                state = .error(response["message"] as? String ?? "Unknown preload error")
            }
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    /// Transcribe Float32 PCM audio (16kHz). Returns the transcript text.
    func transcribe(audio: [Float]) async -> String? {
        guard case .ready = state else {
            NSLog("[Parakeet] transcribe called but state is \(state)")
            return nil
        }
        do {
            let wav = encodeAsWAV(audio: audio, sampleRate: 16000)
            let response = try await sendCommand([
                "command": "transcribe",
                "audio": wav.base64EncodedString(),
            ])
            if (response["status"] as? String) == "ok",
               let text = response["text"] as? String {
                if let elapsed = response["elapsed_ms"] as? Int {
                    NSLog("[Parakeet] Transcribed in %dms: %@", elapsed,
                          String(text.prefix(80)))
                }
                return text
            }
            NSLog("[Parakeet] Transcription error: %@",
                  response["message"] as? String ?? "unknown")
            return nil
        } catch {
            NSLog("[Parakeet] transcribe error: %@", error.localizedDescription)
            return nil
        }
    }

    func unload() {
        state = .unloaded
        Task.detached {
            _ = try? await self.sendCommand(["command": "unload"])
        }
    }

    func stopDaemon() {
        let done = DispatchSemaphore(value: 0)
        Task.detached {
            _ = try? await self.sendCommand(["command": "shutdown"])
            done.signal()
        }
        _ = done.wait(timeout: .now() + Self.daemonShutdownTimeout)
        if let p = daemonProcess, p.isRunning {
            p.terminate()
            p.waitUntilExit()
        }
        daemonProcess = nil
        state = .unloaded
    }

    // MARK: - Daemon lifecycle

    private func ensureDaemonRunning() async throws {
        if await isDaemonAlive() { return }
        try spawnDaemon()
        let deadline = Date().addingTimeInterval(Self.daemonStartTimeout)
        while Date() < deadline {
            try await Task.sleep(nanoseconds: 200_000_000)
            if await isDaemonAlive() { return }
        }
        throw ParakeetASRError.daemonStartTimeout
    }

    private nonisolated func isDaemonAlive() async -> Bool {
        guard FileManager.default.fileExists(atPath: ParakeetDaemonConstants.socketPath) else {
            return false
        }
        do {
            let response = try await sendCommand(["command": "check"])
            return (response["status"] as? String) == "ok"
        } catch {
            return false
        }
    }

    private func spawnDaemon() throws {
        let process = Process()
        process.environment = ProcessInfo.processInfo.environment.merging([
            "CUDA_VISIBLE_DEVICES": "",
        ]) { _, new in new }

        if let bundledDaemon = findBundledDaemon() {
            NSLog("[Parakeet] Spawning bundled daemon: %@", bundledDaemon)
            process.executableURL = URL(fileURLWithPath: bundledDaemon)
        } else {
            let pythonPath = findPython()
            let scriptPath = try findDaemonScript()
            NSLog("[Parakeet] Spawning daemon (dev fallback): %@ %@", pythonPath, scriptPath)
            process.executableURL = URL(fileURLWithPath: pythonPath)
            process.arguments = [scriptPath]
        }

        let logFile = FileHandle(forWritingAtPath: "/tmp/voiceflow_parakeet_daemon.log")
            ?? FileHandle.nullDevice
        process.standardOutput = logFile
        process.standardError = logFile

        try process.run()
        daemonProcess = process
        NSLog("[Parakeet] Daemon spawned with PID %d", process.processIdentifier)
    }

    /// Production path: PyInstaller-frozen daemon shipped inside the .app bundle.
    /// Lives under Resources/ (not MacOS/) so codesign doesn't treat PyInstaller's
    /// nested directories as unsigned sub-bundles. See build.sh for context.
    /// Returns the path if present and executable, nil otherwise.
    private func findBundledDaemon() -> String? {
        let path = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Resources/parakeet-daemon/parakeet-daemon")
            .path
        return FileManager.default.isExecutableFile(atPath: path) ? path : nil
    }

    private func findPython() -> String {
        if let envPython = ProcessInfo.processInfo.environment["VOICEFLOW_PYTHON"],
           FileManager.default.isExecutableFile(atPath: envPython) {
            return envPython
        }
        let home = NSHomeDirectory()
        let venvPython = "\(home)/.local/share/voiceflow/python-env/bin/python3"
        if FileManager.default.isExecutableFile(atPath: venvPython) {
            return venvPython
        }
        if FileManager.default.isExecutableFile(atPath: "/opt/homebrew/bin/python3") {
            return "/opt/homebrew/bin/python3"
        }
        return "/usr/bin/env"
    }

    private func findDaemonScript() throws -> String {
        if let bundlePath = Bundle.main.path(
            forResource: "parakeet_asr_daemon", ofType: "py"
        ) {
            return bundlePath
        }
        let executableDir = Bundle.main.executableURL?
            .deletingLastPathComponent().path ?? ""
        let devPaths = [
            executableDir + "/../../../scripts/parakeet_asr_daemon.py",
            executableDir + "/../../../../scripts/parakeet_asr_daemon.py",
        ]
        for path in devPaths {
            let resolved = (path as NSString).standardizingPath
            if FileManager.default.fileExists(atPath: resolved) {
                return resolved
            }
        }
        let home = NSHomeDirectory()
        let repoPath = "\(home)/voiceflow/scripts/parakeet_asr_daemon.py"
        if FileManager.default.fileExists(atPath: repoPath) {
            return repoPath
        }
        throw ParakeetASRError.daemonScriptNotFound
    }

    // MARK: - Unix socket IPC

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

    private nonisolated static func sendCommandSync(
        _ command: [String: Any]
    ) throws -> [String: Any] {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw ParakeetASRError.socketError("Failed to create socket")
        }
        defer { close(fd) }

        var timeout = timeval(tv_sec: ParakeetDaemonConstants.socketTimeoutSeconds, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = ParakeetDaemonConstants.socketPath.utf8CString
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
            throw ParakeetASRError.socketError(
                "Failed to connect: \(String(cString: strerror(errno)))"
            )
        }

        let jsonData = try JSONSerialization.data(withJSONObject: command)
        var length = UInt32(jsonData.count).bigEndian
        let lengthData = Data(bytes: &length, count: 4)

        try sendAll(fd: fd, data: lengthData)
        try sendAll(fd: fd, data: jsonData)

        let respLengthData = try recvExact(fd: fd, count: 4)
        let respLength = respLengthData.withUnsafeBytes {
            UInt32(bigEndian: $0.load(as: UInt32.self))
        }
        guard respLength > 0, respLength < 100 * 1024 * 1024 else {
            throw ParakeetASRError.socketError("Invalid response length: \(respLength)")
        }

        let respData = try recvExact(fd: fd, count: Int(respLength))
        guard let response = try JSONSerialization.jsonObject(with: respData) as? [String: Any] else {
            throw ParakeetASRError.socketError("Invalid JSON response")
        }
        return response
    }

    private nonisolated static func sendAll(fd: Int32, data: Data) throws {
        try data.withUnsafeBytes { buffer in
            var sent = 0
            let total = buffer.count
            let baseAddress = buffer.baseAddress!
            while sent < total {
                let n = Darwin.send(fd, baseAddress.advanced(by: sent), total - sent, 0)
                guard n > 0 else {
                    throw ParakeetASRError.socketError(
                        "Send failed: \(String(cString: strerror(errno)))"
                    )
                }
                sent += n
            }
        }
    }

    private nonisolated static func recvExact(fd: Int32, count: Int) throws -> Data {
        var data = Data(count: count)
        var received = 0
        try data.withUnsafeMutableBytes { buffer in
            let baseAddress = buffer.baseAddress!
            while received < count {
                let n = Darwin.recv(fd, baseAddress.advanced(by: received), count - received, 0)
                guard n > 0 else {
                    throw ParakeetASRError.socketError(
                        "Recv failed: \(String(cString: strerror(errno)))"
                    )
                }
                received += n
            }
        }
        return data
    }

    // MARK: - WAV encoding

    private func encodeAsWAV(audio: [Float], sampleRate: Int) -> Data {
        let numSamples = audio.count
        let bitsPerSample = 16
        let numChannels = 1
        let byteRate = sampleRate * numChannels * (bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = numSamples * (bitsPerSample / 8)
        let fileSize = 36 + dataSize

        var data = Data(capacity: 44 + dataSize)
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(fileSize).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(numChannels).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(blockAlign).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(bitsPerSample).littleEndian) { Array($0) })
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })

        for sample in audio {
            let clamped = max(-1.0, min(1.0, sample))
            let intSample = Int16(clamped * 32767.0)
            data.append(contentsOf: withUnsafeBytes(of: intSample.littleEndian) { Array($0) })
        }
        return data
    }
}

// MARK: - Constants (nonisolated so they can be used from background threads)

private enum ParakeetDaemonConstants {
    static let socketPath = "/tmp/voiceflow_parakeet_daemon.sock"
    // Model load can pull ~600MB on first run; allow a long socket timeout.
    static let socketTimeoutSeconds: Int = 300
}

// MARK: - Errors

enum ParakeetASRError: LocalizedError {
    case daemonStartTimeout
    case daemonScriptNotFound
    case socketError(String)

    var errorDescription: String? {
        switch self {
        case .daemonStartTimeout:
            return "Parakeet daemon failed to start within timeout"
        case .daemonScriptNotFound:
            return "parakeet_asr_daemon.py not found"
        case .socketError(let msg):
            return "Parakeet socket error: \(msg)"
        }
    }
}
