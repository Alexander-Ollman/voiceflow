import Foundation
import MoonshineVoice

/// Wraps the Moonshine streaming API for real-time speech-to-text.
/// Audio chunks from AudioRecorder are fed in; partial transcript text is published for overlay display.
@MainActor
final class MoonshineStreamingEngine: ObservableObject {
    private var transcriber: Transcriber?
    private var stream: MoonshineVoice.Stream?

    @Published var partialText: String = ""
    @Published var isLoaded = false
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0

    /// Accumulated lines for the current session (partial + completed)
    private var sessionLines: [UInt64: String] = [:]

    enum StreamingModelSize: String, CaseIterable, Identifiable {
        case tiny = "tiny"
        case small = "small"
        case medium = "medium"

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .tiny: return "Tiny"
            case .small: return "Small"
            case .medium: return "Medium"
            }
        }

        var description: String {
            switch self {
            case .tiny: return "34M params — fastest, ~50 MB"
            case .small: return "123M params — good balance, ~160 MB"
            case .medium: return "245M params — best accuracy, ~290 MB"
            }
        }

        var modelArch: ModelArch {
            switch self {
            case .tiny: return .tinyStreaming
            case .small: return .smallStreaming
            case .medium: return .mediumStreaming
            }
        }

        /// Integer value for the Python download CLI (--model-arch)
        var modelArchInt: Int {
            switch self {
            case .tiny: return 2
            case .small: return 4
            case .medium: return 5
            }
        }

        /// Directory name for the downloaded model files
        var dirName: String {
            switch self {
            case .tiny: return "moonshine-tiny-streaming"
            case .small: return "moonshine-small-streaming"
            case .medium: return "moonshine-medium-streaming"
            }
        }

        var downloadSizeMB: Int {
            switch self {
            case .tiny: return 50
            case .small: return 160
            case .medium: return 290
            }
        }
    }

    // MARK: - Model Paths

    /// Base directory for Moonshine model storage
    static var modelsBaseDir: String {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("VoiceFlow/moonshine-models").path
    }

    /// Full path for a specific model size
    static func modelPath(for size: StreamingModelSize) -> String {
        return (modelsBaseDir as NSString).appendingPathComponent(size.dirName)
    }

    /// Check if a model is downloaded
    static func isModelDownloaded(_ size: StreamingModelSize) -> Bool {
        return FileManager.default.fileExists(atPath: modelPath(for: size))
    }

    // MARK: - Model Loading

    /// Load a Moonshine streaming model from disk
    func loadModel(size: StreamingModelSize) throws {
        let path = Self.modelPath(for: size)
        guard FileManager.default.fileExists(atPath: path) else {
            throw NSError(domain: "MoonshineStreamingEngine", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Model not found at \(path). Download it first."])
        }

        // Unload any existing model
        unloadModel()

        NSLog("[MoonshineStreaming] Loading model: %@ from %@", size.rawValue, path)

        let newTranscriber = try Transcriber(
            modelPath: path,
            modelArch: size.modelArch,
            options: [
                TranscriberOption(name: "vad_threshold", value: "0.5"),
            ]
        )

        self.transcriber = newTranscriber
        self.isLoaded = true
        NSLog("[MoonshineStreaming] Model loaded successfully: %@", size.rawValue)
    }

    // MARK: - Streaming Session

    /// Start a new streaming session (call before feeding chunks)
    func beginSession() throws {
        guard let transcriber = transcriber else {
            throw NSError(domain: "MoonshineStreamingEngine", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "No model loaded"])
        }

        // Clean up any previous stream
        endSessionSilently()

        sessionLines.removeAll()
        partialText = ""

        let newStream = try transcriber.createStream(updateInterval: 0.5)

        // Add event listener for real-time updates
        newStream.addListener(StreamEventListener(engine: self))

        try newStream.start()
        self.stream = newStream
        NSLog("[MoonshineStreaming] Session started")
    }

    /// Feed a chunk of 16kHz mono float audio
    func feedAudioChunk(_ samples: [Float]) {
        guard let stream = stream else { return }
        do {
            try stream.addAudio(samples, sampleRate: 16000)
        } catch {
            NSLog("[MoonshineStreaming] Error feeding audio: %@", error.localizedDescription)
        }
    }

    /// End the streaming session, return final transcript
    func endSession() -> String {
        guard let stream = stream else { return partialText }

        do {
            // Final update to flush remaining audio
            let finalTranscript = try stream.updateTranscription()
            for line in finalTranscript.lines {
                sessionLines[line.lineId] = line.text
            }
        } catch {
            NSLog("[MoonshineStreaming] Error on final update: %@", error.localizedDescription)
        }

        do {
            try stream.stop()
        } catch {
            NSLog("[MoonshineStreaming] Error stopping stream: %@", error.localizedDescription)
        }
        stream.close()
        self.stream = nil

        // Build final transcript from all accumulated lines
        let finalText = sessionLines.sorted(by: { $0.key < $1.key })
            .map { $0.value }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        NSLog("[MoonshineStreaming] Session ended, transcript: %@", String(finalText.prefix(100)))

        let result = finalText.isEmpty ? partialText : finalText
        sessionLines.removeAll()
        partialText = ""
        return result
    }

    /// End session without returning transcript (cleanup)
    private func endSessionSilently() {
        guard let stream = stream else { return }
        do { try stream.stop() } catch {}
        stream.close()
        self.stream = nil
    }

    /// Unload model from memory
    func unloadModel() {
        endSessionSilently()
        transcriber?.close()
        transcriber = nil
        isLoaded = false
        partialText = ""
        sessionLines.removeAll()
        NSLog("[MoonshineStreaming] Model unloaded")
    }

    // MARK: - Event Handling

    /// Called from the event listener when transcript text changes
    func handleTranscriptEvent(_ event: TranscriptEvent) {
        if let textChanged = event as? LineTextChanged {
            sessionLines[textChanged.line.lineId] = textChanged.line.text
            updatePartialText()
        } else if let completed = event as? LineCompleted {
            sessionLines[completed.line.lineId] = completed.line.text
            updatePartialText()
        } else if let started = event as? LineStarted {
            sessionLines[started.line.lineId] = started.line.text
            updatePartialText()
        }
    }

    private func updatePartialText() {
        let text = sessionLines.sorted(by: { $0.key < $1.key })
            .map { $0.value }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        partialText = text
    }

    // MARK: - Model Download

    /// Download a model using the moonshine_voice Python package
    func downloadModel(size: StreamingModelSize) async throws {
        await MainActor.run {
            self.isDownloading = true
            self.downloadProgress = 0
        }

        defer {
            Task { @MainActor in
                self.isDownloading = false
            }
        }

        let modelDir = Self.modelPath(for: size)
        let baseDir = Self.modelsBaseDir

        // Ensure base directory exists
        try FileManager.default.createDirectory(atPath: baseDir, withIntermediateDirectories: true)

        // Use Python moonshine_voice.download to fetch the model
        // This requires moonshine-voice pip package installed
        NSLog("[MoonshineStreaming] Starting download for model: %@ (arch %d)", size.rawValue, size.modelArchInt)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", "-m", "moonshine_voice.download",
                             "--language", "en",
                             "--model-arch", "\(size.modelArchInt)"]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()

        // Run the blocking wait on a background thread
        let (terminationStatus, output) = await withCheckedContinuation { (continuation: CheckedContinuation<(Int32, String), Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                process.waitUntilExit()
                let outputData = pipe.fileHandleForReading.readDataToEndOfFile()
                let outputStr = String(data: outputData, encoding: .utf8) ?? ""
                continuation.resume(returning: (process.terminationStatus, outputStr))
            }
        }

        guard terminationStatus == 0 else {
            throw NSError(domain: "MoonshineStreamingEngine", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Model download failed: \(output)"])
        }

        // The download command prints the model path — find it
        // Typical output: "Downloaded model path: /Users/.../base-en"
        var downloadedPath: String?
        for line in output.components(separatedBy: "\n") {
            if line.contains("Downloaded model path:") || line.contains("model path:") {
                downloadedPath = line.components(separatedBy: ": ").last?.trimmingCharacters(in: .whitespacesAndNewlines)
            } else if line.contains("Model path:") {
                downloadedPath = line.components(separatedBy: ": ").last?.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        // If we found a downloaded path, symlink or copy to our expected location
        if let sourcePath = downloadedPath, FileManager.default.fileExists(atPath: sourcePath) {
            // Remove existing directory if any
            if FileManager.default.fileExists(atPath: modelDir) {
                try FileManager.default.removeItem(atPath: modelDir)
            }
            // Create symlink to avoid duplicating large model files
            try FileManager.default.createSymbolicLink(atPath: modelDir, withDestinationPath: sourcePath)
            NSLog("[MoonshineStreaming] Linked model: %@ → %@", modelDir, sourcePath)
        } else {
            // Fallback: check if the model ended up in the default cache location
            let cacheDir = NSHomeDirectory() + "/Library/Caches/moonshine_voice"
            let possiblePaths = [
                cacheDir + "/download.moonshine.ai/model/\(size.rawValue)-streaming-en/quantized",
                cacheDir + "/download.moonshine.ai/model/\(size.rawValue)-streaming-en/quantized/\(size.rawValue)-streaming-en",
            ]
            for path in possiblePaths {
                if FileManager.default.fileExists(atPath: path) {
                    if FileManager.default.fileExists(atPath: modelDir) {
                        try FileManager.default.removeItem(atPath: modelDir)
                    }
                    try FileManager.default.createSymbolicLink(atPath: modelDir, withDestinationPath: path)
                    NSLog("[MoonshineStreaming] Linked model from cache: %@ → %@", modelDir, path)
                    break
                }
            }
        }

        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw NSError(domain: "MoonshineStreamingEngine", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Model download completed but files not found. Output: \(output)"])
        }

        await MainActor.run {
            self.downloadProgress = 1.0
        }

        NSLog("[MoonshineStreaming] Download complete for model: %@", size.rawValue)
    }
}

// MARK: - Stream Event Listener

/// Bridges Moonshine stream events to the engine on the main actor
private final class StreamEventListener: TranscriptEventListener, @unchecked Sendable {
    private weak var engine: MoonshineStreamingEngine?

    init(engine: MoonshineStreamingEngine) {
        self.engine = engine
    }

    func onLineStarted(_ event: LineStarted) {
        let capturedEvent = event as TranscriptEvent
        Task { @MainActor [weak engine] in
            engine?.handleTranscriptEvent(capturedEvent)
        }
    }

    func onLineTextChanged(_ event: LineTextChanged) {
        let capturedEvent = event as TranscriptEvent
        Task { @MainActor [weak engine] in
            engine?.handleTranscriptEvent(capturedEvent)
        }
    }

    func onLineCompleted(_ event: LineCompleted) {
        let capturedEvent = event as TranscriptEvent
        Task { @MainActor [weak engine] in
            engine?.handleTranscriptEvent(capturedEvent)
        }
    }

    func onLineUpdated(_ event: LineUpdated) {
        let capturedEvent = event as TranscriptEvent
        Task { @MainActor [weak engine] in
            engine?.handleTranscriptEvent(capturedEvent)
        }
    }

    func onError(_ event: TranscriptError) {
        NSLog("[MoonshineStreaming] Stream error: %@", event.error.localizedDescription)
    }
}
