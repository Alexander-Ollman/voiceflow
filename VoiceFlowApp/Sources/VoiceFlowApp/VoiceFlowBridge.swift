import Foundation
import VoiceFlowFFI

/// Swift wrapper for VoiceFlow Rust library
@MainActor
final class VoiceFlowBridge: ObservableObject {
    private var handle: OpaquePointer?

    @Published var isInitialized = false
    @Published var isProcessing = false
    @Published var lastError: String?
    @Published var lastResult: ProcessResult?

    struct ProcessResult {
        let formattedText: String
        let rawTranscript: String
        let transcriptionMs: UInt64
        let llmMs: UInt64
        let totalMs: UInt64
    }

    init() {
        // Initialize on background thread to avoid blocking
        Task.detached { [weak self] in
            await self?.initialize()
        }
    }

    deinit {
        if let handle = handle {
            voiceflow_destroy(handle)
        }
    }

    private func initialize() async {
        // Initialize with default config
        let newHandle = voiceflow_init(nil)

        await MainActor.run {
            self.handle = newHandle
            self.isInitialized = newHandle != nil
            if newHandle == nil {
                self.lastError = "Failed to initialize VoiceFlow pipeline"
            }
        }
    }

    /// Process audio samples (16kHz mono float PCM)
    func process(audio: [Float], context: String? = nil) async -> ProcessResult? {
        guard let handle = handle else {
            await MainActor.run {
                self.lastError = "Pipeline not initialized"
            }
            return nil
        }

        await MainActor.run {
            self.isProcessing = true
            self.lastError = nil
        }

        defer {
            Task { @MainActor in
                self.isProcessing = false
            }
        }

        // Call FFI on background thread
        let result: VoiceFlowResult = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let ffiResult = audio.withUnsafeBufferPointer { buffer in
                    let count = UInt(buffer.count)
                    if let ctx = context {
                        return ctx.withCString { ctxPtr in
                            voiceflow_process(handle, buffer.baseAddress, count, ctxPtr)
                        }
                    } else {
                        return voiceflow_process(handle, buffer.baseAddress, count, nil)
                    }
                }
                continuation.resume(returning: ffiResult)
            }
        }

        defer {
            voiceflow_free_result(result)
        }

        if result.success {
            let processResult = ProcessResult(
                formattedText: result.formatted_text.map { String(cString: $0) } ?? "",
                rawTranscript: result.raw_transcript.map { String(cString: $0) } ?? "",
                transcriptionMs: result.transcription_ms,
                llmMs: result.llm_ms,
                totalMs: result.total_ms
            )

            await MainActor.run {
                self.lastResult = processResult
            }

            return processResult
        } else {
            let errorMsg = result.error_message.map { String(cString: $0) } ?? "Unknown error"
            await MainActor.run {
                self.lastError = errorMsg
            }
            return nil
        }
    }

    /// Get library version
    nonisolated static var version: String {
        guard let versionPtr = voiceflow_version() else {
            return "unknown"
        }
        return String(cString: versionPtr)
    }
}
