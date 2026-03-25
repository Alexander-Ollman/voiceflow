import Foundation
import VoiceFlowFFI

/// Swift wrapper for VoiceFlow Rust library
@MainActor
final class VoiceFlowBridge: ObservableObject {
    private var handle: OpaquePointer?
    private var isCleanedUp = false
    var asrEngine: Qwen3ASREngine?

    @Published var isInitialized = false
    @Published var isProcessing = false
    @Published var lastError: String?
    @Published var lastResult: ProcessResult?
    @Published var isConsolidatedMode = false
    /// Whether the STT engine is external (Qwen3-ASR Python daemon) in traditional mode
    @Published var isExternalStt = false

    struct ProcessResult {
        let formattedText: String
        let rawTranscript: String
        let transcriptionMs: UInt64
        let llmMs: UInt64
        let totalMs: UInt64
    }

    /// Memory usage information from the Rust side
    struct MemoryUsage {
        let residentBytes: UInt
        let virtualBytes: UInt
        let peakBytes: UInt

        var residentMB: Double { Double(residentBytes) / 1024 / 1024 }
        var virtualMB: Double { Double(virtualBytes) / 1024 / 1024 }
        var peakMB: Double { Double(peakBytes) / 1024 / 1024 }
    }

    init() {
        // Initialize on background thread to avoid blocking
        Task.detached { [weak self] in
            await self?.initialize()
        }
    }

    deinit {
        // Perform FFI cleanup directly in deinit (thread-safe)
        // Note: can't update @Published properties here, just release resources
        if let handle = handle {
            voiceflow_unload_models(handle)
            voiceflow_prepare_shutdown()
            voiceflow_destroy(handle)
        }
    }

    /// Explicit cleanup - call before app termination for clean shutdown
    func cleanup() {
        guard !isCleanedUp else { return }
        isCleanedUp = true

        // Stop the Python ASR daemon if running
        asrEngine?.stopDaemon()
        asrEngine = nil

        if let handle = handle {
            // Unload models first to release memory
            voiceflow_unload_models(handle)

            // Prepare for shutdown
            voiceflow_prepare_shutdown()

            // Destroy the handle
            voiceflow_destroy(handle)
            self.handle = nil
        }

        isInitialized = false
    }

    /// Unload models from memory without destroying the handle
    /// Models will be reloaded on next use
    func unloadModels() {
        guard let handle = handle else { return }
        voiceflow_unload_models(handle)
    }

    /// Reset the LLM engine, allowing re-initialization on next use
    func resetLLM() {
        guard let handle = handle else { return }
        voiceflow_reset_llm(handle)
    }

    /// Unload the STT engine from memory (when streaming replaces it)
    func unloadStt() {
        guard let handle = handle else { return }
        voiceflow_unload_stt(handle)
    }

    /// Get current memory usage from the Rust side
    nonisolated static func getMemoryUsage() -> MemoryUsage {
        let info = voiceflow_memory_info()
        return MemoryUsage(
            residentBytes: info.resident_bytes,
            virtualBytes: info.virtual_bytes,
            peakBytes: info.peak_bytes
        )
    }

    /// Force garbage collection hint
    nonisolated static func forceGC() {
        voiceflow_force_gc()
    }

    /// Format text through the Rust LLM pipeline (for non-audio use cases like summarization)
    func formatText(_ text: String, context: String?) async -> String? {
        guard let handle = handle else { return nil }

        let result: VoiceFlowResult = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let ffiResult = text.withCString { textPtr in
                    if let ctx = context {
                        return ctx.withCString { ctxPtr in
                            voiceflow_format_text(handle, textPtr, ctxPtr)
                        }
                    } else {
                        return voiceflow_format_text(handle, textPtr, nil)
                    }
                }
                continuation.resume(returning: ffiResult)
            }
        }

        defer { voiceflow_free_result(result) }

        if result.success, let ptr = result.formatted_text {
            return String(cString: ptr)
        }
        return nil
    }

    /// Format text through the Rust LLM pipeline with real-time token streaming.
    /// Each token is delivered to `onToken` on the main actor as it's generated.
    /// Returns the fully post-processed final text (same as `formatText()`).
    func formatTextStreaming(
        _ text: String,
        context: String?,
        onToken: @escaping @MainActor (String) -> Void
    ) async -> String? {
        guard let handle = handle else { return nil }

        let result: VoiceFlowResult = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                // TokenReceiver bridges the C callback to Swift's main actor
                let receiver = TokenReceiver(onToken: onToken)
                let retainedReceiver = Unmanaged.passRetained(receiver)
                let userdata = retainedReceiver.toOpaque()

                let cCallback: TokenCallbackFn = { tokenPtr, ud in
                    guard let tokenPtr = tokenPtr, let ud = ud else { return true }
                    let token = String(cString: tokenPtr)
                    let receiver = Unmanaged<TokenReceiver>.fromOpaque(ud).takeUnretainedValue()
                    receiver.receive(token)
                    return true
                }

                let ffiResult = text.withCString { textPtr in
                    if let ctx = context {
                        return ctx.withCString { ctxPtr in
                            voiceflow_format_text_streaming(handle, textPtr, ctxPtr, cCallback, userdata)
                        }
                    } else {
                        return voiceflow_format_text_streaming(handle, textPtr, nil, cCallback, userdata)
                    }
                }

                // Balance the retain
                retainedReceiver.release()

                continuation.resume(returning: ffiResult)
            }
        }

        defer { voiceflow_free_result(result) }

        if result.success, let ptr = result.formatted_text {
            return String(cString: ptr)
        }
        return nil
    }

    private func initialize() async {
        let consolidatedMode = voiceflow_is_consolidated_mode()
        let externalStt = voiceflow_is_external_stt()

        // Initialize with default config
        let newHandle = voiceflow_init(nil)

        await MainActor.run {
            self.handle = newHandle
            self.isConsolidatedMode = consolidatedMode
            self.isExternalStt = externalStt
            self.isInitialized = newHandle != nil
            if newHandle == nil {
                self.lastError = "Failed to initialize VoiceFlow pipeline"
            }

            // Load persisted user corrections into the Rust pipeline
            if newHandle != nil {
                for pattern in CorrectionManager.shared.patterns {
                    self.addReplacement(original: pattern.original, corrected: pattern.corrected)
                }
            }

            // Start Qwen3-ASR daemon if needed (consolidated mode or external STT)
            let needsDaemon = consolidatedMode || externalStt
            if needsDaemon {
                self.asrEngine = Qwen3ASREngine()

                // Load ASR model in background
                if let engine = self.asrEngine {
                    Task {
                        if let modelIdPtr = voiceflow_current_consolidated_model() {
                            let modelId = String(cString: modelIdPtr)
                            voiceflow_free_string(modelIdPtr)

                            if let modelDirPtr = modelId.withCString({ cStr in
                                voiceflow_consolidated_model_dir(cStr)
                            }) {
                                let modelDir = String(cString: modelDirPtr)
                                voiceflow_free_string(modelDirPtr)

                                NSLog("[VoiceFlowBridge] Loading Qwen3-ASR model from: \(modelDir)")
                                await engine.loadModel(from: modelDir)
                            } else {
                                NSLog("[VoiceFlowBridge] Failed to get model dir for \(modelId)")
                            }
                        } else {
                            NSLog("[VoiceFlowBridge] No consolidated model configured")
                        }
                    }
                }
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

        // Consolidated mode: Qwen3-ASR + Rust post-processing only (no LLM)
        if isConsolidatedMode, let asrEngine = asrEngine {
            let asrStart = CFAbsoluteTimeGetCurrent()
            guard let rawText = await asrEngine.process(audio: audio) else {
                await MainActor.run {
                    self.lastError = "Qwen3-ASR transcription failed"
                }
                return nil
            }
            let asrMs = UInt64((CFAbsoluteTimeGetCurrent() - asrStart) * 1000)

            // Apply Rust post-processing (filler removal, tokenization fix, voice commands, replacements)
            let postProcessed: String = rawText.withCString { cStr in
                guard let resultPtr = voiceflow_post_process_text(cStr) else {
                    return rawText
                }
                let result = String(cString: resultPtr)
                voiceflow_free_string(resultPtr)
                return result
            }

            let processResult = ProcessResult(
                formattedText: postProcessed,
                rawTranscript: rawText,
                transcriptionMs: asrMs,
                llmMs: 0,
                totalMs: asrMs
            )

            await MainActor.run {
                self.lastResult = processResult
            }

            return processResult
        }

        // External STT mode (Qwen3-ASR in traditional pipeline): ASR daemon + LLM formatting
        if isExternalStt, let asrEngine = asrEngine {
            let asrStart = CFAbsoluteTimeGetCurrent()
            guard let rawText = await asrEngine.process(audio: audio) else {
                await MainActor.run {
                    self.lastError = "Qwen3-ASR transcription failed"
                }
                return nil
            }
            let asrMs = UInt64((CFAbsoluteTimeGetCurrent() - asrStart) * 1000)

            // Send raw text through Rust pipeline for post-processing + LLM formatting
            let result: VoiceFlowResult = await withCheckedContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    let ffiResult = rawText.withCString { textPtr in
                        if let ctx = context {
                            return ctx.withCString { ctxPtr in
                                voiceflow_format_text(handle, textPtr, ctxPtr)
                            }
                        } else {
                            return voiceflow_format_text(handle, textPtr, nil)
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
                    rawTranscript: rawText,
                    transcriptionMs: asrMs,
                    llmMs: result.llm_ms,
                    totalMs: asrMs + result.llm_ms
                )

                await MainActor.run {
                    self.lastResult = processResult
                }

                return processResult
            } else {
                let errorMsg = result.error_message.map { String(cString: $0) } ?? "LLM formatting failed"
                await MainActor.run {
                    self.lastError = errorMsg
                }
                return nil
            }
        }

        // Traditional mode (Whisper/Moonshine): call FFI pipeline on background thread
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

    /// Process pre-transcribed text with an image through multimodal LLM formatting.
    /// Used for visual context dictation (Control+Option+Space hotkey).
    func processTextWithImage(_ text: String, context: String?, imageData: Data) async -> ProcessResult? {
        guard let handle = handle else { return nil }

        await MainActor.run {
            self.isProcessing = true
            self.lastError = nil
        }

        defer {
            Task { @MainActor in
                self.isProcessing = false
            }
        }

        let result: VoiceFlowResult = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let ffiResult = imageData.withUnsafeBytes { imageBuffer in
                    let imagePtr = imageBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self)
                    let imageLen = UInt(imageBuffer.count)
                    return text.withCString { textPtr in
                        if let ctx = context {
                            return ctx.withCString { ctxPtr in
                                voiceflow_process_text_with_image(handle, textPtr, ctxPtr, imagePtr, imageLen)
                            }
                        } else {
                            return voiceflow_process_text_with_image(handle, textPtr, nil, imagePtr, imageLen)
                        }
                    }
                }
                continuation.resume(returning: ffiResult)
            }
        }

        defer { voiceflow_free_result(result) }

        if result.success {
            let processResult = ProcessResult(
                formattedText: result.formatted_text.map { String(cString: $0) } ?? "",
                rawTranscript: result.raw_transcript.map { String(cString: $0) } ?? text,
                transcriptionMs: result.transcription_ms,
                llmMs: result.llm_ms,
                totalMs: result.total_ms
            )

            await MainActor.run {
                self.lastResult = processResult
            }
            return processResult
        } else {
            let errorMsg = result.error_message.map { String(cString: $0) } ?? "Multimodal processing failed"
            await MainActor.run {
                self.lastError = errorMsg
            }
            return nil
        }
    }

    /// Add a user-learned correction to the Rust replacement dictionary.
    /// Applied deterministically during pipeline processing.
    func addReplacement(original: String, corrected: String) {
        guard let handle = handle else { return }
        original.withCString { origPtr in
            corrected.withCString { corrPtr in
                _ = voiceflow_add_replacement(handle, origPtr, corrPtr)
            }
        }
    }

    /// Remove a user-learned correction from the Rust replacement dictionary.
    /// Takes effect immediately for subsequent pipeline runs.
    func removeReplacement(original: String) {
        guard let handle = handle else { return }
        original.withCString { origPtr in
            _ = voiceflow_remove_replacement(handle, origPtr)
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

/// Bridges C token callbacks (on background thread) to Swift's MainActor.
/// Retained via Unmanaged during the FFI call lifetime.
private final class TokenReceiver: @unchecked Sendable {
    private let onToken: @MainActor (String) -> Void

    init(onToken: @escaping @MainActor (String) -> Void) {
        self.onToken = onToken
    }

    func receive(_ token: String) {
        let callback = self.onToken
        DispatchQueue.main.async {
            // We're on main thread now, safe to call @MainActor closure
            // Use assumeIsolated to satisfy the compiler
            MainActor.assumeIsolated {
                callback(token)
            }
        }
    }
}
