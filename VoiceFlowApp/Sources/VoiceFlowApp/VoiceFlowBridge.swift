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
    /// Whether the pipeline is in audio-direct mode (Gemma 4 handles STT + formatting)
    @Published var isAudioDirect = false

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

    /// Format text through the deterministic normalization pipeline (no LLM).
    /// Instant — applies rule-based transforms only.
    func formatTextDeterministic(_ text: String) async -> String? {
        guard let handle = handle else { return nil }

        let result: VoiceFlowResult = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let ffiResult = text.withCString { textPtr in
                    voiceflow_format_text_deterministic(handle, textPtr)
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

    /// Process audio directly through an audio-native model (Gemma 4).
    /// Bypasses separate STT — the LLM handles transcription and formatting in one pass.
    func processAudioDirect(audio: [Float], context: String? = nil) async -> ProcessResult? {
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

        let result: VoiceFlowResult = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let ffiResult = audio.withUnsafeBufferPointer { buffer in
                    let count = UInt(buffer.count)
                    if let ctx = context {
                        return ctx.withCString { ctxPtr in
                            voiceflow_process_audio_direct(handle, buffer.baseAddress, count, ctxPtr)
                        }
                    } else {
                        return voiceflow_process_audio_direct(handle, buffer.baseAddress, count, nil)
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

    /// Format text through the Rust LLM pipeline with real-time token streaming.
    /// Each token is delivered to `onToken` on the main actor as it's generated.
    /// Returns the fully post-processed final text (same as `formatText()`).
    ///
    /// Pass `cancellation` to enable mid-stream abort — when cancelled, the
    /// SSE loop terminates on the next token and llama-server frees its slot.
    /// This is essential for live re-formatting where new Parakeet partials
    /// supersede in-flight LLM streams.
    func formatTextStreaming(
        _ text: String,
        context: String?,
        cancellation: LLMStreamCancellation? = nil,
        onToken: @escaping @MainActor (String) -> Void
    ) async -> String? {
        guard let handle = handle else { return nil }

        let result: VoiceFlowResult = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                // TokenReceiver bridges the C callback to Swift's main actor
                let receiver = TokenReceiver(onToken: onToken, cancellation: cancellation)
                let retainedReceiver = Unmanaged.passRetained(receiver)
                let userdata = retainedReceiver.toOpaque()

                let cCallback: TokenCallbackFn = { tokenPtr, ud in
                    guard let tokenPtr = tokenPtr, let ud = ud else { return true }
                    let token = String(cString: tokenPtr)
                    let receiver = Unmanaged<TokenReceiver>.fromOpaque(ud).takeUnretainedValue()
                    return receiver.receive(token)
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
        let audioDirect = voiceflow_is_audio_direct_mode()

        // Initialize with default config
        let newHandle = voiceflow_init(nil)

        await MainActor.run {
            self.handle = newHandle
            self.isConsolidatedMode = consolidatedMode
            self.isExternalStt = externalStt
            self.isAudioDirect = audioDirect
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

            // Start Qwen3-ASR daemon if needed (consolidated mode or external STT, but NOT audio-direct).
            // Skip entirely when Parakeet is the active Swift-side STT — Parakeet handles transcription
            // through its own daemon (parakeet_asr_daemon); the Rust pipeline only does LLM formatting,
            // so the qwen daemon would just idle on ~750MB of RAM loading a model we never call.
            let parakeetActive = ProcessInfo.processInfo.environment["VOICEFLOW_USE_PARAKEET"] != "0"
            let needsDaemon = !audioDirect && !parakeetActive && (consolidatedMode || externalStt)
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

    // MARK: - Intent classification (no LLM, ~µs)

    /// Classify the intent of a freshly-transcribed utterance.
    /// Returns nil if the FFI call fails or the result can't be decoded.
    nonisolated static func classifyIntent(_ text: String) -> IntentResult? {
        guard let jsonPtr = text.withCString({ voiceflow_classify_intent($0) }) else {
            return nil
        }
        defer { voiceflow_free_string(jsonPtr) }
        let json = String(cString: jsonPtr)
        guard let data = json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(IntentResult.self, from: data)
    }

    // MARK: - Retroactive correction (LLM)

    /// Ask the LLM to produce a structured edit for a retroactive correction.
    /// Latency: ~150–400ms depending on model + context length. Off the main thread.
    func retroactiveCorrect(_ input: RetroactiveInput) async -> Edit? {
        guard let handle = handle else { return nil }
        guard let inputJSON = (try? JSONEncoder().encode(input)).flatMap({ String(data: $0, encoding: .utf8) }) else {
            return nil
        }

        let result: String? = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let ptr = inputJSON.withCString { voiceflow_retroactive_correct(handle, $0) }
                guard let ptr = ptr else {
                    continuation.resume(returning: nil)
                    return
                }
                defer { voiceflow_free_string(ptr) }
                continuation.resume(returning: String(cString: ptr))
            }
        }

        guard let result = result, let data = result.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(Edit.self, from: data)
    }

    /// Ask Bonsai whether the latest utterance is a redo (replace) of the
    /// previous dictated output. ~150–400ms; runs off the main thread.
    func assessRedo(_ input: RedoInput) async -> RedoDecision? {
        guard let handle = handle else { return nil }
        guard let inputJSON = (try? JSONEncoder().encode(input)).flatMap({ String(data: $0, encoding: .utf8) }) else {
            return nil
        }

        let result: String? = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let ptr = inputJSON.withCString { voiceflow_assess_redo(handle, $0) }
                guard let ptr = ptr else {
                    continuation.resume(returning: nil)
                    return
                }
                defer { voiceflow_free_string(ptr) }
                continuation.resume(returning: String(cString: ptr))
            }
        }

        guard let result = result, let data = result.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(RedoDecision.self, from: data)
    }

    // MARK: - AI voice commands

    /// Run an AI voice command. ~300ms–1.5s depending on output length.
    func runCommand(_ input: CommandInput) async -> CommandOutput? {
        guard let handle = handle else { return nil }
        guard let inputJSON = (try? JSONEncoder().encode(input))
            .flatMap({ String(data: $0, encoding: .utf8) })
        else { return nil }

        let result: String? = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let ptr = inputJSON.withCString { voiceflow_run_ai_command(handle, $0) }
                guard let ptr = ptr else {
                    continuation.resume(returning: nil)
                    return
                }
                defer { voiceflow_free_string(ptr) }
                continuation.resume(returning: String(cString: ptr))
            }
        }

        guard let result = result, let data = result.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(CommandOutput.self, from: data)
    }
}

/// Cancellation handle for a `formatTextStreaming` call. The caller can
/// `cancel()` at any time; the next token from llama-server triggers a `false`
/// return from the C callback, which the SSE reader in voiceflow-core checks
/// to break its read loop. The HTTP body closes, llama-server frees the slot.
///
/// Used to abort an in-flight Bonsai stream when a fresher Parakeet partial
/// arrives during live re-formatting — otherwise the single llama-server slot
/// would backlog with stale generations.
final class LLMStreamCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
}

/// Bridges C token callbacks (on background thread) to Swift's MainActor.
/// Retained via Unmanaged during the FFI call lifetime.
private final class TokenReceiver: @unchecked Sendable {
    private let onToken: @MainActor (String) -> Void
    private let cancellation: LLMStreamCancellation?

    init(onToken: @escaping @MainActor (String) -> Void,
         cancellation: LLMStreamCancellation? = nil) {
        self.onToken = onToken
        self.cancellation = cancellation
    }

    /// Returns `false` if the caller has cancelled, telling the C side to
    /// break the SSE loop and close the connection.
    func receive(_ token: String) -> Bool {
        if cancellation?.isCancelled == true { return false }
        let callback = self.onToken
        DispatchQueue.main.async {
            MainActor.assumeIsolated {
                callback(token)
            }
        }
        return true
    }
}
