//! LLM backend trait for unified interface across inference engines

use anyhow::Result;

/// Callback for receiving streamed tokens. Return `true` to continue, `false` to abort.
pub type TokenCallback<'a> = &'a mut dyn FnMut(&str) -> bool;

/// Trait for LLM inference backends
/// Allows switching between mistral.rs and llama.cpp transparently
pub trait LlmBackendTrait: Send + Sync {
    /// Format a transcript using the LLM
    ///
    /// # Arguments
    /// * `prompt` - The full formatted prompt including system instructions
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0-2.0)
    /// * `top_p` - Top-p sampling parameter (0.0-1.0)
    fn generate(&self, prompt: &str, max_tokens: u32, temperature: f32, top_p: f32) -> Result<String>;

    /// Get the backend name for logging
    fn name(&self) -> &'static str;

    /// Generate with per-token streaming callback.
    /// Default implementation falls back to non-streaming `generate()`.
    fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        on_token: TokenCallback<'_>,
    ) -> Result<String> {
        let _ = on_token;
        self.generate(prompt, max_tokens, temperature, top_p)
    }

    /// Whether this backend supports real-time token streaming
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Generate text with an image (multimodal)
    fn generate_with_image(
        &self,
        _prompt: &str,
        _image: &[u8],
        _max_tokens: u32,
        _temperature: f32,
        _top_p: f32,
    ) -> Result<String> {
        anyhow::bail!("Multimodal not supported by this backend")
    }

    /// Generate text with an image, streaming tokens
    fn generate_with_image_streaming(
        &self,
        _prompt: &str,
        _image: &[u8],
        _max_tokens: u32,
        _temperature: f32,
        _top_p: f32,
        _on_token: TokenCallback<'_>,
    ) -> Result<String> {
        anyhow::bail!("Multimodal streaming not supported by this backend")
    }

    /// Whether this backend supports multimodal inference
    fn supports_multimodal(&self) -> bool {
        false
    }

    /// Generate text directly from audio (audio-to-text, no separate STT)
    ///
    /// # Arguments
    /// * `audio_pcm` - PCM audio samples (f32, 16kHz)
    /// * `sample_rate` - Sample rate of the audio (typically 16000)
    /// * `prompt` - System/user prompt for the generation
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `top_p` - Top-p sampling parameter
    fn generate_from_audio(
        &self,
        _audio_pcm: &[f32],
        _sample_rate: u32,
        _prompt: &str,
        _max_tokens: u32,
        _temperature: f32,
        _top_p: f32,
    ) -> Result<String> {
        anyhow::bail!("Audio-direct generation not supported by this backend")
    }

    /// Generate text from audio with per-token streaming
    fn generate_from_audio_streaming(
        &self,
        _audio_pcm: &[f32],
        _sample_rate: u32,
        _prompt: &str,
        _max_tokens: u32,
        _temperature: f32,
        _top_p: f32,
        _on_token: TokenCallback<'_>,
    ) -> Result<String> {
        anyhow::bail!("Audio-direct streaming not supported by this backend")
    }

    /// Whether this backend supports direct audio-to-text generation
    fn supports_audio_direct(&self) -> bool {
        false
    }

    /// Generate output constrained to a JSON schema.
    ///
    /// The returned string is guaranteed to parse as JSON conforming to `schema`
    /// when the backend declares `supports_structured() == true`. Backends that
    /// don't support schema constraints either return an error or fall back to
    /// best-effort prompt-only JSON output.
    ///
    /// Used by AI commands and retroactive correction where the caller needs a
    /// strict shape (e.g. `{action, anchor, replacement, confidence}`) and
    /// can't afford to parse free prose.
    fn generate_structured(
        &self,
        _prompt: &str,
        _schema: &serde_json::Value,
        _max_tokens: u32,
        _temperature: f32,
        _top_p: f32,
    ) -> Result<String> {
        anyhow::bail!("Structured generation not supported by this backend")
    }

    /// Whether this backend can enforce a JSON schema on generation.
    fn supports_structured(&self) -> bool {
        false
    }

    /// Generate using the server's chat-completions endpoint with a separate
    /// system + user message. Lets llama-server apply the model's native chat
    /// template instead of relying on our manual `<|im_start|>` wrapping.
    /// Essential for chat-tuned models like Bonsai where the bare /v1/completions
    /// path produces "Human:/AI:" style rambling.
    ///
    /// `stop` is an optional list of stop sequences the server should respect.
    fn generate_chat(
        &self,
        _system: &str,
        _user: &str,
        _max_tokens: u32,
        _temperature: f32,
        _top_p: f32,
        _stop: &[&str],
    ) -> Result<String> {
        anyhow::bail!("Chat-completions generation not supported by this backend")
    }

    /// Whether this backend supports the chat-completions path.
    fn supports_chat(&self) -> bool {
        false
    }
}
