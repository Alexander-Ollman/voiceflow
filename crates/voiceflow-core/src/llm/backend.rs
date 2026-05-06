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
}
