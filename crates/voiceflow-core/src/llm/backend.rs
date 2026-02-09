//! LLM backend trait for unified interface across inference engines

use anyhow::Result;

/// Trait for LLM inference backends
/// Allows switching between mistral.rs and llama.cpp transparently
pub trait LlmBackendTrait: Send + Sync {
    /// Format a transcript using the LLM
    ///
    /// # Arguments
    /// * `transcript` - The raw transcript to format
    /// * `prompt` - The full formatted prompt including system instructions
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0-2.0)
    /// * `top_p` - Top-p sampling parameter (0.0-1.0)
    fn generate(&self, prompt: &str, max_tokens: u32, temperature: f32, top_p: f32) -> Result<String>;

    /// Get the backend name for logging
    fn name(&self) -> &'static str;
}
