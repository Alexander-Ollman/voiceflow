//! LLM engine with switchable backends
//!
//! Supports multiple inference backends:
//! - mistral.rs: Good for Qwen, Gemma2, Phi-2
//! - llama.cpp: Supports all GGUF architectures (SmolLM3, Gemma3n, Phi-4, etc.)

use crate::config::{Config, LlmBackend};
use crate::llm::backend::LlmBackendTrait;
use crate::llm::llamacpp_backend::LlamaCppBackend;
use crate::llm::mistralrs_backend::MistralRsBackend;
use crate::llm::prompts::{format_prompt, post_process_output};
use crate::runtime;
use anyhow::{Context, Result};

/// LLM engine for text formatting with switchable backends
pub struct LlmEngine {
    backend: Box<dyn LlmBackendTrait>,
    config: Config,
}

impl LlmEngine {
    /// Create a new LLM engine with the given configuration (async)
    ///
    /// Automatically selects the appropriate backend based on model architecture:
    /// - mistral.rs for Qwen, Gemma2, Phi-2
    /// - llama.cpp for SmolLM3, Gemma3n, Phi-4, and other architectures
    pub async fn new_async(config: &Config) -> Result<Self> {
        let model_path = config.llm_model_path()?;

        if !model_path.exists() {
            anyhow::bail!(
                "LLM model not found at {:?}. Run 'voiceflow setup' to download models.",
                model_path
            );
        }

        let selected_backend = config.llm_model.backend();
        tracing::info!(
            "Loading LLM model {} with {} backend",
            config.llm_model.display_name(),
            selected_backend.display_name()
        );

        // Create the appropriate backend based on model architecture
        let backend: Box<dyn LlmBackendTrait> = match selected_backend {
            LlmBackend::MistralRs => {
                let b = MistralRsBackend::new_async(config)
                    .await
                    .context("Failed to initialize mistral.rs backend")?;
                Box::new(b)
            }
            LlmBackend::LlamaCpp => {
                let b = LlamaCppBackend::new(config)
                    .context("Failed to initialize llama.cpp backend")?;
                Box::new(b)
            }
        };

        tracing::info!(
            "LLM engine ready: {} via {}",
            config.llm_model.display_name(),
            backend.name()
        );

        Ok(Self {
            backend,
            config: config.clone(),
        })
    }

    /// Create a new LLM engine (blocking wrapper for sync contexts)
    ///
    /// Uses the global shared Tokio runtime to avoid memory leaks from
    /// creating multiple runtimes.
    pub fn new(config: &Config) -> Result<Self> {
        let config = config.clone();
        runtime::block_on(Self::new_async(&config))
    }

    /// Format a transcript using the LLM (async)
    pub async fn format_async(&self, transcript: &str, prompt_template: &str) -> Result<String> {
        let prompt = format_prompt(prompt_template, transcript, &self.config);

        tracing::debug!("LLM prompt length: {} chars", prompt.len());

        // Generate using the backend
        let output = self.backend.generate(
            &prompt,
            self.config.llm_options.max_tokens,
            self.config.llm_options.temperature,
            self.config.llm_options.top_p,
        )?;

        // Strip thinking tags and apply post-processing
        let output = strip_thinking_tags(&output);

        tracing::debug!("LLM output length: {} chars", output.len());

        Ok(output)
    }

    /// Format a transcript using the LLM (blocking wrapper)
    ///
    /// Uses the global shared Tokio runtime to avoid memory leaks from
    /// creating multiple runtimes per call.
    pub fn format(&self, transcript: &str, prompt_template: &str) -> Result<String> {
        runtime::block_on(self.format_async(transcript, prompt_template))
    }

    /// Get the name of the active backend
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }
}

/// Detect available hardware acceleration
pub fn detect_hardware() -> &'static str {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "Metal (Apple Silicon)"
    }
    #[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
    {
        "CPU (Intel Mac)"
    }
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        "CUDA (NVIDIA GPU)"
    }
    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    {
        "CPU (Linux)"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "CPU"
    }
}

/// Strip <think>...</think> tags from model output and apply post-processing
fn strip_thinking_tags(text: &str) -> String {
    // Remove <think>...</think> blocks (including empty ones)
    let mut result = text.to_string();

    // Handle <think>content</think> pattern
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            let end_tag_end = end + "</think>".len();
            result = format!(
                "{}{}",
                &result[..start],
                result[end_tag_end..].trim_start()
            );
        } else {
            break;
        }
    }

    // Apply shared post-processing (punctuation spacing, UI quoting, ellipsis normalization)
    post_process_output(&result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for strip_thinking_tags (which now uses post_process_output internally)
    #[test]
    fn test_strip_thinking_tags_empty() {
        assert_eq!(strip_thinking_tags("Hello world"), "Hello world");
    }

    #[test]
    fn test_strip_thinking_tags_with_content() {
        assert_eq!(
            strip_thinking_tags("<think>some reasoning</think>Hello world"),
            "Hello world"
        );
    }

    #[test]
    fn test_strip_thinking_tags_multiple() {
        assert_eq!(
            strip_thinking_tags("<think>first</think>Hello <think>second</think>world"),
            "Hello world"
        );
    }

    #[test]
    fn test_strip_thinking_tags_empty_tags() {
        assert_eq!(
            strip_thinking_tags("<think></think>Hello"),
            "Hello"
        );
    }

    // Integration test for strip_thinking_tags with post-processing
    #[test]
    fn test_strip_thinking_tags_full_pipeline() {
        let input = "<think>Let me think</think>go to Settings.Click Submit";
        let result = strip_thinking_tags(input);
        assert!(result.contains("'Settings'"));
        assert!(result.contains("'Submit'"));
        assert!(!result.contains("<think>"));
    }
}
