//! LLM engine using mistral.rs for cross-platform inference
//! Supports Metal (macOS), CUDA (Linux), and CPU fallback

use crate::config::Config;
use crate::llm::prompts::{format_prompt, post_process_output};
use anyhow::{Context, Result};
use mistralrs::{GgufModelBuilder, Model, RequestBuilder, TextMessages, TextMessageRole};
use std::sync::Arc;

/// LLM engine for text formatting using mistral.rs
pub struct LlmEngine {
    model: Arc<Model>,
    config: Config,
}

impl LlmEngine {
    /// Create a new LLM engine with the given configuration (async)
    pub async fn new_async(config: &Config) -> Result<Self> {
        let model_path = config.llm_model_path()?;

        if !model_path.exists() {
            anyhow::bail!(
                "LLM model not found at {:?}. Run 'voiceflow setup' to download models.",
                model_path
            );
        }

        tracing::info!("Loading LLM model from {:?}", model_path);

        // Get parent directory and filename
        let model_dir = model_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_string_lossy()
            .to_string();
        let model_file = model_path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| config.llm_model.filename().to_string());

        // Build model using mistral.rs async API
        // Note: We avoid PagedAttention for now as it can cause Metal shader conflicts
        let model = GgufModelBuilder::new(model_dir, vec![model_file])
            .with_logging()
            .build()
            .await
            .context("Failed to load LLM model with mistral.rs")?;

        tracing::info!("LLM model loaded: {}", config.llm_model.display_name());

        Ok(Self {
            model: Arc::new(model),
            config: config.clone(),
        })
    }

    /// Create a new LLM engine (blocking wrapper for sync contexts)
    pub fn new(config: &Config) -> Result<Self> {
        // Use tokio's current runtime if available, otherwise create one
        match tokio::runtime::Handle::try_current() {
            Ok(_handle) => {
                // We're in an async context - use spawn_blocking to avoid nested runtime
                let config = config.clone();
                std::thread::scope(|s| {
                    s.spawn(|| {
                        let rt = tokio::runtime::Runtime::new()?;
                        rt.block_on(Self::new_async(&config))
                    }).join().unwrap()
                })
            }
            Err(_) => {
                // No runtime, create one
                let rt = tokio::runtime::Runtime::new()
                    .context("Failed to create tokio runtime")?;
                rt.block_on(Self::new_async(config))
            }
        }
    }

    /// Format a transcript using the LLM (async)
    pub async fn format_async(&self, transcript: &str, prompt_template: &str) -> Result<String> {
        let prompt = format_prompt(prompt_template, transcript, &self.config);

        tracing::debug!("LLM prompt length: {} chars", prompt.len());

        // Build messages with thinking disabled for fast inference (enable_thinking defaults to false)
        let messages = TextMessages::new()
            .enable_thinking(self.config.llm_options.enable_thinking)
            .add_message(TextMessageRole::User, &prompt);

        // Build request with sampling parameters
        let request = RequestBuilder::from(messages)
            .set_sampler_max_len(self.config.llm_options.max_tokens as usize)
            .set_sampler_temperature(self.config.llm_options.temperature as f64)
            .set_sampler_topp(self.config.llm_options.top_p as f64);

        // Run inference
        let response = self.model.send_chat_request(request).await
            .context("LLM inference failed")?;

        // Extract response text and strip any thinking tags
        let output = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| strip_thinking_tags(s.trim()))
            .unwrap_or_default();

        tracing::debug!("LLM output length: {} chars", output.len());

        Ok(output)
    }

    /// Format a transcript using the LLM (blocking wrapper)
    pub fn format(&self, transcript: &str, prompt_template: &str) -> Result<String> {
        match tokio::runtime::Handle::try_current() {
            Ok(_handle) => {
                // We're in an async context - use spawn_blocking
                let model = Arc::clone(&self.model);
                let config = self.config.clone();
                let transcript = transcript.to_string();
                let prompt_template = prompt_template.to_string();

                std::thread::scope(|s| {
                    s.spawn(|| {
                        let rt = tokio::runtime::Runtime::new()?;
                        rt.block_on(async {
                            let prompt = format_prompt(&prompt_template, &transcript, &config);
                            let messages = TextMessages::new()
                                .enable_thinking(config.llm_options.enable_thinking)
                                .add_message(TextMessageRole::User, &prompt);

                            let request = RequestBuilder::from(messages)
                                .set_sampler_max_len(config.llm_options.max_tokens as usize)
                                .set_sampler_temperature(config.llm_options.temperature as f64)
                                .set_sampler_topp(config.llm_options.top_p as f64);

                            let response = model.send_chat_request(request).await
                                .context("LLM inference failed")?;

                            let output = response
                                .choices
                                .first()
                                .and_then(|c| c.message.content.as_ref())
                                .map(|s| strip_thinking_tags(s.trim()))
                                .unwrap_or_default();

                            Ok(output)
                        })
                    }).join().unwrap()
                })
            }
            Err(_) => {
                // No runtime, create one
                let rt = tokio::runtime::Runtime::new()
                    .context("Failed to create tokio runtime")?;
                rt.block_on(self.format_async(transcript, prompt_template))
            }
        }
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

// Post-processing functions moved to prompts.rs for better separation of concerns.
// This module now uses crate::llm::prompts::post_process_output for output cleanup.

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
