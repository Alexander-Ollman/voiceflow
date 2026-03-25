//! LLM engine with llama.cpp backend
//!
//! All Qwen3.5 models use llama.cpp for inference (Hybrid DeltaNet architecture).
//! Supports multimodal (text + vision) via optional mmproj projector.

use crate::config::Config;
use crate::llm::backend::LlmBackendTrait;
use crate::llm::llamacpp_backend::LlamaCppBackend;
use crate::llm::prompts::{format_prompt, post_process_output};
use anyhow::{Context, Result};

/// LLM engine for text formatting with switchable backends
pub struct LlmEngine {
    backend: Box<dyn LlmBackendTrait>,
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

        tracing::info!(
            "Loading LLM model {} with llama.cpp backend",
            config.llm_model.display_name(),
        );

        let backend: Box<dyn LlmBackendTrait> = {
            let b = LlamaCppBackend::new(config)
                .context("Failed to initialize llama.cpp backend")?;
            Box::new(b)
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

    /// Create a new LLM engine (blocking/sync)
    ///
    /// Loads the model directly on the calling thread. This avoids creating
    /// a temporary Tokio runtime for model loading, which would cause Metal
    /// GPU state issues when the runtime is dropped (the Metal context gets
    /// invalidated, leading to segfaults during later inference calls).
    pub fn new(config: &Config) -> Result<Self> {
        let model_path = config.llm_model_path()?;

        if !model_path.exists() {
            anyhow::bail!(
                "LLM model not found at {:?}. Run 'voiceflow setup' to download models.",
                model_path
            );
        }

        tracing::info!(
            "Loading LLM model {} with llama.cpp backend",
            config.llm_model.display_name(),
        );

        let backend: Box<dyn LlmBackendTrait> = {
            let b = LlamaCppBackend::new(config)
                .context("Failed to initialize llama.cpp backend")?;
            Box::new(b)
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

        if output.trim().is_empty() {
            tracing::warn!("LLM output empty after stripping thinking tags, falling back to transcript");
            return Ok(transcript.trim().to_string());
        }

        Ok(output)
    }

    /// Format a transcript using the LLM (blocking)
    ///
    /// Runs inference directly on the calling thread. All llama.cpp operations
    /// are synchronous, so no async runtime is needed.
    pub fn format(&self, transcript: &str, prompt_template: &str) -> Result<String> {
        let prompt = format_prompt(prompt_template, transcript, &self.config);

        tracing::debug!("LLM prompt length: {} chars", prompt.len());

        let output = self.backend.generate(
            &prompt,
            self.config.llm_options.max_tokens,
            self.config.llm_options.temperature,
            self.config.llm_options.top_p,
        )?;

        let output = strip_thinking_tags(&output);

        tracing::debug!("LLM output length: {} chars", output.len());

        // If output is empty (e.g. model spent all tokens thinking), fall back to transcript
        if output.trim().is_empty() {
            tracing::warn!("LLM output empty after stripping thinking tags, falling back to transcript");
            return Ok(transcript.trim().to_string());
        }

        Ok(output)
    }

    /// Format a transcript with per-token streaming callback.
    /// Tokens pass through a ThinkingFilter to suppress `<think>...</think>` content.
    /// Returns the fully post-processed final text (same as `format()`).
    pub fn format_streaming(
        &self,
        transcript: &str,
        prompt_template: &str,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        let prompt = format_prompt(prompt_template, transcript, &self.config);
        tracing::debug!("LLM streaming prompt length: {} chars", prompt.len());

        let mut filter = ThinkingFilter::new();

        let raw_output = self.backend.generate_streaming(
            &prompt,
            self.config.llm_options.max_tokens,
            self.config.llm_options.temperature,
            self.config.llm_options.top_p,
            &mut |token: &str| -> bool {
                if let Some(visible) = filter.process_token(token) {
                    on_token(&visible)
                } else {
                    true // inside <think> block, keep generating
                }
            },
        )?;

        // Flush any buffered text from the filter
        if let Some(remaining) = filter.flush() {
            let _ = on_token(&remaining);
        }

        // Final text gets full post-processing (same as non-streaming path)
        let output = strip_thinking_tags(&raw_output);
        tracing::debug!("LLM streaming output length: {} chars", output.len());
        Ok(output)
    }

    /// Format a transcript with an image using the multimodal LLM
    pub fn format_with_image(
        &self,
        transcript: &str,
        prompt_template: &str,
        image: &[u8],
    ) -> Result<String> {
        let prompt = format_prompt(prompt_template, transcript, &self.config);
        tracing::debug!("LLM multimodal prompt length: {} chars, image: {} bytes", prompt.len(), image.len());

        let output = self.backend.generate_with_image(
            &prompt,
            image,
            self.config.llm_options.max_tokens,
            self.config.llm_options.temperature,
            self.config.llm_options.top_p,
        )?;

        let output = strip_thinking_tags(&output);
        tracing::debug!("LLM multimodal output length: {} chars", output.len());
        Ok(output)
    }

    /// Format a transcript with an image, streaming tokens
    pub fn format_with_image_streaming(
        &self,
        transcript: &str,
        prompt_template: &str,
        image: &[u8],
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        let prompt = format_prompt(prompt_template, transcript, &self.config);
        tracing::debug!("LLM multimodal streaming prompt length: {} chars, image: {} bytes", prompt.len(), image.len());

        let mut filter = ThinkingFilter::new();

        let raw_output = self.backend.generate_with_image_streaming(
            &prompt,
            image,
            self.config.llm_options.max_tokens,
            self.config.llm_options.temperature,
            self.config.llm_options.top_p,
            &mut |token: &str| -> bool {
                if let Some(visible) = filter.process_token(token) {
                    on_token(&visible)
                } else {
                    true
                }
            },
        )?;

        if let Some(remaining) = filter.flush() {
            let _ = on_token(&remaining);
        }

        let output = strip_thinking_tags(&raw_output);
        tracing::debug!("LLM multimodal streaming output length: {} chars", output.len());
        Ok(output)
    }

    /// Check if the backend supports multimodal inference
    pub fn supports_multimodal(&self) -> bool {
        self.backend.supports_multimodal()
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

/// Streaming filter that suppresses `<think>...</think>` content token-by-token.
///
/// Handles tags that span token boundaries by buffering partial tag prefixes.
/// When inside a thinking block, tokens are consumed silently.
/// When outside, visible text is emitted via `process_token()`.
struct ThinkingFilter {
    in_thinking: bool,
    buffer: String,
}

impl ThinkingFilter {
    fn new() -> Self {
        Self {
            in_thinking: false,
            buffer: String::new(),
        }
    }

    /// Process a single token. Returns `Some(visible_text)` if there's text to show,
    /// `None` if the token was consumed (inside thinking block or buffered).
    fn process_token(&mut self, token: &str) -> Option<String> {
        self.buffer.push_str(token);

        if self.in_thinking {
            // Look for </think> in the buffer
            if let Some(end_pos) = self.buffer.find("</think>") {
                let after = self.buffer[end_pos + "</think>".len()..].to_string();
                self.buffer.clear();
                self.in_thinking = false;
                if after.is_empty() {
                    return None;
                }
                // Recurse to handle any remaining text (could contain another <think>)
                return self.process_token(&after);
            }
            // Still inside thinking — keep buffering, but trim to avoid unbounded growth.
            // Keep only the last 20 chars (enough to detect </think> spanning tokens).
            if self.buffer.len() > 20 {
                let keep_from = self.buffer.len() - 20;
                self.buffer = self.buffer[keep_from..].to_string();
            }
            return None;
        }

        // Not in thinking — check for <think> tag
        if let Some(start_pos) = self.buffer.find("<think>") {
            // Emit everything before the tag
            let before = self.buffer[..start_pos].to_string();
            let after = self.buffer[start_pos + "<think>".len()..].to_string();
            self.buffer = after;
            self.in_thinking = true;
            // Check if </think> is already in the remaining buffer
            let _ = self.process_token(""); // process remaining buffer
            if before.is_empty() {
                return None;
            }
            return Some(before);
        }

        // Check if buffer ends with a partial prefix of "<think>"
        let tag = "<think>";
        for prefix_len in (1..tag.len()).rev() {
            if self.buffer.ends_with(&tag[..prefix_len]) {
                // Hold back the partial prefix — it might complete into a tag
                let emit_end = self.buffer.len() - prefix_len;
                if emit_end == 0 {
                    return None; // entire buffer is a partial prefix
                }
                let emit = self.buffer[..emit_end].to_string();
                self.buffer = self.buffer[emit_end..].to_string();
                return Some(emit);
            }
        }

        // No tag concern — emit the whole buffer
        let emit = std::mem::take(&mut self.buffer);
        if emit.is_empty() {
            return None;
        }
        Some(emit)
    }

    /// Flush any remaining buffered text at end of generation.
    fn flush(&mut self) -> Option<String> {
        if self.in_thinking || self.buffer.is_empty() {
            self.buffer.clear();
            return None;
        }
        let remaining = std::mem::take(&mut self.buffer);
        Some(remaining)
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
            // Unclosed <think> block (model hit max_tokens during thinking)
            // Strip everything from <think> onwards since the actual answer hasn't started
            result = result[..start].trim_end().to_string();
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

    #[test]
    fn test_strip_thinking_tags_unclosed() {
        // Model hit max_tokens during thinking — no </think> present
        assert_eq!(
            strip_thinking_tags("<think>some long reasoning that never ends"),
            ""
        );
    }

    #[test]
    fn test_strip_thinking_tags_unclosed_with_prefix() {
        assert_eq!(
            strip_thinking_tags("Hello <think>reasoning without end"),
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

    // ThinkingFilter tests
    #[test]
    fn test_filter_no_tags() {
        let mut f = ThinkingFilter::new();
        let mut out = String::new();
        for token in &["Hello", " world"] {
            if let Some(v) = f.process_token(token) {
                out.push_str(&v);
            }
        }
        if let Some(v) = f.flush() { out.push_str(&v); }
        assert_eq!(out, "Hello world");
    }

    #[test]
    fn test_filter_suppresses_thinking() {
        let mut f = ThinkingFilter::new();
        let mut out = String::new();
        for token in &["<think>", "reasoning", "</think>", "Hello"] {
            if let Some(v) = f.process_token(token) {
                out.push_str(&v);
            }
        }
        if let Some(v) = f.flush() { out.push_str(&v); }
        assert_eq!(out, "Hello");
    }

    #[test]
    fn test_filter_tag_spanning_tokens() {
        let mut f = ThinkingFilter::new();
        let mut out = String::new();
        // Tag split across tokens: "<thi" + "nk>" + "hidden" + "</thi" + "nk>" + "visible"
        for token in &["<thi", "nk>", "hidden", "</thi", "nk>", "visible"] {
            if let Some(v) = f.process_token(token) {
                out.push_str(&v);
            }
        }
        if let Some(v) = f.flush() { out.push_str(&v); }
        assert_eq!(out, "visible");
    }

    #[test]
    fn test_filter_text_before_think() {
        let mut f = ThinkingFilter::new();
        let mut out = String::new();
        for token in &["Hello ", "<think>", "reasoning", "</think>", " world"] {
            if let Some(v) = f.process_token(token) {
                out.push_str(&v);
            }
        }
        if let Some(v) = f.flush() { out.push_str(&v); }
        assert_eq!(out, "Hello  world");
    }

    #[test]
    fn test_filter_flush_emits_buffered() {
        let mut f = ThinkingFilter::new();
        let mut out = String::new();
        // Partial prefix of <think> at end — should be emitted on flush
        if let Some(v) = f.process_token("Hello<") {
            out.push_str(&v);
        }
        if let Some(v) = f.flush() { out.push_str(&v); }
        assert_eq!(out, "Hello<");
    }
}
