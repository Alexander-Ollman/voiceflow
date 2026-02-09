//! mistral.rs backend for LLM inference
//! Supports Qwen, Gemma2, Phi-2, and other architectures implemented in mistral.rs

use crate::config::Config;
use crate::llm::backend::LlmBackendTrait;
use crate::runtime;
use anyhow::{Context, Result};
use mistralrs::{GgufModelBuilder, Model, RequestBuilder, TextMessageRole, TextMessages};
use std::sync::Arc;

/// mistral.rs backend implementation
pub struct MistralRsBackend {
    model: Arc<Model>,
}

impl MistralRsBackend {
    /// Create a new mistral.rs backend (async)
    pub async fn new_async(config: &Config) -> Result<Self> {
        let model_path = config.llm_model_path()?;

        if !model_path.exists() {
            anyhow::bail!(
                "LLM model not found at {:?}. Run 'voiceflow setup' to download models.",
                model_path
            );
        }

        tracing::info!("Loading LLM model with mistral.rs from {:?}", model_path);

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
        let model = GgufModelBuilder::new(model_dir, vec![model_file])
            .with_logging()
            .build()
            .await
            .context("Failed to load LLM model with mistral.rs")?;

        tracing::info!(
            "mistral.rs backend loaded: {}",
            config.llm_model.display_name()
        );

        Ok(Self {
            model: Arc::new(model),
        })
    }

    /// Create a new mistral.rs backend (blocking)
    pub fn new(config: &Config) -> Result<Self> {
        let config = config.clone();
        runtime::block_on(Self::new_async(&config))
    }

    /// Generate text (async)
    pub async fn generate_async(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        let messages = TextMessages::new()
            .enable_thinking(false)
            .add_message(TextMessageRole::User, prompt);

        let request = RequestBuilder::from(messages)
            .set_sampler_max_len(max_tokens as usize)
            .set_sampler_temperature(temperature as f64)
            .set_sampler_topp(top_p as f64);

        let response = self
            .model
            .send_chat_request(request)
            .await
            .context("mistral.rs inference failed")?;

        let output = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        Ok(output)
    }
}

impl LlmBackendTrait for MistralRsBackend {
    fn generate(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        runtime::block_on(self.generate_async(prompt, max_tokens, temperature, top_p))
    }

    fn name(&self) -> &'static str {
        "mistral.rs"
    }
}
