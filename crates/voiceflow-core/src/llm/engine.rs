//! LLM engine using llama.cpp

use crate::config::Config;
use crate::llm::prompts::format_prompt;
use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::num::NonZeroU32;
use std::sync::OnceLock;

// Global backend initialization
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn get_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().expect("Failed to initialize llama backend"))
}

/// LLM engine for text formatting
pub struct LlmEngine {
    model: LlamaModel,
    config: Config,
    seed: u32,
}

impl LlmEngine {
    /// Create a new LLM engine with the given configuration
    pub fn new(config: &Config) -> Result<Self> {
        let model_path = config.llm_model_path()?;

        if !model_path.exists() {
            anyhow::bail!(
                "LLM model not found at {:?}. Run 'voiceflow setup' to download models.",
                model_path
            );
        }

        tracing::info!("Loading LLM model from {:?}", model_path);

        let _backend = get_backend();

        // Configure model parameters
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(config.llm_options.n_gpu_layers as u32);

        let model = LlamaModel::load_from_file(
            _backend,
            model_path.to_str().unwrap(),
            &model_params,
        )
        .context("Failed to load LLM model")?;

        tracing::info!("LLM model loaded: {}", config.llm_model.display_name());

        // Generate a random seed based on current time
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(42);

        Ok(Self {
            model,
            config: config.clone(),
            seed,
        })
    }

    /// Format a transcript using the LLM
    ///
    /// # Arguments
    /// * `transcript` - Raw transcript from Whisper
    /// * `prompt_template` - Prompt template with {transcript} placeholder
    pub fn format(&mut self, transcript: &str, prompt_template: &str) -> Result<String> {
        let prompt = format_prompt(prompt_template, transcript, &self.config);

        tracing::debug!("LLM prompt length: {} chars", prompt.len());

        // Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048))
            .with_n_batch(512);

        let mut ctx = self
            .model
            .new_context(get_backend(), ctx_params)
            .context("Failed to create LLM context")?;

        // Tokenize prompt
        let tokens = self
            .model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
            .context("Failed to tokenize prompt")?;

        tracing::debug!("Prompt tokenized to {} tokens", tokens.len());

        // Create batch and add tokens
        let mut batch = LlamaBatch::new(512, 1);

        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)?;
        }

        // Decode prompt
        ctx.decode(&mut batch)?;

        // Generate response
        let mut output_tokens = Vec::new();
        let max_tokens = self.config.llm_options.max_tokens as usize;
        let eos_token = self.model.token_eos();

        for i in 0..max_tokens {
            // Get logits for last token
            let logits = ctx.candidates_ith(batch.n_tokens() - 1);

            // Sample next token
            let mut candidates = LlamaTokenDataArray::from_iter(logits, false);

            // Simple token sampling with seed for reproducibility
            let new_token = candidates.sample_token(self.seed.wrapping_add(i as u32));

            // Check for EOS
            if new_token == eos_token {
                break;
            }

            output_tokens.push(new_token);

            // Prepare next batch
            batch.clear();
            batch.add(new_token, tokens.len() as i32 + output_tokens.len() as i32 - 1, &[0], true)?;

            ctx.decode(&mut batch)?;
        }

        // Detokenize output
        let output = output_tokens
            .iter()
            .filter_map(|t| self.model.token_to_str(*t, llama_cpp_2::model::Special::Tokenize).ok())
            .collect::<String>();

        Ok(output.trim().to_string())
    }
}
