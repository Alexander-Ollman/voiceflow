//! llama.cpp backend for LLM inference via llama-cpp-2 crate
//! Supports all GGUF architectures including SmolLM3, Gemma3n, Phi-4, etc.

use crate::config::Config;
use crate::llm::backend::LlmBackendTrait;
use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::sync::Arc;

/// llama.cpp backend implementation
pub struct LlamaCppBackend {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    model_name: String,
}

impl LlamaCppBackend {
    /// Create a new llama.cpp backend
    pub fn new(config: &Config) -> Result<Self> {
        let model_path = config.llm_model_path()?;

        if !model_path.exists() {
            anyhow::bail!(
                "LLM model not found at {:?}. Run 'voiceflow setup' to download models.",
                model_path
            );
        }

        tracing::info!("Loading LLM model with llama.cpp from {:?}", model_path);

        // Initialize the llama.cpp backend
        let backend =
            LlamaBackend::init().context("Failed to initialize llama.cpp backend")?;

        // Configure model parameters
        // Note: n_gpu_layers controls how many layers go to GPU (Metal on macOS, CUDA on Linux)
        // Default to 0 (CPU only) for stability - can be increased when Metal support is verified
        let model_params = LlamaModelParams::default();

        // Load the model
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .context("Failed to load model with llama.cpp")?;

        let model_name = config.llm_model.display_name().to_string();
        tracing::info!("llama.cpp backend loaded: {}", model_name);

        Ok(Self {
            backend: Arc::new(backend),
            model: Arc::new(model),
            model_name,
        })
    }

    /// Generate text completion
    fn generate_completion(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        // Create context parameters
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(4096)); // Context size

        // Create inference context
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("Failed to create llama.cpp context")?;

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .context("Failed to tokenize prompt")?;

        let n_ctx = ctx.n_ctx() as usize;
        let tokens_len = tokens.len();

        if tokens_len + max_tokens as usize > n_ctx {
            tracing::warn!(
                "Prompt ({} tokens) + max_tokens ({}) exceeds context size ({})",
                tokens_len,
                max_tokens,
                n_ctx
            );
        }

        // Create a batch for the prompt tokens
        let mut batch = LlamaBatch::new(n_ctx, 1);

        // Add prompt tokens to batch
        let last_idx = tokens_len - 1;
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == last_idx;
            batch
                .add(*token, i as i32, &[0], is_last)
                .context("Failed to add token to batch")?;
        }

        // Process the prompt
        ctx.decode(&mut batch)
            .context("Failed to decode prompt batch")?;

        // Set up sampler with temperature and top_p
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(temperature),
            LlamaSampler::top_p(top_p, 1),
            LlamaSampler::dist(42), // Random seed
        ]);

        // Generate tokens
        let mut output_tokens: Vec<LlamaToken> = Vec::new();
        let mut n_cur = tokens_len;

        for _ in 0..max_tokens {
            // Sample the next token
            let new_token = sampler.sample(&ctx, batch.n_tokens() as i32 - 1);

            // Check for end of generation
            if self.model.is_eog_token(new_token) {
                break;
            }

            output_tokens.push(new_token);

            // Prepare batch for next iteration
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .context("Failed to add generated token to batch")?;

            n_cur += 1;

            // Decode the new token
            ctx.decode(&mut batch)
                .context("Failed to decode generated token")?;
        }

        // Convert tokens to string
        let mut output = String::new();
        for token in output_tokens {
            let token_str = self
                .model
                .token_to_str(token, Special::Tokenize)
                .unwrap_or_default();
            output.push_str(&token_str);
        }

        Ok(output.trim().to_string())
    }
}

impl LlmBackendTrait for LlamaCppBackend {
    fn generate(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        self.generate_completion(prompt, max_tokens, temperature, top_p)
    }

    fn name(&self) -> &'static str {
        "llama.cpp"
    }
}
