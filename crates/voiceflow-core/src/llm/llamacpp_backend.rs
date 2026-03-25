//! llama.cpp backend for LLM inference via llama-cpp-2 crate
//! Supports all GGUF architectures including Qwen3.5.
//! Multimodal support via mtmd feature for vision+text inference.

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
use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams, MtmdBitmap, MtmdInputText};
use std::io::Write;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

/// Write a message to the shared debug log
fn log_perf(msg: &str) {
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/voiceflow_debug.log")
    {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(file, "[{}] [perf] {}", timestamp, msg);
    }
}

/// llama.cpp backend implementation
pub struct LlamaCppBackend {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    mtmd_ctx: Option<MtmdContext>,
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

        let model_params = LlamaModelParams::default();

        // Load the model
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .context("Failed to load model with llama.cpp")?;

        // Load mmproj for multimodal support if available
        let mtmd_ctx = if let Ok(Some(mmproj_path)) = config.mmproj_model_path() {
            if mmproj_path.exists() {
                tracing::info!("Loading mmproj for multimodal support from {:?}", mmproj_path);
                let mtmd_params = MtmdContextParams::default();
                let mmproj_str = mmproj_path.to_str()
                    .ok_or_else(|| anyhow::anyhow!("mmproj path is not valid UTF-8"))?;
                match MtmdContext::init_from_file(mmproj_str, &model, &mtmd_params) {
                    Ok(ctx) => {
                        tracing::info!("Multimodal context loaded successfully");
                        Some(ctx)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load mmproj (multimodal disabled): {}", e);
                        None
                    }
                }
            } else {
                tracing::debug!("mmproj file not found at {:?}, multimodal disabled", mmproj_path);
                None
            }
        } else {
            None
        };

        tracing::info!("llama.cpp backend loaded: {} (multimodal: {})", config.llm_model.display_name(), mtmd_ctx.is_some());

        Ok(Self {
            backend: Arc::new(backend),
            model: Arc::new(model),
            mtmd_ctx,
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
        let t_total = Instant::now();

        // Create context parameters - use 8192 to accommodate long prompts
        let n_ctx_size: u32 = 8192;
        let n_batch_size: u32 = 512; // Process prompt in chunks to avoid Metal limits
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx_size))
            .with_n_batch(n_batch_size);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("Failed to create llama.cpp context")?;

        // Tokenize the prompt — skip BOS when using chat template markers
        // (the <|im_start|> token serves as BOS for Qwen models)
        let t_tokenize = Instant::now();
        let add_bos = if prompt.starts_with("<|im_start|>") {
            AddBos::Never
        } else {
            AddBos::Always
        };
        let tokens = self
            .model
            .str_to_token(prompt, add_bos)
            .context("Failed to tokenize prompt")?;

        // Extra budget for thinking tokens (<think>...</think>) that the model
        // may produce before the actual answer. These get stripped later.
        let thinking_budget: u32 = 1024;
        let total_max_tokens = max_tokens + thinking_budget;

        let n_ctx = ctx.n_ctx() as usize;
        let tokens_len = tokens.len();
        let tokenize_ms = t_tokenize.elapsed().as_millis();

        // Truncate prompt if it exceeds context window (leave room for output)
        let max_prompt_tokens = n_ctx.saturating_sub(total_max_tokens as usize);
        let tokens = if tokens_len > max_prompt_tokens {
            tracing::warn!(
                "Prompt ({} tokens) exceeds max prompt size ({}), truncating",
                tokens_len, max_prompt_tokens
            );
            &tokens[..max_prompt_tokens]
        } else {
            &tokens[..]
        };
        let tokens_len = tokens.len();

        // Process prompt in batches of n_batch_size to avoid Metal/GGML limits
        let t_prompt = Instant::now();
        let mut batch = LlamaBatch::new(n_batch_size as usize, 1);
        let mut n_decoded = 0usize;

        while n_decoded < tokens_len {
            batch.clear();
            let chunk_end = (n_decoded + n_batch_size as usize).min(tokens_len);
            for i in n_decoded..chunk_end {
                let is_last = i == tokens_len - 1;
                batch
                    .add(tokens[i], i as i32, &[0], is_last)
                    .context("Failed to add token to batch")?;
            }
            ctx.decode(&mut batch)
                .context("Failed to decode prompt batch")?;
            n_decoded = chunk_end;
        }
        let prompt_ms = t_prompt.elapsed().as_millis();

        // Set up sampler with temperature and top_p
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(
                512,  // penalty_last_n: look back 512 tokens to prevent output duplication
                1.1,  // penalty_repeat: mild repeat penalty
                0.0,  // penalty_freq: disabled
                0.0,  // penalty_present: disabled
            ),
            LlamaSampler::temp(temperature),
            LlamaSampler::top_p(top_p, 1),
            LlamaSampler::dist(42), // Random seed
        ]);

        // Generate tokens with thinking-aware budget:
        // - Thinking tokens (<think>...</think>) get up to thinking_budget extra tokens
        // - Visible output tokens are capped at max_tokens
        let t_gen = Instant::now();
        let mut output_tokens: Vec<LlamaToken> = Vec::new();
        let mut n_cur = tokens_len;
        let mut in_thinking = false;
        let mut visible_count: u32 = 0;
        let mut thinking_count: u32 = 0;
        let mut token_buf = String::new();

        for _ in 0..total_max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() as i32 - 1);

            if self.model.is_eog_token(new_token) {
                break;
            }

            output_tokens.push(new_token);

            // Track thinking state for budget management
            let token_str = self
                .model
                .token_to_str(new_token, Special::Tokenize)
                .unwrap_or_default();
            token_buf.push_str(&token_str);

            if !in_thinking {
                if token_buf.contains("<think>") {
                    in_thinking = true;
                    token_buf.clear();
                } else {
                    visible_count += 1;
                    // Clear buffer periodically to avoid unbounded growth
                    if token_buf.len() > 20 && !token_buf.contains('<') {
                        token_buf.clear();
                    }
                }
            } else {
                thinking_count += 1;
                if token_buf.contains("</think>") {
                    in_thinking = false;
                    token_buf.clear();
                } else if token_buf.len() > 20 {
                    let keep = token_buf.len() - 15;
                    token_buf = token_buf[keep..].to_string();
                }
            }

            // Stop when visible output reaches max_tokens
            if visible_count >= max_tokens {
                break;
            }
            // Stop if thinking is taking too long
            if thinking_count >= thinking_budget {
                tracing::warn!("Thinking budget exhausted ({} tokens)", thinking_budget);
                break;
            }

            // Prepare batch for next iteration
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .context("Failed to add generated token to batch")?;

            n_cur += 1;

            ctx.decode(&mut batch)
                .context("Failed to decode generated token")?;
        }
        let gen_ms = t_gen.elapsed().as_millis();
        let total_generated = output_tokens.len();

        // Convert tokens to string
        let mut output = String::new();
        for token in output_tokens {
            let token_str = self
                .model
                .token_to_str(token, Special::Tokenize)
                .unwrap_or_default();
            output.push_str(&token_str);
        }

        let total_ms = t_total.elapsed().as_millis();
        let gen_tok_per_sec = if gen_ms > 0 { (total_generated as f64 / gen_ms as f64) * 1000.0 } else { 0.0 };
        let prompt_tok_per_sec = if prompt_ms > 0 { (tokens_len as f64 / prompt_ms as f64) * 1000.0 } else { 0.0 };
        log_perf(&format!(
            "{}ms total | prompt: {} tok in {}ms ({:.0} t/s) | gen: {} tok in {}ms ({:.0} t/s, {} thinking + {} visible) | tokenize: {}ms",
            total_ms, tokens_len, prompt_ms, prompt_tok_per_sec,
            total_generated, gen_ms, gen_tok_per_sec, thinking_count, visible_count,
            tokenize_ms,
        ));

        Ok(output.trim().to_string())
    }

    /// Generate text completion with per-token streaming callback
    fn generate_completion_streaming(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        // Create context parameters - use 8192 to accommodate long prompts
        let n_ctx_size: u32 = 8192;
        let n_batch_size: u32 = 512;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx_size))
            .with_n_batch(n_batch_size);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("Failed to create llama.cpp context")?;

        // Skip BOS when using chat template markers
        let add_bos = if prompt.starts_with("<|im_start|>") {
            AddBos::Never
        } else {
            AddBos::Always
        };
        let tokens = self
            .model
            .str_to_token(prompt, add_bos)
            .context("Failed to tokenize prompt")?;

        let thinking_budget: u32 = 1024;
        let total_max_tokens = max_tokens + thinking_budget;

        let n_ctx = ctx.n_ctx() as usize;
        let tokens_len = tokens.len();

        tracing::debug!("Prompt tokenized to {} tokens (context: {})", tokens_len, n_ctx);

        let max_prompt_tokens = n_ctx.saturating_sub(total_max_tokens as usize);
        let tokens = if tokens_len > max_prompt_tokens {
            tracing::warn!(
                "Prompt ({} tokens) exceeds max prompt size ({}), truncating",
                tokens_len, max_prompt_tokens
            );
            &tokens[..max_prompt_tokens]
        } else {
            &tokens[..]
        };
        let tokens_len = tokens.len();

        // Process prompt in batches of n_batch_size to avoid Metal/GGML limits
        let mut batch = LlamaBatch::new(n_batch_size as usize, 1);
        let mut n_decoded = 0usize;

        while n_decoded < tokens_len {
            batch.clear();
            let chunk_end = (n_decoded + n_batch_size as usize).min(tokens_len);
            for i in n_decoded..chunk_end {
                let is_last = i == tokens_len - 1;
                batch
                    .add(tokens[i], i as i32, &[0], is_last)
                    .context("Failed to add token to batch")?;
            }
            ctx.decode(&mut batch)
                .context("Failed to decode prompt batch")?;
            n_decoded = chunk_end;
        }

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(temperature),
            LlamaSampler::top_p(top_p, 1),
            LlamaSampler::dist(42),
        ]);

        let mut output = String::new();
        let mut n_cur = tokens_len;
        let mut in_thinking = false;
        let mut visible_count: u32 = 0;
        let mut thinking_count: u32 = 0;
        let mut tag_buf = String::new();

        for _ in 0..total_max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() as i32 - 1);

            if self.model.is_eog_token(new_token) {
                break;
            }

            let token_str = self
                .model
                .token_to_str(new_token, Special::Tokenize)
                .unwrap_or_default();

            // Track thinking state
            tag_buf.push_str(&token_str);
            if !in_thinking {
                if tag_buf.contains("<think>") {
                    in_thinking = true;
                    tag_buf.clear();
                } else {
                    visible_count += 1;
                    if tag_buf.len() > 20 && !tag_buf.contains('<') {
                        tag_buf.clear();
                    }
                }
            } else {
                thinking_count += 1;
                if tag_buf.contains("</think>") {
                    in_thinking = false;
                    tag_buf.clear();
                } else if tag_buf.len() > 20 {
                    let keep = tag_buf.len() - 15;
                    tag_buf = tag_buf[keep..].to_string();
                }
            }

            if !token_str.is_empty() {
                output.push_str(&token_str);
                if !on_token(&token_str) {
                    break;
                }
            }

            if visible_count >= max_tokens || thinking_count >= thinking_budget {
                break;
            }

            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .context("Failed to add generated token to batch")?;

            n_cur += 1;

            ctx.decode(&mut batch)
                .context("Failed to decode generated token")?;
        }

        Ok(output.trim().to_string())
    }

    /// Generate text completion with an image (multimodal)
    fn generate_with_image_impl(
        &self,
        prompt: &str,
        image: &[u8],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        mut on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<String> {
        let mtmd_ctx = self.mtmd_ctx.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Multimodal not available: mmproj not loaded"))?;

        let n_ctx_size: u32 = 8192;
        let n_batch_size: u32 = 512;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx_size))
            .with_n_batch(n_batch_size);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("Failed to create llama.cpp context")?;

        // Create bitmap from image bytes using mtmd context
        let bitmap = MtmdBitmap::from_buffer(mtmd_ctx, image)
            .context("Failed to create bitmap from image data")?;

        // Tokenize the prompt with image using mtmd
        let input_text = MtmdInputText {
            text: prompt.to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])
            .context("Failed to tokenize multimodal prompt")?;

        // Evaluate all chunks (text and image tokens) using batch size limit
        let n_past = chunks.eval_chunks(mtmd_ctx, &ctx, 0, 0, n_batch_size as i32, true)
            .context("Failed to evaluate multimodal chunks")?;
        let mut n_past = n_past;

        // Set up sampler
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(temperature),
            LlamaSampler::top_p(top_p, 1),
            LlamaSampler::dist(42),
        ]);

        // Generate tokens
        let mut output = String::new();
        let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, -1);

            if self.model.is_eog_token(new_token) {
                break;
            }

            let token_str = self
                .model
                .token_to_str(new_token, Special::Tokenize)
                .unwrap_or_default();

            if !token_str.is_empty() {
                output.push_str(&token_str);
                if let Some(ref mut cb) = on_token {
                    if !cb(&token_str) {
                        break;
                    }
                }
            }

            batch.clear();
            batch
                .add(new_token, n_past, &[0], true)
                .context("Failed to add generated token to batch")?;

            n_past += 1;

            ctx.decode(&mut batch)
                .context("Failed to decode generated token")?;
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

    fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        on_token: super::backend::TokenCallback<'_>,
    ) -> Result<String> {
        self.generate_completion_streaming(prompt, max_tokens, temperature, top_p, on_token)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn generate_with_image(
        &self,
        prompt: &str,
        image: &[u8],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        self.generate_with_image_impl(prompt, image, max_tokens, temperature, top_p, None)
    }

    fn generate_with_image_streaming(
        &self,
        prompt: &str,
        image: &[u8],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        on_token: super::backend::TokenCallback<'_>,
    ) -> Result<String> {
        self.generate_with_image_impl(prompt, image, max_tokens, temperature, top_p, Some(on_token))
    }

    fn supports_multimodal(&self) -> bool {
        self.mtmd_ctx.is_some()
    }
}
