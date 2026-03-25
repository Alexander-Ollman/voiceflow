//! Test: load model + create context + tokenize + decode + sample
//! Run: cargo run --release -p voiceflow-core --no-default-features --example test_qwen35
//!
//! IMPORTANT: Must be run with --no-default-features to avoid whisper-rs GGML symbol
//! conflicts that cause segfaults in llama.cpp's graph_reserve.

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;

use voiceflow_core::config::Config;

fn test_model(backend: &LlamaBackend, label: &str, path: &std::path::Path) {
    eprint!("[{}] Loading... ", label);
    let model_params = LlamaModelParams::default();
    let model = match LlamaModel::load_from_file(backend, path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("LOAD FAILED: {}", e);
            return;
        }
    };
    eprintln!("loaded.");

    eprint!("  Context... ");
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512));
    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("CONTEXT FAILED: {}", e);
            return;
        }
    };
    eprintln!("OK");

    eprint!("  Tokenize... ");
    let tokens = match model.str_to_token("Hello", AddBos::Always) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("TOKENIZE FAILED: {}", e);
            return;
        }
    };
    eprintln!("{} tokens", tokens.len());

    eprint!("  Decode... ");
    let mut batch = LlamaBatch::new(512, 1);
    let last_idx = tokens.len() - 1;
    for (i, token) in tokens.iter().enumerate() {
        batch.add(*token, i as i32, &[0], i == last_idx).unwrap();
    }
    match ctx.decode(&mut batch) {
        Ok(_) => eprintln!("OK"),
        Err(e) => {
            eprintln!("DECODE FAILED: {}", e);
            return;
        }
    }

    eprint!("  Sample... ");
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(42),
    ]);
    let new_token = sampler.sample(&ctx, batch.n_tokens() as i32 - 1);
    eprintln!("token_id={}", new_token.0);

    eprintln!("  [{}] PASS", label);
}

fn main() {
    let base = Config::models_dir().expect("models dir");
    eprintln!("Models dir: {:?}", base);

    let backend = LlamaBackend::init().expect("backend init");

    let models = [
        ("0.8B", "Qwen3.5-0.8B-Q4_K_M.gguf"),
        ("2B", "Qwen3.5-2B-Q4_K_M.gguf"),
        ("4B", "Qwen3.5-4B-Q4_K_M.gguf"),
    ];

    for (label, filename) in &models {
        let path = base.join(filename);
        if !path.exists() {
            eprintln!("[{}] SKIP — file not found", label);
            continue;
        }
        test_model(&backend, label, &path);
    }

    println!("ALL DONE");
}
