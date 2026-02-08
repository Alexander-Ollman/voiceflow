//! LLM-based text formatting
//!
//! Supports multiple inference backends:
//! - mistral.rs: Good for Qwen, Gemma2, Phi-2 (some newer architectures unsupported)
//! - llama.cpp: Supports all GGUF architectures (SmolLM3, Gemma3n, Phi-4, etc.)

mod backend;
mod engine;
mod llamacpp_backend;
mod mistralrs_backend;
pub(crate) mod numbers;
mod prompts;

pub use backend::LlmBackendTrait;
pub use engine::{detect_hardware, LlmEngine};
pub use llamacpp_backend::LlamaCppBackend;
pub use mistralrs_backend::MistralRsBackend;
pub use prompts::format_prompt;
