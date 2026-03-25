//! LLM-based text formatting
//!
//! Uses llama.cpp backend for all GGUF models (Qwen3.5, etc.)
//! Supports multimodal inference via mtmd feature for vision+text.

mod backend;
mod engine;
mod llamacpp_backend;
pub(crate) mod numbers;
mod prompts;

pub use backend::LlmBackendTrait;
pub use engine::{detect_hardware, LlmEngine};
pub use llamacpp_backend::LlamaCppBackend;
pub use prompts::format_prompt;
