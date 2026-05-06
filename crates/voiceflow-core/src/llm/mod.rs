//! LLM-based text formatting
//!
//! Uses llama.cpp backend for all GGUF models (Qwen3.5, etc.)
//! Supports multimodal inference via mtmd feature for vision+text.
//! Optional mistral.rs backend for Gemma 4 with native audio support.

mod backend;
mod engine;
mod llamacpp_backend;
#[cfg(feature = "mistralrs")]
mod mistralrs_backend;
mod openai_server_backend;
pub(crate) mod numbers;
pub(crate) mod prompts;

pub use backend::LlmBackendTrait;
pub use engine::{detect_hardware, LlmEngine};
pub use llamacpp_backend::LlamaCppBackend;
#[cfg(feature = "mistralrs")]
pub use mistralrs_backend::MistralRsBackend;
pub use openai_server_backend::OpenAIServerBackend;
pub use prompts::format_prompt;
