//! LLM-based text formatting

mod engine;
mod prompts;

pub use engine::{detect_hardware, LlmEngine};
pub use prompts::format_prompt;
