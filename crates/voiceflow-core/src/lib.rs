//! VoiceFlow Core - Voice-to-formatted-text pipeline
//!
//! This library provides the core functionality for:
//! - Audio capture and voice activity detection
//! - Speech-to-text via Whisper
//! - LLM-based text reformatting with Qwen3/SmolLM3
//! - Context-aware prompt selection

pub mod audio;
pub mod config;
pub mod context;
pub mod llm;
pub mod transcribe;

mod pipeline;

pub use config::{Config, LlmModel, WhisperModel};
pub use pipeline::{Pipeline, PipelineResult, Timings};

/// Process audio samples and return formatted text
///
/// This is the main entry point for the library.
pub fn process_audio(
    audio: &[f32],
    context: Option<&str>,
    config: &Config,
) -> anyhow::Result<PipelineResult> {
    let mut pipeline = Pipeline::new(config)?;
    pipeline.process(audio, context)
}
