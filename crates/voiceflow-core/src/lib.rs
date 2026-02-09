//! VoiceFlow Core - Voice-to-formatted-text pipeline
//!
//! This library provides the core functionality for:
//! - Audio capture and voice activity detection
//! - Speech-to-text via Whisper
//! - LLM-based text reformatting with Qwen3/SmolLM3
//! - Context-aware prompt selection
//! - Prosody analysis for punctuation detection

pub mod audio;
pub mod config;
pub mod context;
pub mod llm;
pub mod prosody;
pub mod runtime;
pub mod transcribe;

mod pipeline;

pub use config::{Config, LlmModel, WhisperModel, ConfigError, PipelineMode, ConsolidatedModel, VlmModel, env_vars};
pub use pipeline::{Pipeline, PipelineResult, ProsodyOptions, Timings, RecoveryConfig, PipelineError};
pub use prosody::{ProsodyHints, PitchContour};

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
