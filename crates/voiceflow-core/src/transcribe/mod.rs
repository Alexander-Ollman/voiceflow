//! Speech-to-text transcription engines

mod whisper;
mod moonshine;

pub use whisper::{WhisperEngine, WordTimestamp, TranscriptionResult};
pub use moonshine::MoonshineEngine;
