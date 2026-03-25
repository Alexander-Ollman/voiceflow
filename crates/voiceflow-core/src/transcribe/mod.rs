//! Speech-to-text transcription engines

#[cfg(feature = "whisper")]
mod whisper;
mod moonshine;

#[cfg(feature = "whisper")]
pub use whisper::{WhisperEngine, WordTimestamp, TranscriptionResult};
#[cfg(not(feature = "whisper"))]
pub use moonshine_types::{WordTimestamp, TranscriptionResult};

pub use moonshine::MoonshineEngine;

/// When whisper is not available, provide the shared types from here
#[cfg(not(feature = "whisper"))]
mod moonshine_types {
    /// A word with its timestamp information
    #[derive(Debug, Clone)]
    pub struct WordTimestamp {
        /// The word text
        pub word: String,
        /// Start time in milliseconds
        pub start_ms: i64,
        /// End time in milliseconds
        pub end_ms: i64,
        /// Probability/confidence (0.0 - 1.0)
        pub probability: f32,
    }

    /// Result of transcription with optional word timestamps
    #[derive(Debug, Clone)]
    pub struct TranscriptionResult {
        /// Full transcribed text
        pub text: String,
        /// Word-level timestamps (if enabled)
        pub word_timestamps: Vec<WordTimestamp>,
    }
}
