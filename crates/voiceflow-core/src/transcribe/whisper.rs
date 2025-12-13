//! Whisper speech-to-text engine

use crate::config::Config;
use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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

/// Whisper-based speech-to-text engine
pub struct WhisperEngine {
    ctx: WhisperContext,
}

impl WhisperEngine {
    /// Create a new Whisper engine with the given configuration
    pub fn new(config: &Config) -> Result<Self> {
        let model_path = config.whisper_model_path()?;

        if !model_path.exists() {
            anyhow::bail!(
                "Whisper model not found at {:?}. Run 'voiceflow setup' to download models.",
                model_path
            );
        }

        tracing::info!("Loading Whisper model from {:?}", model_path);

        let ctx = WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            WhisperContextParameters::default(),
        )
        .context("Failed to load Whisper model")?;

        Ok(Self { ctx })
    }

    /// Transcribe audio samples to text
    ///
    /// # Arguments
    /// * `audio` - PCM f32 samples at 16kHz (caller must resample first)
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        Ok(self.transcribe_with_timestamps(audio, false)?.text)
    }

    /// Transcribe audio samples with word-level timestamps
    ///
    /// # Arguments
    /// * `audio` - PCM f32 samples at 16kHz (caller must resample first)
    /// * `enable_timestamps` - Whether to extract word-level timestamps
    pub fn transcribe_with_timestamps(
        &mut self,
        audio: &[f32],
        enable_timestamps: bool,
    ) -> Result<TranscriptionResult> {
        // Audio must already be 16kHz - caller is responsible for resampling
        let audio_16k = audio;

        // Create whisper parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Configure for speed
        params.set_n_threads(std::thread::available_parallelism()?.get() as i32);
        params.set_language(Some("en"));
        params.set_translate(false);
        params.set_no_context(true);
        params.set_single_segment(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Enable token-level timestamps for word extraction
        if enable_timestamps {
            params.set_token_timestamps(true);
            params.set_max_len(1); // Force word-level segmentation
        }

        // Suppress non-speech tokens
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        // Create state and run inference
        let mut state = self.ctx.create_state()?;
        state.full(params, audio_16k)?;

        // Collect all segments
        let num_segments = state.full_n_segments()?;
        let mut text = String::new();
        let mut word_timestamps = Vec::new();

        for i in 0..num_segments {
            if let Ok(segment) = state.full_get_segment_text(i) {
                text.push_str(&segment);
                text.push(' ');
            }

            // Extract word timestamps if enabled
            if enable_timestamps {
                if let Ok(num_tokens) = state.full_n_tokens(i) {
                    for j in 0..num_tokens {
                        // Get token text
                        if let Ok(token_text) = state.full_get_token_text(i, j) {
                            let token_str = token_text.trim();

                            // Skip empty tokens and special tokens
                            if token_str.is_empty() || token_str.starts_with('[') {
                                continue;
                            }

                            // Get timing info
                            if let Ok(token_data) = state.full_get_token_data(i, j) {
                                // Convert from centiseconds to milliseconds
                                let start_ms = (token_data.t0 as i64) * 10;
                                let end_ms = (token_data.t1 as i64) * 10;

                                word_timestamps.push(WordTimestamp {
                                    word: token_str.to_string(),
                                    start_ms,
                                    end_ms,
                                    probability: token_data.p,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            word_timestamps,
        })
    }

    /// Get word timestamps as tuples for prosody analysis
    pub fn get_word_timestamp_tuples(result: &TranscriptionResult) -> Vec<(String, i64, i64)> {
        result
            .word_timestamps
            .iter()
            .map(|w| (w.word.clone(), w.start_ms, w.end_ms))
            .collect()
    }
}
