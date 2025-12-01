//! Whisper speech-to-text engine

use crate::audio::resample_to_16khz;
use crate::config::Config;
use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Whisper-based speech-to-text engine
pub struct WhisperEngine {
    ctx: WhisperContext,
    input_sample_rate: u32,
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

        Ok(Self {
            ctx,
            input_sample_rate: config.audio.sample_rate,
        })
    }

    /// Transcribe audio samples to text
    ///
    /// # Arguments
    /// * `audio` - PCM f32 samples at the configured input sample rate
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        // Resample to 16kHz if needed
        let audio_16k = resample_to_16khz(audio, self.input_sample_rate)?;

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

        // Suppress non-speech tokens
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        // Create state and run inference
        let mut state = self.ctx.create_state()?;
        state.full(params, &audio_16k)?;

        // Collect all segments
        let num_segments = state.full_n_segments()?;
        let mut text = String::new();

        for i in 0..num_segments {
            if let Ok(segment) = state.full_get_segment_text(i) {
                text.push_str(&segment);
                text.push(' ');
            }
        }

        Ok(text.trim().to_string())
    }
}
