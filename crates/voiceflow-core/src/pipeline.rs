//! Main processing pipeline: Audio → Transcription → LLM Formatting

use crate::{config::Config, llm::LlmEngine, transcribe::WhisperEngine};
use anyhow::Result;
use std::time::Instant;

/// Result from the processing pipeline
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Raw transcript from Whisper
    pub raw_transcript: String,
    /// LLM-formatted output
    pub formatted_text: String,
    /// Processing timings
    pub timings: Timings,
}

/// Processing time breakdown
#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub transcription_ms: u64,
    pub llm_formatting_ms: u64,
    pub total_ms: u64,
}

/// The main VoiceFlow pipeline
pub struct Pipeline {
    whisper: WhisperEngine,
    llm: LlmEngine,
    config: Config,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: &Config) -> Result<Self> {
        tracing::info!("Initializing VoiceFlow pipeline");
        tracing::info!("  Whisper model: {:?}", config.whisper_model);
        tracing::info!("  LLM model: {}", config.llm_model.display_name());

        let whisper = WhisperEngine::new(config)?;
        let llm = LlmEngine::new(config)?;

        Ok(Self {
            whisper,
            llm,
            config: config.clone(),
        })
    }

    /// Process audio samples and return formatted text
    ///
    /// # Arguments
    /// * `audio` - PCM audio samples (f32, any sample rate - will be resampled)
    /// * `context` - Optional context hint (email, slack, code, etc.)
    pub fn process(&mut self, audio: &[f32], context: Option<&str>) -> Result<PipelineResult> {
        let start = Instant::now();

        // Step 1: Transcribe audio with Whisper
        tracing::debug!("Transcribing {} samples", audio.len());
        let t1 = Instant::now();
        let raw_transcript = self.whisper.transcribe(audio)?;
        let transcription_ms = t1.elapsed().as_millis() as u64;
        tracing::debug!("Transcription took {}ms: {}", transcription_ms, raw_transcript);

        if raw_transcript.trim().is_empty() {
            return Ok(PipelineResult {
                raw_transcript: String::new(),
                formatted_text: String::new(),
                timings: Timings {
                    transcription_ms,
                    llm_formatting_ms: 0,
                    total_ms: start.elapsed().as_millis() as u64,
                },
            });
        }

        // Step 2: Get prompt for context
        let prompt_template = self.config.get_prompt_for_context(context);

        // Step 3: Format with LLM
        tracing::debug!("Formatting with LLM (context: {:?})", context);
        let t2 = Instant::now();
        let formatted_text = self.llm.format(&raw_transcript, &prompt_template)?;
        let llm_formatting_ms = t2.elapsed().as_millis() as u64;
        tracing::debug!("LLM formatting took {}ms", llm_formatting_ms);

        let total_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            "Pipeline complete in {}ms (transcribe: {}ms, format: {}ms)",
            total_ms,
            transcription_ms,
            llm_formatting_ms
        );

        Ok(PipelineResult {
            raw_transcript,
            formatted_text,
            timings: Timings {
                transcription_ms,
                llm_formatting_ms,
                total_ms,
            },
        })
    }

    /// Process audio without LLM formatting (raw transcription only)
    pub fn transcribe_only(&mut self, audio: &[f32]) -> Result<PipelineResult> {
        let start = Instant::now();

        let raw_transcript = self.whisper.transcribe(audio)?;
        let transcription_ms = start.elapsed().as_millis() as u64;

        Ok(PipelineResult {
            raw_transcript: raw_transcript.clone(),
            formatted_text: raw_transcript,
            timings: Timings {
                transcription_ms,
                llm_formatting_ms: 0,
                total_ms: transcription_ms,
            },
        })
    }
}
