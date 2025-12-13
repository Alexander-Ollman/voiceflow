//! Main processing pipeline: Audio → Transcription → LLM Formatting

use crate::{
    config::{Config, SttEngine as SttEngineConfig},
    llm::LlmEngine,
    prosody::{self, ProsodyHints, replace_voice_commands, concatenate_spelled_words_aggressive, ReplacementDictionary},
    transcribe::{WhisperEngine, MoonshineEngine, TranscriptionResult},
};
use anyhow::{Context, Result};
use std::time::Instant;

/// Error recovery configuration
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Maximum retries for LLM initialization
    pub llm_max_retries: u32,
    /// Delay between retries in milliseconds
    pub retry_delay_ms: u64,
    /// Whether to fall back to transcription-only mode on LLM failure
    pub fallback_to_transcribe_only: bool,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            llm_max_retries: 2,
            retry_delay_ms: 500,
            fallback_to_transcribe_only: true,
        }
    }
}

/// Pipeline error with actionable context
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("STT model not found: {path}. Run 'voiceflow setup' or download from the app's Settings → Models tab")]
    SttModelNotFound { path: String },

    #[error("LLM model not found: {path}. Run 'voiceflow setup' or download from the app's Settings → Models tab")]
    LlmModelNotFound { path: String },

    #[error("STT initialization failed after {attempts} attempts: {message}")]
    SttInitFailed { attempts: u32, message: String },

    #[error("LLM initialization failed after {attempts} attempts: {message}")]
    LlmInitFailed { attempts: u32, message: String },

    #[error("Transcription failed: {message}")]
    TranscriptionFailed { message: String },

    #[error("LLM formatting failed: {message}. Returning raw transcript.")]
    LlmFormattingFailed { message: String },

    #[error("Audio too short: {duration_ms}ms (minimum: 100ms)")]
    AudioTooShort { duration_ms: u64 },
}

/// Unified STT engine wrapper
enum SttEngine {
    Whisper(WhisperEngine),
    Moonshine(MoonshineEngine),
}

impl SttEngine {
    fn new(config: &Config) -> Result<Self> {
        match config.stt_engine {
            SttEngineConfig::Whisper => {
                tracing::info!("Using Whisper STT engine: {:?}", config.whisper_model);
                Ok(Self::Whisper(WhisperEngine::new(config)?))
            }
            SttEngineConfig::Moonshine => {
                tracing::info!("Using Moonshine STT engine: {:?}", config.moonshine_model);
                Ok(Self::Moonshine(MoonshineEngine::new(config)?))
            }
        }
    }

    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        match self {
            Self::Whisper(engine) => engine.transcribe(audio),
            Self::Moonshine(engine) => engine.transcribe(audio),
        }
    }

    fn transcribe_with_timestamps(&mut self, audio: &[f32], enable_timestamps: bool) -> Result<TranscriptionResult> {
        eprintln!("SttEngine: transcribe_with_timestamps called, {} samples", audio.len());
        match self {
            Self::Whisper(engine) => {
                eprintln!("SttEngine: Using Whisper engine");
                engine.transcribe_with_timestamps(audio, enable_timestamps)
            },
            Self::Moonshine(engine) => {
                eprintln!("SttEngine: Using Moonshine engine");
                engine.transcribe_with_timestamps(audio, enable_timestamps)
            },
        }
    }

    /// Check if this engine supports word-level timestamps
    fn supports_timestamps(&self) -> bool {
        matches!(self, Self::Whisper(_))
    }
}

/// Result from the processing pipeline
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Raw transcript from Whisper
    pub raw_transcript: String,
    /// LLM-formatted output
    pub formatted_text: String,
    /// Processing timings
    pub timings: Timings,
    /// Prosody analysis results (if enabled)
    pub prosody_hints: Option<ProsodyHints>,
}

/// Processing time breakdown
#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub transcription_ms: u64,
    pub prosody_ms: u64,
    pub llm_formatting_ms: u64,
    pub total_ms: u64,
}

/// Prosody analysis options
#[derive(Debug, Clone, Default)]
pub struct ProsodyOptions {
    /// Enable voice command detection (e.g., "period" → ".")
    pub voice_commands: bool,
    /// Enable pause-based punctuation analysis
    pub pause_analysis: bool,
    /// Enable pitch contour analysis for question detection
    pub pitch_analysis: bool,
    /// Pass prosody hints to LLM for better decisions
    pub llm_hints: bool,
}

impl ProsodyOptions {
    /// Create options with all features enabled
    pub fn all() -> Self {
        Self {
            voice_commands: true,
            pause_analysis: true,
            pitch_analysis: true,
            llm_hints: true,
        }
    }

    /// Create options with no features enabled
    pub fn none() -> Self {
        Self::default()
    }

    /// Check if any prosody feature is enabled
    pub fn any_enabled(&self) -> bool {
        self.voice_commands || self.pause_analysis || self.pitch_analysis || self.llm_hints
    }
}

/// The main VoiceFlow pipeline
pub struct Pipeline {
    stt: SttEngine,
    llm: Option<LlmEngine>,
    config: Config,
    prosody_options: ProsodyOptions,
    replacements: ReplacementDictionary,
    recovery_config: RecoveryConfig,
    /// Tracks if LLM initialization has permanently failed
    llm_permanently_failed: bool,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration
    /// LLM is lazily initialized on first use
    pub fn new(config: &Config) -> Result<Self> {
        Self::new_with_recovery(config, RecoveryConfig::default())
    }

    /// Create a new pipeline with custom recovery configuration
    pub fn new_with_recovery(config: &Config, recovery_config: RecoveryConfig) -> Result<Self> {
        tracing::info!("Initializing VoiceFlow pipeline");
        tracing::info!("  STT engine: {}", config.stt_engine.display_name());
        tracing::info!("  LLM model: {}", config.llm_model.display_name());

        let stt = SttEngine::new(config)
            .context("Failed to initialize speech-to-text engine")?;
        let replacements = ReplacementDictionary::load_default();
        tracing::info!("  Loaded {} text replacements", replacements.len());

        Ok(Self {
            stt,
            llm: None, // Lazy initialization
            config: config.clone(),
            prosody_options: ProsodyOptions::all(), // Enable all by default
            replacements,
            recovery_config,
            llm_permanently_failed: false,
        })
    }

    /// Set prosody analysis options
    pub fn set_prosody_options(&mut self, options: ProsodyOptions) {
        self.prosody_options = options;
    }

    /// Set recovery configuration
    pub fn set_recovery_config(&mut self, config: RecoveryConfig) {
        self.recovery_config = config;
    }

    /// Get or initialize the LLM engine with retry logic
    fn get_llm(&mut self) -> Result<&mut LlmEngine> {
        // If LLM has permanently failed, return error immediately
        if self.llm_permanently_failed {
            anyhow::bail!(PipelineError::LlmInitFailed {
                attempts: self.recovery_config.llm_max_retries,
                message: "LLM initialization previously failed permanently".to_string(),
            });
        }

        if self.llm.is_none() {
            let mut last_error = None;

            for attempt in 1..=self.recovery_config.llm_max_retries {
                tracing::info!("Initializing LLM engine (attempt {}/{})", attempt, self.recovery_config.llm_max_retries);

                match LlmEngine::new(&self.config) {
                    Ok(engine) => {
                        self.llm = Some(engine);
                        tracing::info!("LLM engine initialized successfully");
                        break;
                    }
                    Err(e) => {
                        tracing::warn!("LLM initialization attempt {} failed: {}", attempt, e);
                        last_error = Some(e);

                        if attempt < self.recovery_config.llm_max_retries {
                            std::thread::sleep(std::time::Duration::from_millis(
                                self.recovery_config.retry_delay_ms,
                            ));
                        }
                    }
                }
            }

            if self.llm.is_none() {
                self.llm_permanently_failed = true;
                let err_msg = last_error
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "Unknown error".to_string());

                return Err(PipelineError::LlmInitFailed {
                    attempts: self.recovery_config.llm_max_retries,
                    message: err_msg,
                }.into());
            }
        }

        Ok(self.llm.as_mut().unwrap())
    }

    /// Reset the LLM state, allowing re-initialization attempts
    pub fn reset_llm(&mut self) {
        self.llm = None;
        self.llm_permanently_failed = false;
        tracing::info!("LLM state reset, will attempt re-initialization on next use");
    }

    /// Check if the LLM is ready for use
    pub fn is_llm_ready(&self) -> bool {
        self.llm.is_some() && !self.llm_permanently_failed
    }

    /// Check if the pipeline can fall back to transcription-only mode
    pub fn can_fallback(&self) -> bool {
        self.recovery_config.fallback_to_transcribe_only
    }

    /// Process audio samples and return formatted text
    ///
    /// # Arguments
    /// * `audio` - PCM audio samples (f32, any sample rate - will be resampled)
    /// * `context` - Optional context hint (email, slack, code, etc.)
    pub fn process(&mut self, audio: &[f32], context: Option<&str>) -> Result<PipelineResult> {
        eprintln!("Pipeline: process() called with {} samples", audio.len());
        let start = Instant::now();

        // Determine if we need timestamps for prosody analysis (only if engine supports it)
        let need_timestamps = self.prosody_options.pause_analysis && self.stt.supports_timestamps();
        eprintln!("Pipeline: need_timestamps={}", need_timestamps);

        // Step 1: Transcribe audio with STT engine
        eprintln!("Pipeline: Starting transcription...");
        tracing::debug!("Transcribing {} samples", audio.len());
        let t1 = Instant::now();
        let transcription_result = self.stt.transcribe_with_timestamps(audio, need_timestamps)?;
        eprintln!("Pipeline: Transcription complete: '{}'", transcription_result.text);
        let transcription_ms = t1.elapsed().as_millis() as u64;
        tracing::debug!("Transcription took {}ms: {}", transcription_ms, transcription_result.text);

        let mut raw_transcript = transcription_result.text.clone();

        if raw_transcript.trim().is_empty() {
            return Ok(PipelineResult {
                raw_transcript: String::new(),
                formatted_text: String::new(),
                timings: Timings {
                    transcription_ms,
                    prosody_ms: 0,
                    llm_formatting_ms: 0,
                    total_ms: start.elapsed().as_millis() as u64,
                },
                prosody_hints: None,
            });
        }

        // Step 2: Prosody analysis
        let t2 = Instant::now();
        let mut prosody_hints = None;

        if self.prosody_options.any_enabled() {
            // Apply voice commands first (before prosody analysis)
            if self.prosody_options.voice_commands {
                raw_transcript = replace_voice_commands(&raw_transcript);
                tracing::debug!("After voice commands: {}", raw_transcript);
            }

            // Concatenate spelled-out letters (e.g., "S M O L L M" → "SMOLLM")
            // This runs before LLM to catch obvious patterns
            raw_transcript = concatenate_spelled_words_aggressive(&raw_transcript);
            tracing::debug!("After spelled word concatenation: {}", raw_transcript);

            // Apply user-defined replacements from replacements.toml
            raw_transcript = self.replacements.apply(&raw_transcript);
            tracing::debug!("After dictionary replacements: {}", raw_transcript);

            // Run prosody analysis (only use timestamps if available)
            if self.prosody_options.pause_analysis || self.prosody_options.pitch_analysis {
                let word_timestamps = if self.prosody_options.pause_analysis && !transcription_result.word_timestamps.is_empty() {
                    Some(WhisperEngine::get_word_timestamp_tuples(&transcription_result))
                } else {
                    None
                };

                let hints = prosody::analyze_prosody(audio, word_timestamps.as_deref());
                tracing::debug!("Prosody analysis: {:?}", hints);
                prosody_hints = Some(hints);
            }
        }
        let prosody_ms = t2.elapsed().as_millis() as u64;

        // Step 3: Get prompt for context
        let mut prompt_template = self.config.get_prompt_for_context(context);

        // Add prosody hints to prompt if enabled
        if self.prosody_options.llm_hints {
            if let Some(ref hints) = prosody_hints {
                let hint_context = hints.to_llm_context();
                if !hint_context.is_empty() {
                    prompt_template = format!("{}{}", prompt_template, hint_context);
                }
            }
        }

        // Step 4: Format with LLM (lazy init here, with fallback)
        tracing::debug!("Formatting with LLM (context: {:?})", context);
        let t3 = Instant::now();

        let (formatted_text, llm_formatting_ms) = match self.get_llm() {
            Ok(llm) => {
                match llm.format(&raw_transcript, &prompt_template) {
                    Ok(text) => {
                        let ms = t3.elapsed().as_millis() as u64;
                        tracing::debug!("LLM formatting took {}ms", ms);
                        (text, ms)
                    }
                    Err(e) => {
                        // LLM formatting failed - try fallback
                        tracing::warn!("LLM formatting failed: {}. Falling back to raw transcript.", e);
                        if self.recovery_config.fallback_to_transcribe_only {
                            (raw_transcript.clone(), 0)
                        } else {
                            return Err(PipelineError::LlmFormattingFailed {
                                message: e.to_string(),
                            }.into());
                        }
                    }
                }
            }
            Err(e) => {
                // LLM initialization failed - try fallback
                tracing::warn!("LLM initialization failed: {}. Falling back to raw transcript.", e);
                if self.recovery_config.fallback_to_transcribe_only {
                    (raw_transcript.clone(), 0)
                } else {
                    return Err(e);
                }
            }
        };

        let total_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            "Pipeline complete in {}ms (transcribe: {}ms, prosody: {}ms, format: {}ms)",
            total_ms,
            transcription_ms,
            prosody_ms,
            llm_formatting_ms
        );

        Ok(PipelineResult {
            raw_transcript,
            formatted_text,
            timings: Timings {
                transcription_ms,
                prosody_ms,
                llm_formatting_ms,
                total_ms,
            },
            prosody_hints,
        })
    }

    /// Process audio without LLM formatting (raw transcription only)
    pub fn transcribe_only(&mut self, audio: &[f32]) -> Result<PipelineResult> {
        let start = Instant::now();

        let mut raw_transcript = self.stt.transcribe(audio)?;
        let transcription_ms = start.elapsed().as_millis() as u64;

        // Apply voice commands even in transcribe-only mode
        if self.prosody_options.voice_commands {
            raw_transcript = replace_voice_commands(&raw_transcript);
        }

        // Concatenate spelled-out letters
        raw_transcript = concatenate_spelled_words_aggressive(&raw_transcript);

        // Apply user-defined replacements
        raw_transcript = self.replacements.apply(&raw_transcript);

        Ok(PipelineResult {
            raw_transcript: raw_transcript.clone(),
            formatted_text: raw_transcript,
            timings: Timings {
                transcription_ms,
                prosody_ms: 0,
                llm_formatting_ms: 0,
                total_ms: transcription_ms,
            },
            prosody_hints: None,
        })
    }
}
