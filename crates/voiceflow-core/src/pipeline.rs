//! Main processing pipeline: Audio → Transcription → LLM Formatting

use crate::{
    config::{Config, SttEngine as SttEngineConfig},
    llm::LlmEngine,
    prosody::{self, ProsodyHints, replace_voice_commands, concatenate_spelled_words_aggressive, ReplacementDictionary, fix_tokenization_artifacts, remove_filler_words},
    text_normalize,
    transcribe::{MoonshineEngine, TranscriptionResult},
};
#[cfg(feature = "whisper")]
use crate::transcribe::WhisperEngine;
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
    #[cfg(feature = "whisper")]
    Whisper(WhisperEngine),
    Moonshine(MoonshineEngine),
}

impl SttEngine {
    fn new(config: &Config) -> Result<Option<Self>> {
        match config.stt_engine {
            SttEngineConfig::Whisper => {
                #[cfg(feature = "whisper")]
                {
                    tracing::info!("Using Whisper STT engine: {:?}", config.whisper_model);
                    Ok(Some(Self::Whisper(WhisperEngine::new(config)?)))
                }
                #[cfg(not(feature = "whisper"))]
                {
                    anyhow::bail!("Whisper STT engine not available (compiled without 'whisper' feature)")
                }
            }
            SttEngineConfig::Moonshine => {
                tracing::info!("Using Moonshine STT engine: {:?}", config.moonshine_model);
                Ok(Some(Self::Moonshine(MoonshineEngine::new(config)?)))
            }
            SttEngineConfig::Qwen3Asr => {
                tracing::info!("Using Qwen3-ASR (external Python daemon) - no Rust STT engine needed");
                Ok(None)
            }
        }
    }

    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        match self {
            #[cfg(feature = "whisper")]
            Self::Whisper(engine) => engine.transcribe(audio),
            Self::Moonshine(engine) => engine.transcribe(audio),
        }
    }

    fn transcribe_with_timestamps(&mut self, audio: &[f32], enable_timestamps: bool) -> Result<TranscriptionResult> {
        match self {
            #[cfg(feature = "whisper")]
            Self::Whisper(engine) => engine.transcribe_with_timestamps(audio, enable_timestamps),
            Self::Moonshine(engine) => engine.transcribe_with_timestamps(audio, enable_timestamps),
        }
    }

    /// Check if this engine supports word-level timestamps
    fn supports_timestamps(&self) -> bool {
        match self {
            #[cfg(feature = "whisper")]
            Self::Whisper(_) => true,
            Self::Moonshine(_) => false,
        }
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
    /// Time to first visible token (from start of audio input)
    pub ttft_ms: u64,
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
    /// STT engine - None when using external STT (Qwen3-ASR)
    stt: Option<SttEngine>,
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

        // Audio-direct mode: skip STT entirely — audio goes directly to LLM
        let stt = if config.is_audio_direct_mode() {
            tracing::info!("  Audio-direct mode - STT skipped (LLM handles transcription)");
            None
        } else {
            let s = SttEngine::new(config)
                .context("Failed to initialize speech-to-text engine")?;
            if s.is_none() {
                tracing::info!("  External STT engine selected - Rust STT skipped");
            }
            s
        };
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

    /// Unload the STT engine from memory
    /// The pipeline will not be able to transcribe audio until reinitialized
    pub fn unload_stt(&mut self) {
        if self.stt.is_some() {
            tracing::info!("Unloading STT engine from memory");
            self.stt = None;
        }
    }

    /// Reset the LLM state, allowing re-initialization attempts
    pub fn reset_llm(&mut self) {
        self.llm = None;
        self.llm_permanently_failed = false;
        tracing::info!("LLM state reset, will attempt re-initialization on next use");
    }

    /// Unload the LLM from memory
    /// The LLM will be reloaded on next use
    pub fn unload_llm(&mut self) {
        if self.llm.is_some() {
            tracing::info!("Unloading LLM from memory");
            self.llm = None;
            // Don't set llm_permanently_failed - we want to allow reloading
            tracing::info!("LLM unloaded successfully");
        }
    }

    /// Unload all models from memory (LLM and STT)
    /// This releases significant memory but requires reinit before next use
    pub fn unload_all(&mut self) {
        tracing::info!("Unloading all models from memory");

        self.unload_llm();
        self.unload_stt();

        tracing::info!("All models unloaded");
    }

    /// Check if the LLM is ready for use
    pub fn is_llm_ready(&self) -> bool {
        self.llm.is_some() && !self.llm_permanently_failed
    }

    /// Add a user-learned correction to the replacement dictionary at runtime.
    /// The correction is applied by the existing `replacements.apply()` step.
    pub fn add_user_correction(&mut self, original: &str, corrected: &str) {
        self.replacements.add_user_correction(original.to_string(), corrected.to_string());
    }

    /// Remove a user-learned correction from the replacement dictionary at runtime.
    /// Returns true if the correction was found and removed.
    pub fn remove_user_correction(&mut self, original: &str) -> bool {
        self.replacements.remove_user_correction(original)
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
        let start = Instant::now();

        let stt = self.stt.as_mut().ok_or_else(|| {
            anyhow::anyhow!("STT engine not available (using external STT). Use process_text() instead.")
        })?;

        // Determine if we need timestamps for prosody analysis (only if engine supports it)
        let need_timestamps = self.prosody_options.pause_analysis && stt.supports_timestamps();

        // Step 1: Transcribe audio with STT engine
        tracing::debug!("Transcribing {} samples", audio.len());
        let t1 = Instant::now();
        let transcription_result = stt.transcribe_with_timestamps(audio, need_timestamps)?;
        let transcription_ms = t1.elapsed().as_millis() as u64;
        tracing::debug!("Transcription took {}ms: {}", transcription_ms, transcription_result.text);

        let mut raw_transcript = transcription_result.text.clone();

        // Fix tokenization artifacts from STT (mid-word punctuation, mid-word caps)
        // This should happen early, before other processing
        raw_transcript = fix_tokenization_artifacts(&raw_transcript);
        tracing::debug!("After tokenization fix: {}", raw_transcript);

        // Remove filler words (um, uh, ah, hmm, etc.)
        raw_transcript = remove_filler_words(&raw_transcript);
        tracing::debug!("After filler removal: {}", raw_transcript);

        if raw_transcript.trim().is_empty() {
            return Ok(PipelineResult {
                raw_transcript: String::new(),
                formatted_text: String::new(),
                timings: Timings {
                    transcription_ms,
                    prosody_ms: 0,
                    llm_formatting_ms: 0,
                    total_ms: start.elapsed().as_millis() as u64,
                    ttft_ms: 0,
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
            }

            // Concatenate spelled-out letters (e.g., "S M O L L M" → "SMOLLM")
            // This runs before LLM to catch obvious patterns
            raw_transcript = concatenate_spelled_words_aggressive(&raw_transcript);

            // Apply user-defined replacements from replacements.toml
            raw_transcript = self.replacements.apply(&raw_transcript);

            // Run prosody analysis (only use timestamps if available)
            if self.prosody_options.pause_analysis || self.prosody_options.pitch_analysis {
                let word_timestamps: Option<Vec<(String, i64, i64)>> = if self.prosody_options.pause_analysis && !transcription_result.word_timestamps.is_empty() {
                    #[cfg(feature = "whisper")]
                    {
                        Some(WhisperEngine::get_word_timestamp_tuples(&transcription_result))
                    }
                    #[cfg(not(feature = "whisper"))]
                    {
                        None
                    }
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
        // If context is a long string (from Swift UI with embedded formatting instructions,
        // visual context, input context, etc.), inject it into the base prompt BEFORE the
        // "## Input" section so the LLM sees context before the transcript/output markers.
        // Short strings like "email"/"code" are lookup keys for named prompt files.
        let mut prompt_template = if let Some(ctx) = context {
            if ctx.len() > 50 {
                // Long context = Swift-side prompt with embedded sections
                let base = self.config.get_prompt_for_context(None);
                // Insert before ## Input so the LLM sees: instructions → context → transcript → output
                if let Some(input_pos) = base.find("## Input") {
                    format!("{}\n{}\n\n{}", &base[..input_pos].trim_end(), ctx, &base[input_pos..])
                } else {
                    format!("{}\n{}", base, ctx)
                }
            } else {
                self.config.get_prompt_for_context(Some(ctx))
            }
        } else {
            self.config.get_prompt_for_context(None)
        };

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
        // Uses streaming internally to capture TTFT (time to first visible token)
        tracing::debug!("Formatting with LLM (context: {:?})", context);
        let t3 = Instant::now();

        let (formatted_text, llm_formatting_ms, llm_ttft_ms) = match self.get_llm() {
            Ok(llm) => {
                match llm.format_with_ttft(&raw_transcript, &prompt_template) {
                    Ok((text, ttft)) => {
                        let ms = t3.elapsed().as_millis() as u64;
                        tracing::debug!("LLM formatting took {}ms (TTFT: {}ms)", ms, ttft);
                        (text, ms, ttft)
                    }
                    Err(e) => {
                        // LLM formatting failed - try fallback
                        tracing::warn!("LLM formatting failed: {}. Falling back to raw transcript.", e);
                        if self.recovery_config.fallback_to_transcribe_only {
                            (raw_transcript.clone(), 0, 0)
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
                    (raw_transcript.clone(), 0, 0)
                } else {
                    return Err(e);
                }
            }
        };

        // Apply mid-sentence continuation fix (lowercase first char when continuing)
        let formatted_text = if is_mid_sentence_continuation(context) {
            tracing::debug!("Applying mid-sentence continuation casing fix");
            fix_continuation_casing(&formatted_text)
        } else {
            formatted_text
        };

        let total_ms = start.elapsed().as_millis() as u64;
        // TTFT from audio input = STT time + prosody time + LLM TTFT
        let ttft_ms = transcription_ms + prosody_ms + llm_ttft_ms;
        tracing::info!(
            "Pipeline complete in {}ms (transcribe: {}ms, prosody: {}ms, format: {}ms, ttft: {}ms)",
            total_ms,
            transcription_ms,
            prosody_ms,
            llm_formatting_ms,
            ttft_ms,
        );

        Ok(PipelineResult {
            raw_transcript,
            formatted_text,
            timings: Timings {
                transcription_ms,
                prosody_ms,
                llm_formatting_ms,
                total_ms,
                ttft_ms,
            },
            prosody_hints,
        })
    }

    /// Process audio without LLM formatting (raw transcription only)
    pub fn transcribe_only(&mut self, audio: &[f32]) -> Result<PipelineResult> {
        let start = Instant::now();

        let stt = self.stt.as_mut().ok_or_else(|| {
            anyhow::anyhow!("STT engine not available (using external STT)")
        })?;

        let mut raw_transcript = stt.transcribe(audio)?;
        let transcription_ms = start.elapsed().as_millis() as u64;

        // Fix tokenization artifacts from STT (mid-word punctuation, mid-word caps)
        raw_transcript = fix_tokenization_artifacts(&raw_transcript);

        // Remove filler words
        raw_transcript = remove_filler_words(&raw_transcript);

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
                ttft_ms: transcription_ms, // In transcribe-only mode, TTFT = transcription time
            },
            prosody_hints: None,
        })
    }

    /// Process pre-transcribed text through post-processing + LLM formatting.
    /// Used when an external STT engine (e.g. Qwen3-ASR) provides the raw transcript.
    pub fn process_text(&mut self, text: &str, context: Option<&str>) -> Result<PipelineResult> {
        let start = Instant::now();

        let mut raw_transcript = text.to_string();

        // Fix tokenization artifacts
        raw_transcript = fix_tokenization_artifacts(&raw_transcript);

        // Remove filler words
        raw_transcript = remove_filler_words(&raw_transcript);

        if raw_transcript.trim().is_empty() {
            return Ok(PipelineResult {
                raw_transcript: String::new(),
                formatted_text: String::new(),
                timings: Timings {
                    transcription_ms: 0,
                    prosody_ms: 0,
                    llm_formatting_ms: 0,
                    total_ms: start.elapsed().as_millis() as u64,
                    ttft_ms: 0,
                },
                prosody_hints: None,
            });
        }

        // Post-processing (voice commands, spelled words, replacements)
        let t2 = Instant::now();
        if self.prosody_options.voice_commands {
            raw_transcript = replace_voice_commands(&raw_transcript);
        }
        raw_transcript = concatenate_spelled_words_aggressive(&raw_transcript);
        raw_transcript = self.replacements.apply(&raw_transcript);
        let prosody_ms = t2.elapsed().as_millis() as u64;

        // LLM formatting
        // Same logic as process(): inject long context before ## Input section
        let prompt_template = if let Some(ctx) = context {
            if ctx.len() > 50 {
                let base = self.config.get_prompt_for_context(None);
                if let Some(input_pos) = base.find("## Input") {
                    format!("{}\n{}\n\n{}", &base[..input_pos].trim_end(), ctx, &base[input_pos..])
                } else {
                    format!("{}\n{}", base, ctx)
                }
            } else {
                self.config.get_prompt_for_context(Some(ctx))
            }
        } else {
            self.config.get_prompt_for_context(None)
        };

        tracing::debug!("Formatting external transcript with LLM (context: {:?})", context);
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
                tracing::warn!("LLM initialization failed: {}. Falling back to raw transcript.", e);
                if self.recovery_config.fallback_to_transcribe_only {
                    (raw_transcript.clone(), 0)
                } else {
                    return Err(e);
                }
            }
        };

        // Apply mid-sentence continuation fix (lowercase first char when continuing)
        let formatted_text = if is_mid_sentence_continuation(context) {
            tracing::debug!("Applying mid-sentence continuation casing fix");
            fix_continuation_casing(&formatted_text)
        } else {
            formatted_text
        };

        let total_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            "process_text complete in {}ms (prosody: {}ms, format: {}ms)",
            total_ms, prosody_ms, llm_formatting_ms
        );

        Ok(PipelineResult {
            raw_transcript,
            formatted_text,
            timings: Timings {
                transcription_ms: 0,
                prosody_ms,
                llm_formatting_ms,
                total_ms,
                ttft_ms: 0,
            },
            prosody_hints: None,
        })
    }

    /// Process pre-transcribed text with an image through post-processing + multimodal LLM.
    /// Used for visual context dictation (Control+Option+Space hotkey).
    pub fn process_text_with_image(
        &mut self,
        text: &str,
        image_data: &[u8],
        context: Option<&str>,
    ) -> Result<PipelineResult> {
        let start = Instant::now();

        let mut raw_transcript = text.to_string();

        // Same pre-processing as process_text()
        raw_transcript = fix_tokenization_artifacts(&raw_transcript);
        raw_transcript = remove_filler_words(&raw_transcript);

        if raw_transcript.trim().is_empty() {
            return Ok(PipelineResult {
                raw_transcript: String::new(),
                formatted_text: String::new(),
                timings: Timings {
                    transcription_ms: 0,
                    prosody_ms: 0,
                    llm_formatting_ms: 0,
                    total_ms: start.elapsed().as_millis() as u64,
                    ttft_ms: 0,
                },
                prosody_hints: None,
            });
        }

        // Post-processing (voice commands, spelled words, replacements)
        let t2 = Instant::now();
        if self.prosody_options.voice_commands {
            raw_transcript = replace_voice_commands(&raw_transcript);
        }
        raw_transcript = concatenate_spelled_words_aggressive(&raw_transcript);
        raw_transcript = self.replacements.apply(&raw_transcript);
        let prosody_ms = t2.elapsed().as_millis() as u64;

        // Build prompt template (same logic as process_text)
        let prompt_template = if let Some(ctx) = context {
            if ctx.len() > 50 {
                let base = self.config.get_prompt_for_context(None);
                if let Some(input_pos) = base.find("## Input") {
                    format!("{}\n{}\n\n{}", &base[..input_pos].trim_end(), ctx, &base[input_pos..])
                } else {
                    format!("{}\n{}", base, ctx)
                }
            } else {
                self.config.get_prompt_for_context(Some(ctx))
            }
        } else {
            self.config.get_prompt_for_context(None)
        };

        tracing::debug!("Formatting with multimodal LLM (image: {} bytes)", image_data.len());
        let t3 = Instant::now();

        let (formatted_text, llm_formatting_ms) = match self.get_llm() {
            Ok(llm) => {
                if llm.supports_multimodal() {
                    match llm.format_with_image(&raw_transcript, &prompt_template, image_data) {
                        Ok(text) => {
                            let ms = t3.elapsed().as_millis() as u64;
                            tracing::debug!("Multimodal LLM formatting took {}ms", ms);
                            (text, ms)
                        }
                        Err(e) => {
                            tracing::warn!("Multimodal LLM formatting failed: {}. Falling back to text-only.", e);
                            // Fall back to text-only formatting
                            match llm.format(&raw_transcript, &prompt_template) {
                                Ok(text) => (text, t3.elapsed().as_millis() as u64),
                                Err(e2) => {
                                    tracing::warn!("Text-only fallback also failed: {}", e2);
                                    (raw_transcript.clone(), 0)
                                }
                            }
                        }
                    }
                } else {
                    tracing::warn!("Multimodal not supported, falling back to text-only LLM");
                    match llm.format(&raw_transcript, &prompt_template) {
                        Ok(text) => (text, t3.elapsed().as_millis() as u64),
                        Err(e) => {
                            tracing::warn!("LLM formatting failed: {}", e);
                            (raw_transcript.clone(), 0)
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("LLM initialization failed: {}. Returning raw transcript.", e);
                (raw_transcript.clone(), 0)
            }
        };

        let formatted_text = if is_mid_sentence_continuation(context) {
            fix_continuation_casing(&formatted_text)
        } else {
            formatted_text
        };

        let total_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            "process_text_with_image complete in {}ms (prosody: {}ms, format: {}ms)",
            total_ms, prosody_ms, llm_formatting_ms
        );

        Ok(PipelineResult {
            raw_transcript,
            formatted_text,
            timings: Timings {
                transcription_ms: 0,
                prosody_ms,
                llm_formatting_ms,
                total_ms,
                ttft_ms: 0,
            },
            prosody_hints: None,
        })
    }

    /// Process pre-transcribed text with LLM token streaming.
    /// Identical pre-processing to `process_text()`, but calls `format_streaming()` for
    /// real-time token delivery via `on_token` callback.
    pub fn process_text_streaming(
        &mut self,
        text: &str,
        context: Option<&str>,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> Result<PipelineResult> {
        let start = Instant::now();

        let mut raw_transcript = text.to_string();

        // Fix tokenization artifacts
        raw_transcript = fix_tokenization_artifacts(&raw_transcript);

        // Remove filler words
        raw_transcript = remove_filler_words(&raw_transcript);

        if raw_transcript.trim().is_empty() {
            return Ok(PipelineResult {
                raw_transcript: String::new(),
                formatted_text: String::new(),
                timings: Timings {
                    transcription_ms: 0,
                    prosody_ms: 0,
                    llm_formatting_ms: 0,
                    total_ms: start.elapsed().as_millis() as u64,
                    ttft_ms: 0,
                },
                prosody_hints: None,
            });
        }

        // Post-processing (voice commands, spelled words, replacements)
        let t2 = Instant::now();
        if self.prosody_options.voice_commands {
            raw_transcript = replace_voice_commands(&raw_transcript);
        }
        raw_transcript = concatenate_spelled_words_aggressive(&raw_transcript);
        raw_transcript = self.replacements.apply(&raw_transcript);
        let prosody_ms = t2.elapsed().as_millis() as u64;

        // Build prompt template (same logic as process_text)
        let prompt_template = if let Some(ctx) = context {
            if ctx.len() > 50 {
                let base = self.config.get_prompt_for_context(None);
                if let Some(input_pos) = base.find("## Input") {
                    format!("{}\n{}\n\n{}", &base[..input_pos].trim_end(), ctx, &base[input_pos..])
                } else {
                    format!("{}\n{}", base, ctx)
                }
            } else {
                self.config.get_prompt_for_context(Some(ctx))
            }
        } else {
            self.config.get_prompt_for_context(None)
        };

        tracing::debug!("Streaming format of external transcript (context: {:?})", context);
        let t3 = Instant::now();

        let (formatted_text, llm_formatting_ms) = match self.get_llm() {
            Ok(llm) => {
                match llm.format_streaming(&raw_transcript, &prompt_template, on_token) {
                    Ok(text) => {
                        let ms = t3.elapsed().as_millis() as u64;
                        tracing::debug!("LLM streaming formatting took {}ms", ms);
                        (text, ms)
                    }
                    Err(e) => {
                        tracing::warn!("LLM streaming formatting failed: {}. Falling back to raw transcript.", e);
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
                tracing::warn!("LLM initialization failed: {}. Falling back to raw transcript.", e);
                if self.recovery_config.fallback_to_transcribe_only {
                    (raw_transcript.clone(), 0)
                } else {
                    return Err(e);
                }
            }
        };

        // Apply mid-sentence continuation fix
        let formatted_text = if is_mid_sentence_continuation(context) {
            tracing::debug!("Applying mid-sentence continuation casing fix");
            fix_continuation_casing(&formatted_text)
        } else {
            formatted_text
        };

        let total_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            "process_text_streaming complete in {}ms (prosody: {}ms, format: {}ms)",
            total_ms, prosody_ms, llm_formatting_ms
        );

        Ok(PipelineResult {
            raw_transcript,
            formatted_text,
            timings: Timings {
                transcription_ms: 0,
                prosody_ms,
                llm_formatting_ms,
                total_ms,
                ttft_ms: 0,
            },
            prosody_hints: None,
        })
    }

    /// Process audio directly through a single audio-native model (e.g., Gemma 4).
    /// Bypasses the separate STT step — the LLM handles both transcription and formatting.
    ///
    /// # Arguments
    /// * `audio` - PCM audio samples (f32, 16kHz)
    /// * `context` - Optional context hint for the prompt
    pub fn process_audio_direct(
        &mut self,
        audio: &[f32],
        context: Option<&str>,
    ) -> Result<PipelineResult> {
        let start = Instant::now();

        // Validate audio length
        let duration_secs = audio.len() as f32 / 16000.0;
        if duration_secs < 0.1 {
            return Err(PipelineError::AudioTooShort {
                duration_ms: (duration_secs * 1000.0) as u64,
            }
            .into());
        }
        if duration_secs > 30.0 {
            tracing::warn!(
                "Audio duration {:.1}s exceeds 30s limit for audio-direct mode, will be truncated by backend",
                duration_secs
            );
        }

        // Build prompt (audio_direct template, no transcript placeholder needed).
        // Personal dictionary is injected as a [PERSONAL_DICTIONARY] block so
        // the prompt's Spelling Resolution section can match against it.
        let prompt_template = self.config.get_prompt_for_context(Some("audio_direct"));
        let prompt = if !self.config.personal_dictionary.is_empty() {
            let dict_str = self.config.personal_dictionary.join(", ");
            prompt_template.replace(
                "{personal_dictionary}",
                &format!("\n[PERSONAL_DICTIONARY]\n{}\n", dict_str),
            )
        } else {
            prompt_template.replace("{personal_dictionary}", "")
        };

        // Inject additional context if provided
        let prompt = if let Some(ctx) = context {
            if ctx.len() > 50 {
                format!("{}\n\n{}", prompt, ctx)
            } else {
                prompt
            }
        } else {
            prompt
        };

        // Call the LLM with audio directly (using streaming to measure TTFT)
        tracing::debug!(
            "Audio-direct processing: {:.1}s audio, prompt {} chars",
            duration_secs,
            prompt.len()
        );
        let t_llm = Instant::now();

        let (formatted_text, llm_ttft_ms) = match self.get_llm() {
            Ok(llm) => {
                if !llm.supports_audio_direct() {
                    anyhow::bail!(
                        "Current LLM backend ({}) does not support audio-direct mode. \
                         Use a Gemma 4 model with mistral.rs backend.",
                        llm.backend_name()
                    );
                }
                match llm.transcribe_audio_direct_with_ttft(audio, 16000, &prompt) {
                    Ok((text, ttft)) => (text, ttft),
                    Err(e) => {
                        tracing::warn!("Audio-direct transcription failed: {}", e);
                        if self.recovery_config.fallback_to_transcribe_only {
                            (String::new(), 0)
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("LLM initialization failed for audio-direct: {}", e);
                return Err(e);
            }
        };
        let llm_ms = t_llm.elapsed().as_millis() as u64;

        // Apply deterministic post-processing (voice commands, spelling, replacements)
        let mut text = formatted_text.clone();
        if self.prosody_options.voice_commands {
            text = replace_voice_commands(&text);
        }
        text = concatenate_spelled_words_aggressive(&text);
        text = self.replacements.apply(&text);

        let total_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            "Audio-direct pipeline complete in {}ms (llm: {}ms, ttft: {}ms)",
            total_ms,
            llm_ms,
            llm_ttft_ms,
        );

        Ok(PipelineResult {
            raw_transcript: formatted_text, // In audio-direct mode, the LLM output IS the transcript
            formatted_text: text,
            timings: Timings {
                transcription_ms: 0, // No separate STT step
                prosody_ms: 0,
                llm_formatting_ms: llm_ms,
                total_ms,
                ttft_ms: llm_ttft_ms, // For audio-direct, TTFT = model TTFT
            },
            prosody_hints: None,
        })
    }

    /// Process audio through STT + deterministic normalization (no LLM).
    ///
    /// Uses `text_normalize::normalize()` instead of the LLM step.
    /// Expected latency: STT time + <5ms for normalization.
    pub fn process_deterministic(&mut self, audio: &[f32]) -> Result<PipelineResult> {
        let start = Instant::now();

        let stt = self.stt.as_mut().ok_or_else(|| {
            anyhow::anyhow!("STT engine not available (using external STT). Use process_text_deterministic() instead.")
        })?;

        let t1 = Instant::now();
        let raw_transcript = stt.transcribe(audio)?;
        let transcription_ms = t1.elapsed().as_millis() as u64;

        let result = text_normalize::normalize(&raw_transcript, &self.replacements);
        let total_ms = start.elapsed().as_millis() as u64;

        tracing::info!(
            "Deterministic pipeline complete in {}ms (STT: {}ms, normalize: {}ms, transforms: {})",
            total_ms,
            transcription_ms,
            total_ms - transcription_ms,
            result.transforms_applied,
        );

        Ok(PipelineResult {
            raw_transcript,
            formatted_text: result.text,
            timings: Timings {
                transcription_ms,
                prosody_ms: 0,
                llm_formatting_ms: 0,
                total_ms,
                ttft_ms: transcription_ms, // No LLM — TTFT = STT time
            },
            prosody_hints: None,
        })
    }

    /// Process pre-transcribed text through deterministic normalization (no LLM).
    ///
    /// Used when an external STT engine provides the raw transcript and we want
    /// fast deterministic post-processing without the LLM.
    pub fn process_text_deterministic(&mut self, text: &str) -> Result<PipelineResult> {
        let start = Instant::now();

        let result = text_normalize::normalize(text, &self.replacements);
        let total_ms = start.elapsed().as_millis() as u64;

        tracing::info!(
            "Deterministic text processing complete in {}ms (transforms: {})",
            total_ms,
            result.transforms_applied,
        );

        Ok(PipelineResult {
            raw_transcript: text.to_string(),
            formatted_text: result.text,
            timings: Timings {
                transcription_ms: 0,
                prosody_ms: 0,
                llm_formatting_ms: 0,
                total_ms,
                ttft_ms: 0,
            },
            prosody_hints: None,
        })
    }
}

/// Check if context indicates the user is continuing mid-sentence
fn is_mid_sentence_continuation(context: Option<&str>) -> bool {
    context.map_or(false, |c| c.contains("[MID_SENTENCE_CONTINUATION]"))
}

/// Fix capitalization for mid-sentence continuations.
///
/// When continuing from text that ends mid-sentence, the LLM often incorrectly
/// capitalizes the first word. This lowercases it unless it's "I"/contractions
/// or an acronym (all-caps like "API").
fn fix_continuation_casing(text: &str) -> String {
    let first_char = match text.chars().next() {
        Some(c) if c.is_uppercase() => c,
        _ => return text.to_string(),
    };

    // Get the first word (letters, digits, apostrophes for contractions)
    let first_word: String = text
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '\'')
        .collect();

    // Don't lowercase "I", "I'm", "I'll", "I've", "I'd"
    if first_word == "I" || first_word.starts_with("I'") {
        return text.to_string();
    }

    // Don't lowercase acronyms (all-caps, 2+ letters: "API", "URL", "CEO")
    if first_word.len() > 1 && first_word.chars().filter(|c| c.is_alphabetic()).all(|c| c.is_uppercase()) {
        return text.to_string();
    }

    // Lowercase the first character
    let lower: String = first_char.to_lowercase().collect();
    format!("{}{}", lower, &text[first_char.len_utf8()..])
}

#[cfg(test)]
mod continuation_tests {
    use super::*;

    // ========================================================================
    // Marker detection
    // ========================================================================

    #[test]
    fn test_mid_sentence_detected() {
        assert!(is_mid_sentence_continuation(Some("some stuff\n[MID_SENTENCE_CONTINUATION]")));
        assert!(!is_mid_sentence_continuation(Some("no marker here")));
        assert!(!is_mid_sentence_continuation(None));
    }

    #[test]
    fn test_mid_sentence_marker_anywhere_in_context() {
        let ctx = "formatting prompt\n[INPUT CONTEXT]\nsome text\n[MID_SENTENCE_CONTINUATION]";
        assert!(is_mid_sentence_continuation(Some(ctx)));
    }

    #[test]
    fn test_no_marker_after_period() {
        // Context that ends with a period should NOT have the marker
        assert!(!is_mid_sentence_continuation(Some("text ends here.")));
    }

    // ========================================================================
    // fix_continuation_casing — basic lowercasing
    // ========================================================================

    #[test]
    fn test_fix_casing_basic() {
        assert_eq!(fix_continuation_casing("Bought some milk."), "bought some milk.");
    }

    #[test]
    fn test_fix_casing_already_lower() {
        assert_eq!(fix_continuation_casing("bought some milk."), "bought some milk.");
    }

    #[test]
    fn test_fix_casing_empty() {
        assert_eq!(fix_continuation_casing(""), "");
    }

    #[test]
    fn test_fix_casing_single_char() {
        assert_eq!(fix_continuation_casing("A"), "a");
    }

    // ========================================================================
    // fix_continuation_casing — preserving special words
    // ========================================================================

    #[test]
    fn test_fix_casing_preserves_i() {
        assert_eq!(fix_continuation_casing("I think so."), "I think so.");
    }

    #[test]
    fn test_fix_casing_preserves_i_contractions() {
        assert_eq!(fix_continuation_casing("I'm not sure."), "I'm not sure.");
        assert_eq!(fix_continuation_casing("I'll be there."), "I'll be there.");
        assert_eq!(fix_continuation_casing("I've been waiting."), "I've been waiting.");
        assert_eq!(fix_continuation_casing("I'd rather not."), "I'd rather not.");
    }

    #[test]
    fn test_fix_casing_preserves_acronyms() {
        assert_eq!(fix_continuation_casing("API endpoint here"), "API endpoint here");
        assert_eq!(fix_continuation_casing("URL is broken"), "URL is broken");
        assert_eq!(fix_continuation_casing("CEO of the company"), "CEO of the company");
        assert_eq!(fix_continuation_casing("HTML is fine"), "HTML is fine");
    }

    // ========================================================================
    // fix_continuation_casing — realistic continuation scenarios
    // ========================================================================

    // Simulates: field has "I went to the store and" → user dictates → LLM outputs:
    #[test]
    fn test_continuation_after_and() {
        assert_eq!(
            fix_continuation_casing("Bought some milk and bread."),
            "bought some milk and bread."
        );
    }

    // Simulates: field has "The meeting is about" → user dictates:
    #[test]
    fn test_continuation_after_about() {
        assert_eq!(
            fix_continuation_casing("The new product launch."),
            "the new product launch."
        );
    }

    // Simulates: field has "I need to" → user dictates:
    #[test]
    fn test_continuation_after_to() {
        assert_eq!(
            fix_continuation_casing("Finish the report and send it."),
            "finish the report and send it."
        );
    }

    // Simulates: field has "She said that" → user dictates:
    #[test]
    fn test_continuation_after_that() {
        assert_eq!(
            fix_continuation_casing("Everything was fine."),
            "everything was fine."
        );
    }

    // Simulates: field has "We should probably" → user dictates:
    #[test]
    fn test_continuation_after_probably() {
        assert_eq!(
            fix_continuation_casing("Wait until tomorrow."),
            "wait until tomorrow."
        );
    }

    // Continuation starting with "I" — should stay uppercase
    // Simulates: field has "and then" → user dictates:
    #[test]
    fn test_continuation_starting_with_i() {
        assert_eq!(
            fix_continuation_casing("I went home."),
            "I went home."
        );
    }

    // Continuation starting with an acronym — should stay uppercase
    // Simulates: field has "we need to fix the" → user dictates:
    #[test]
    fn test_continuation_starting_with_acronym() {
        assert_eq!(
            fix_continuation_casing("API before the release."),
            "API before the release."
        );
    }

    // Continuation with number — nothing to lowercase
    #[test]
    fn test_continuation_starting_with_number() {
        assert_eq!(
            fix_continuation_casing("3:30 PM on Tuesday."),
            "3:30 PM on Tuesday."
        );
    }

    // Continuation with punctuation-only output
    #[test]
    fn test_continuation_punctuation_only() {
        assert_eq!(fix_continuation_casing("..."), "...");
        assert_eq!(fix_continuation_casing("?"), "?");
    }

    // Multiple sentences in continuation — only first word lowercased
    #[test]
    fn test_continuation_multi_sentence() {
        assert_eq!(
            fix_continuation_casing("Went to the store. Then came home."),
            "went to the store. Then came home."
        );
    }
}
