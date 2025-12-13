//! Configuration management for VoiceFlow

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::env;

/// Configuration validation error
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid temperature: {value}. Must be between 0.0 and 2.0")]
    InvalidTemperature { value: f32 },

    #[error("Invalid top_p: {value}. Must be between 0.0 and 1.0")]
    InvalidTopP { value: f32 },

    #[error("Invalid max_tokens: {value}. Must be between 1 and 8192")]
    InvalidMaxTokens { value: u32 },

    #[error("Invalid sample_rate: {value}. Must be one of: 8000, 16000, 22050, 44100, 48000")]
    InvalidSampleRate { value: u32 },

    #[error("Invalid VAD threshold: {value}. Must be between 0.0 and 1.0")]
    InvalidVadThreshold { value: f32 },

    #[error("Invalid silence duration: {value}ms. Must be between 100 and 5000")]
    InvalidSilenceDuration { value: u32 },

    #[error("Unknown context: {context}. Valid contexts: default, email, slack, code")]
    InvalidContext { context: String },
}

/// Speech-to-Text engine selection
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SttEngine {
    #[default]
    Whisper,
    Moonshine,
}

impl SttEngine {
    pub fn display_name(&self) -> &str {
        match self {
            Self::Whisper => "Whisper",
            Self::Moonshine => "Moonshine",
        }
    }
}

/// Moonshine model sizes
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MoonshineModel {
    #[default]
    Tiny,
    Base,
}

impl MoonshineModel {
    /// Get the model directory name (contains preprocessor, encoder, decoder)
    pub fn dir_name(&self) -> &str {
        match self {
            Self::Tiny => "moonshine-tiny",
            Self::Base => "moonshine-base",
        }
    }

    /// Get HuggingFace repo for downloading
    pub fn hf_repo(&self) -> &str {
        "UsefulSensors/moonshine"
    }

    /// Get the ONNX files path prefix within the repo
    pub fn onnx_path(&self) -> &str {
        match self {
            Self::Tiny => "onnx/tiny",
            Self::Base => "onnx/base",
        }
    }

    /// Get display name
    pub fn display_name(&self) -> &str {
        match self {
            Self::Tiny => "Moonshine Tiny (27M)",
            Self::Base => "Moonshine Base (62M)",
        }
    }

    /// Get estimated model size in MB
    pub fn size_mb(&self) -> u32 {
        match self {
            Self::Tiny => 190,
            Self::Base => 400,
        }
    }

    /// Get all available Moonshine models
    pub fn all_models() -> Vec<MoonshineModel> {
        vec![Self::Tiny, Self::Base]
    }

    /// Required ONNX files for this model
    pub fn required_files(&self) -> Vec<&'static str> {
        vec!["preprocess.onnx", "encode.onnx", "uncached_decode.onnx", "cached_decode.onnx"]
    }
}

/// Supported LLM models (non-Meta, permissive licenses)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum LlmModel {
    /// Qwen3 1.7B - Fast, multilingual (Apache 2.0)
    Qwen3_1_7B,
    /// Qwen3 4B - Higher quality, multilingual (Apache 2.0)
    Qwen3_4B,
    /// SmolLM3 3B - Hugging Face efficient model (Apache 2.0)
    SmolLM3_3B,
    /// Gemma 2 2B - Google's compact model
    Gemma2_2B,
    /// Phi-2 - Microsoft's 2.7B model (test)
    Phi2,
    /// Custom model path
    Custom(String),
}

impl Default for LlmModel {
    fn default() -> Self {
        Self::Qwen3_1_7B
    }
}

impl LlmModel {
    /// Get the model filename for downloading (exact HuggingFace filenames)
    pub fn filename(&self) -> &str {
        match self {
            // Keep original filename for backwards compatibility with existing downloads
            Self::Qwen3_1_7B => "qwen3-1.7b-q4_k_m.gguf",
            Self::Qwen3_4B => "Qwen3-4B-Q4_K_M.gguf",
            Self::SmolLM3_3B => "SmolLM3-Q4_K_M.gguf",
            Self::Gemma2_2B => "gemma-2-2b-it-Q4_K_M.gguf",
            Self::Phi2 => "phi-2-q4.gguf",
            Self::Custom(path) => path,
        }
    }

    /// Get the Hugging Face repo for this model
    pub fn hf_repo(&self) -> Option<&str> {
        match self {
            Self::Qwen3_1_7B => Some("Qwen/Qwen3-1.7B-GGUF"),
            Self::Qwen3_4B => Some("Qwen/Qwen3-4B-GGUF"),
            Self::SmolLM3_3B => Some("ggml-org/SmolLM3-3B-GGUF"),
            Self::Gemma2_2B => Some("bartowski/gemma-2-2b-it-GGUF"),
            Self::Phi2 => Some("TheBloke/phi-2-GGUF"),
            Self::Custom(_) => None,
        }
    }

    /// Get display name
    pub fn display_name(&self) -> &str {
        match self {
            Self::Qwen3_1_7B => "Qwen3 1.7B",
            Self::Qwen3_4B => "Qwen3 4B",
            Self::SmolLM3_3B => "SmolLM3 3B",
            Self::Gemma2_2B => "Gemma 2 2B",
            Self::Phi2 => "Phi-2",
            Self::Custom(path) => path,
        }
    }

    /// Get estimated model size in GB (Q4_K_M quantization, from HuggingFace)
    pub fn size_gb(&self) -> f32 {
        match self {
            Self::Qwen3_1_7B => 1.28,
            Self::Qwen3_4B => 2.5,
            Self::SmolLM3_3B => 1.92,
            Self::Gemma2_2B => 1.71,
            Self::Phi2 => 1.6,
            Self::Custom(_) => 0.0,
        }
    }

    /// Get all available models (excluding Custom)
    pub fn all_models() -> Vec<LlmModel> {
        vec![
            Self::Qwen3_1_7B,
            Self::Qwen3_4B,
            Self::SmolLM3_3B,
            Self::Gemma2_2B,
        ]
    }
}

/// Supported Whisper model sizes
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WhisperModel {
    Tiny,
    #[default]
    Base,
    Small,
    Medium,
}

impl WhisperModel {
    pub fn filename(&self) -> &str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::Base => "ggml-base.bin",
            Self::Small => "ggml-small.bin",
            Self::Medium => "ggml-medium.bin",
        }
    }

    pub fn url(&self) -> &str {
        match self {
            Self::Tiny => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            Self::Base => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            Self::Small => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            Self::Medium => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        }
    }
}

/// LLM generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmOptions {
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature (0.0 = deterministic, 1.0 = creative)
    pub temperature: f32,
    /// Top-p nucleus sampling
    pub top_p: f32,
    /// Number of GPU layers to offload (-1 = all)
    pub n_gpu_layers: i32,
    /// Disable thinking/reasoning mode for faster inference
    pub enable_thinking: bool,
}

impl Default for LlmOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.3,
            top_p: 0.9,
            n_gpu_layers: -1, // All layers on GPU (mistral.rs handles this automatically)
            enable_thinking: false, // Fast inference, no chain-of-thought
        }
    }
}

/// Audio capture settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOptions {
    /// Input sample rate (will be resampled to 16kHz for Whisper)
    pub sample_rate: u32,
    /// Voice activity detection threshold
    pub vad_threshold: f32,
    /// Silence duration (ms) to trigger end of speech
    pub silence_duration_ms: u32,
}

impl Default for AudioOptions {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            vad_threshold: 0.01,
            silence_duration_ms: 800,
        }
    }
}

/// Main configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// STT engine selection (whisper or moonshine)
    #[serde(default)]
    pub stt_engine: SttEngine,
    /// Whisper model size (used when stt_engine is Whisper)
    pub whisper_model: WhisperModel,
    /// Moonshine model size (used when stt_engine is Moonshine)
    #[serde(default)]
    pub moonshine_model: MoonshineModel,
    /// LLM model selection
    pub llm_model: LlmModel,
    /// LLM generation options
    pub llm_options: LlmOptions,
    /// Audio capture options
    pub audio: AudioOptions,
    /// Default context when none specified
    pub default_context: String,
    /// Personal dictionary words
    pub personal_dictionary: Vec<String>,
    /// Auto-copy to clipboard
    pub auto_clipboard: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            stt_engine: SttEngine::default(),
            whisper_model: WhisperModel::default(),
            moonshine_model: MoonshineModel::default(),
            llm_model: LlmModel::default(),
            llm_options: LlmOptions::default(),
            audio: AudioOptions::default(),
            default_context: "default".to_string(),
            personal_dictionary: vec![],
            auto_clipboard: true,
        }
    }
}

/// Environment variable names for configuration overrides
pub mod env_vars {
    pub const STT_ENGINE: &str = "VOICEFLOW_STT_ENGINE";
    pub const WHISPER_MODEL: &str = "VOICEFLOW_WHISPER_MODEL";
    pub const MOONSHINE_MODEL: &str = "VOICEFLOW_MOONSHINE_MODEL";
    pub const LLM_MODEL: &str = "VOICEFLOW_LLM_MODEL";
    pub const LLM_TEMPERATURE: &str = "VOICEFLOW_LLM_TEMPERATURE";
    pub const LLM_MAX_TOKENS: &str = "VOICEFLOW_LLM_MAX_TOKENS";
    pub const LLM_TOP_P: &str = "VOICEFLOW_LLM_TOP_P";
    pub const ENABLE_THINKING: &str = "VOICEFLOW_ENABLE_THINKING";
    pub const DEFAULT_CONTEXT: &str = "VOICEFLOW_DEFAULT_CONTEXT";
    pub const MODELS_DIR: &str = "VOICEFLOW_MODELS_DIR";
}

impl Config {
    /// Load configuration from file or use defaults
    pub fn load(path: Option<&str>) -> Result<Self> {
        let config_path = match path {
            Some(p) => PathBuf::from(p),
            None => Self::default_config_path()?,
        };

        let mut config = if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config from {:?}", config_path))?;
            toml::from_str(&contents)
                .with_context(|| format!("Failed to parse config from {:?}", config_path))?
        } else {
            Self::default()
        };

        // Apply environment variable overrides
        config.apply_env_overrides();

        Ok(config)
    }

    /// Load configuration with validation
    pub fn load_validated(path: Option<&str>) -> Result<Self> {
        let config = Self::load(path)?;
        config.validate()?;
        Ok(config)
    }

    /// Apply environment variable overrides to the configuration
    pub fn apply_env_overrides(&mut self) {
        // STT engine
        if let Ok(val) = env::var(env_vars::STT_ENGINE) {
            match val.to_lowercase().as_str() {
                "whisper" => self.stt_engine = SttEngine::Whisper,
                "moonshine" => self.stt_engine = SttEngine::Moonshine,
                _ => tracing::warn!("Unknown STT engine from env: {}", val),
            }
        }

        // Whisper model
        if let Ok(val) = env::var(env_vars::WHISPER_MODEL) {
            match val.to_lowercase().as_str() {
                "tiny" => self.whisper_model = WhisperModel::Tiny,
                "base" => self.whisper_model = WhisperModel::Base,
                "small" => self.whisper_model = WhisperModel::Small,
                "medium" => self.whisper_model = WhisperModel::Medium,
                _ => tracing::warn!("Unknown Whisper model from env: {}", val),
            }
        }

        // Moonshine model
        if let Ok(val) = env::var(env_vars::MOONSHINE_MODEL) {
            match val.to_lowercase().as_str() {
                "tiny" => self.moonshine_model = MoonshineModel::Tiny,
                "base" => self.moonshine_model = MoonshineModel::Base,
                _ => tracing::warn!("Unknown Moonshine model from env: {}", val),
            }
        }

        // LLM temperature
        if let Ok(val) = env::var(env_vars::LLM_TEMPERATURE) {
            if let Ok(temp) = val.parse::<f32>() {
                self.llm_options.temperature = temp;
            } else {
                tracing::warn!("Invalid LLM temperature from env: {}", val);
            }
        }

        // LLM max tokens
        if let Ok(val) = env::var(env_vars::LLM_MAX_TOKENS) {
            if let Ok(tokens) = val.parse::<u32>() {
                self.llm_options.max_tokens = tokens;
            } else {
                tracing::warn!("Invalid LLM max_tokens from env: {}", val);
            }
        }

        // LLM top_p
        if let Ok(val) = env::var(env_vars::LLM_TOP_P) {
            if let Ok(top_p) = val.parse::<f32>() {
                self.llm_options.top_p = top_p;
            } else {
                tracing::warn!("Invalid LLM top_p from env: {}", val);
            }
        }

        // Enable thinking
        if let Ok(val) = env::var(env_vars::ENABLE_THINKING) {
            self.llm_options.enable_thinking = val.to_lowercase() == "true" || val == "1";
        }

        // Default context
        if let Ok(val) = env::var(env_vars::DEFAULT_CONTEXT) {
            self.default_context = val;
        }
    }

    /// Validate the configuration values
    pub fn validate(&self) -> Result<()> {
        // Validate LLM options
        if self.llm_options.temperature < 0.0 || self.llm_options.temperature > 2.0 {
            return Err(ConfigError::InvalidTemperature {
                value: self.llm_options.temperature,
            }.into());
        }

        if self.llm_options.top_p < 0.0 || self.llm_options.top_p > 1.0 {
            return Err(ConfigError::InvalidTopP {
                value: self.llm_options.top_p,
            }.into());
        }

        if self.llm_options.max_tokens == 0 || self.llm_options.max_tokens > 8192 {
            return Err(ConfigError::InvalidMaxTokens {
                value: self.llm_options.max_tokens,
            }.into());
        }

        // Validate audio options
        let valid_sample_rates = [8000, 16000, 22050, 44100, 48000];
        if !valid_sample_rates.contains(&self.audio.sample_rate) {
            return Err(ConfigError::InvalidSampleRate {
                value: self.audio.sample_rate,
            }.into());
        }

        if self.audio.vad_threshold < 0.0 || self.audio.vad_threshold > 1.0 {
            return Err(ConfigError::InvalidVadThreshold {
                value: self.audio.vad_threshold,
            }.into());
        }

        if self.audio.silence_duration_ms < 100 || self.audio.silence_duration_ms > 5000 {
            return Err(ConfigError::InvalidSilenceDuration {
                value: self.audio.silence_duration_ms,
            }.into());
        }

        // Validate context
        let valid_contexts = ["default", "email", "slack", "code"];
        if !valid_contexts.contains(&self.default_context.as_str()) {
            return Err(ConfigError::InvalidContext {
                context: self.default_context.clone(),
            }.into());
        }

        Ok(())
    }

    /// Save configuration to file
    pub fn save(&self, path: Option<&str>) -> Result<()> {
        let config_path = match path {
            Some(p) => PathBuf::from(p),
            None => Self::default_config_path()?,
        };

        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;
        Ok(())
    }

    /// Get the default config file path
    pub fn default_config_path() -> Result<PathBuf> {
        let proj_dirs = ProjectDirs::from("com", "era-laboratories", "voiceflow")
            .context("Could not determine config directory")?;
        Ok(proj_dirs.config_dir().join("config.toml"))
    }

    /// Get the models directory
    pub fn models_dir() -> Result<PathBuf> {
        let proj_dirs = ProjectDirs::from("com", "era-laboratories", "voiceflow")
            .context("Could not determine data directory")?;
        let models_dir = proj_dirs.data_dir().join("models");
        std::fs::create_dir_all(&models_dir)?;
        Ok(models_dir)
    }

    /// Get the prompts directory
    pub fn prompts_dir() -> Result<PathBuf> {
        let proj_dirs = ProjectDirs::from("com", "era-laboratories", "voiceflow")
            .context("Could not determine data directory")?;
        let prompts_dir = proj_dirs.data_dir().join("prompts");
        std::fs::create_dir_all(&prompts_dir)?;
        Ok(prompts_dir)
    }

    /// Get full path to Whisper model
    pub fn whisper_model_path(&self) -> Result<PathBuf> {
        Ok(Self::models_dir()?.join(self.whisper_model.filename()))
    }

    /// Get full path to LLM model
    pub fn llm_model_path(&self) -> Result<PathBuf> {
        Ok(Self::models_dir()?.join(self.llm_model.filename()))
    }

    /// Get directory containing Moonshine ONNX models
    pub fn moonshine_model_dir(&self) -> Result<PathBuf> {
        Ok(Self::models_dir()?.join(self.moonshine_model.dir_name()))
    }

    /// Check if configured Moonshine model files are downloaded
    pub fn moonshine_model_downloaded(&self) -> bool {
        self.moonshine_model_downloaded_for(&self.moonshine_model)
    }

    /// Check if specified Moonshine model files are downloaded
    pub fn moonshine_model_downloaded_for(&self, model: &MoonshineModel) -> bool {
        if let Ok(models_dir) = Self::models_dir() {
            let model_dir = models_dir.join(model.dir_name());
            model
                .required_files()
                .iter()
                .all(|f| model_dir.join(f).exists())
        } else {
            false
        }
    }

    /// Get prompt template for a given context
    pub fn get_prompt_for_context(&self, context: Option<&str>) -> String {
        let ctx = context.unwrap_or(&self.default_context);

        // Try to load from prompts directory
        if let Ok(prompts_dir) = Self::prompts_dir() {
            let prompt_file = prompts_dir.join(format!("{}.txt", ctx));
            if let Ok(contents) = std::fs::read_to_string(&prompt_file) {
                return contents;
            }
        }

        // Fallback to built-in prompts
        match ctx {
            "email" => include_str!("../../../prompts/email.txt").to_string(),
            "slack" => include_str!("../../../prompts/slack.txt").to_string(),
            "code" => include_str!("../../../prompts/code.txt").to_string(),
            _ => include_str!("../../../prompts/default.txt").to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_temperature() {
        let mut config = Config::default();
        config.llm_options.temperature = 3.0;
        assert!(config.validate().is_err());

        config.llm_options.temperature = -0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_top_p() {
        let mut config = Config::default();
        config.llm_options.top_p = 1.5;
        assert!(config.validate().is_err());

        config.llm_options.top_p = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_max_tokens() {
        let mut config = Config::default();
        config.llm_options.max_tokens = 0;
        assert!(config.validate().is_err());

        config.llm_options.max_tokens = 10000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_sample_rate() {
        let mut config = Config::default();
        config.audio.sample_rate = 12345;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_valid_sample_rates() {
        let mut config = Config::default();
        for rate in [8000, 16000, 22050, 44100, 48000] {
            config.audio.sample_rate = rate;
            assert!(config.validate().is_ok(), "Sample rate {} should be valid", rate);
        }
    }

    #[test]
    fn test_invalid_vad_threshold() {
        let mut config = Config::default();
        config.audio.vad_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_silence_duration() {
        let mut config = Config::default();
        config.audio.silence_duration_ms = 50;
        assert!(config.validate().is_err());

        config.audio.silence_duration_ms = 10000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_context() {
        let mut config = Config::default();
        config.default_context = "invalid_context".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_valid_contexts() {
        let mut config = Config::default();
        for ctx in ["default", "email", "slack", "code"] {
            config.default_context = ctx.to_string();
            assert!(config.validate().is_ok(), "Context '{}' should be valid", ctx);
        }
    }

    #[test]
    fn test_env_var_names() {
        // Ensure all env var names are unique and properly prefixed
        let vars = [
            env_vars::STT_ENGINE,
            env_vars::WHISPER_MODEL,
            env_vars::MOONSHINE_MODEL,
            env_vars::LLM_MODEL,
            env_vars::LLM_TEMPERATURE,
            env_vars::LLM_MAX_TOKENS,
            env_vars::LLM_TOP_P,
            env_vars::ENABLE_THINKING,
            env_vars::DEFAULT_CONTEXT,
            env_vars::MODELS_DIR,
        ];

        for var in &vars {
            assert!(var.starts_with("VOICEFLOW_"), "Env var {} should be prefixed", var);
        }

        // Check uniqueness
        let mut unique: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for var in &vars {
            assert!(unique.insert(var), "Duplicate env var: {}", var);
        }
    }
}
