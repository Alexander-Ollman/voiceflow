//! Configuration management for VoiceFlow

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Supported LLM models (non-Meta, permissive licenses)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum LlmModel {
    /// Qwen3 1.7B - Fast, multilingual (Apache 2.0)
    Qwen3_1_7B,
    /// SmolLM3 3B - Hugging Face efficient model (Apache 2.0)
    SmolLM3_3B,
    /// Gemma 2 2B - Google's compact model
    Gemma2_2B,
    /// Custom model path
    Custom(String),
}

impl Default for LlmModel {
    fn default() -> Self {
        Self::Qwen3_1_7B
    }
}

impl LlmModel {
    /// Get the model filename for downloading
    pub fn filename(&self) -> &str {
        match self {
            Self::Qwen3_1_7B => "qwen3-1.7b-q4_k_m.gguf",
            Self::SmolLM3_3B => "smollm3-3b-q4_k_m.gguf",
            Self::Gemma2_2B => "gemma-2-2b-q4_k_m.gguf",
            Self::Custom(path) => path,
        }
    }

    /// Get the Hugging Face repo for this model
    pub fn hf_repo(&self) -> Option<&str> {
        match self {
            Self::Qwen3_1_7B => Some("Qwen/Qwen3-1.7B-GGUF"),
            Self::SmolLM3_3B => Some("ggml-org/SmolLM3-3B-GGUF"),
            Self::Gemma2_2B => Some("google/gemma-2-2b-it-GGUF"),
            Self::Custom(_) => None,
        }
    }

    /// Get display name
    pub fn display_name(&self) -> &str {
        match self {
            Self::Qwen3_1_7B => "Qwen3 1.7B",
            Self::SmolLM3_3B => "SmolLM3 3B",
            Self::Gemma2_2B => "Gemma 2 2B",
            Self::Custom(path) => path,
        }
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
            n_gpu_layers: -1,
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
    /// Whisper model size
    pub whisper_model: WhisperModel,
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
            whisper_model: WhisperModel::default(),
            llm_model: LlmModel::default(),
            llm_options: LlmOptions::default(),
            audio: AudioOptions::default(),
            default_context: "default".to_string(),
            personal_dictionary: vec![],
            auto_clipboard: true,
        }
    }
}

impl Config {
    /// Load configuration from file or use defaults
    pub fn load(path: Option<&str>) -> Result<Self> {
        let config_path = match path {
            Some(p) => PathBuf::from(p),
            None => Self::default_config_path()?,
        };

        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config from {:?}", config_path))?;
            toml::from_str(&contents)
                .with_context(|| format!("Failed to parse config from {:?}", config_path))
        } else {
            Ok(Self::default())
        }
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
