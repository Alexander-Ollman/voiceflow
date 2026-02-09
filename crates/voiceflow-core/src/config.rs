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
#[serde(rename_all = "kebab-case")]
pub enum SttEngine {
    Whisper,
    #[default]
    Moonshine,
    /// Qwen3-ASR: external Python daemon handles transcription.
    /// In traditional mode, output is piped through the LLM for formatting.
    Qwen3Asr,
}

impl SttEngine {
    pub fn display_name(&self) -> &str {
        match self {
            Self::Whisper => "Whisper",
            Self::Moonshine => "Moonshine",
            Self::Qwen3Asr => "Qwen3-ASR",
        }
    }

    /// Whether this engine runs externally (not in the Rust pipeline).
    pub fn is_external(&self) -> bool {
        matches!(self, Self::Qwen3Asr)
    }
}

/// Moonshine model sizes
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MoonshineModel {
    Tiny,
    #[default]
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

/// Pipeline mode selection
/// Determines whether the pipeline uses separate STT + LLM stages,
/// or a single consolidated model that handles both.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum PipelineMode {
    /// Traditional pipeline: STT (Whisper/Moonshine) → Prosody → LLM
    #[default]
    SttPlusLlm,
    /// Consolidated: a single model handles audio-to-formatted-text (e.g., Qwen3-ASR via MLX Swift)
    Consolidated,
}

impl PipelineMode {
    pub fn display_name(&self) -> &str {
        match self {
            Self::SttPlusLlm => "STT + LLM (traditional)",
            Self::Consolidated => "Consolidated (single model)",
        }
    }
}

/// Consolidated model options — replaces both STT and LLM in a single model
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ConsolidatedModel {
    /// Qwen3-ASR 0.6B — lightweight, fast consolidated model
    #[default]
    Qwen3Asr0_6B,
    /// Qwen3-ASR 1.7B — higher quality consolidated model
    Qwen3Asr1_7B,
}

impl ConsolidatedModel {
    pub fn display_name(&self) -> &str {
        match self {
            Self::Qwen3Asr0_6B => "Qwen3-ASR 0.6B",
            Self::Qwen3Asr1_7B => "Qwen3-ASR 1.7B",
        }
    }

    pub fn dir_name(&self) -> &str {
        match self {
            Self::Qwen3Asr0_6B => "qwen3-asr-0.6b",
            Self::Qwen3Asr1_7B => "qwen3-asr-1.7b",
        }
    }

    pub fn hf_repo(&self) -> &str {
        match self {
            Self::Qwen3Asr0_6B => "Qwen/Qwen3-ASR-0.6B",
            Self::Qwen3Asr1_7B => "Qwen/Qwen3-ASR-1.7B",
        }
    }

    pub fn size_gb(&self) -> f32 {
        match self {
            Self::Qwen3Asr0_6B => 1.2,
            Self::Qwen3Asr1_7B => 3.4,
        }
    }

    /// Required files for PyTorch/qwen-asr inference
    pub fn required_files(&self) -> Vec<&'static str> {
        match self {
            Self::Qwen3Asr0_6B => vec![
                "config.json",
                "chat_template.json",
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "generation_config.json",
                "model.safetensors",
            ],
            Self::Qwen3Asr1_7B => vec![
                "config.json",
                "chat_template.json",
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        }
    }

    pub fn all_models() -> Vec<ConsolidatedModel> {
        vec![Self::Qwen3Asr0_6B, Self::Qwen3Asr1_7B]
    }
}

/// Vision-Language Model options — multimodal models for image + text understanding
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum VlmModel {
    /// Jina VLM — jinaai's vision-language model (~9.9 GB)
    #[default]
    JinaVlm,
    /// Qwen3-VL 2B Instruct — Qwen's compact vision-language model (~4.3 GB)
    Qwen3VL2B,
}

impl VlmModel {
    pub fn display_name(&self) -> &str {
        match self {
            Self::JinaVlm => "Jina VLM",
            Self::Qwen3VL2B => "Qwen3-VL 2B Instruct",
        }
    }

    pub fn dir_name(&self) -> &str {
        match self {
            Self::JinaVlm => "jina-vlm",
            Self::Qwen3VL2B => "qwen3-vl-2b-instruct",
        }
    }

    pub fn hf_repo(&self) -> &str {
        match self {
            Self::JinaVlm => "jinaai/jina-vlm",
            Self::Qwen3VL2B => "Qwen/Qwen3-VL-2B-Instruct",
        }
    }

    pub fn size_gb(&self) -> f32 {
        match self {
            Self::JinaVlm => 9.9,
            Self::Qwen3VL2B => 4.3,
        }
    }

    /// Required files for downloading and verifying the model
    pub fn required_files(&self) -> Vec<&'static str> {
        match self {
            Self::JinaVlm => vec![
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template.jinja",
                "model.safetensors.index.json",
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
                // Custom model code (needed for trust_remote_code)
                "modeling_jvlm.py",
                "blocks_jvlm.py",
                "configuration_jvlm.py",
                "image_processing_jvlm.py",
                "processing_jvlm.py",
            ],
            Self::Qwen3VL2B => vec![
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "chat_template.json",
                "model.safetensors",
            ],
        }
    }

    pub fn all_models() -> Vec<VlmModel> {
        vec![Self::JinaVlm, Self::Qwen3VL2B]
    }
}

/// LLM inference backend selection
/// Different backends support different model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum LlmBackend {
    /// mistral.rs - Good for Qwen, Gemma2, Phi-2
    /// Some newer architectures (SmolLM3, Gemma3n, Phi-4) not yet supported
    #[default]
    MistralRs,
    /// llama.cpp via llama-cpp-2 crate - Supports all GGUF architectures
    /// Recommended for SmolLM3, Gemma3n, Phi-4, and other newer models
    LlamaCpp,
}

impl LlmBackend {
    pub fn display_name(&self) -> &str {
        match self {
            Self::MistralRs => "mistral.rs",
            Self::LlamaCpp => "llama.cpp",
        }
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
    /// Gemma 3n E2B - Google's multimodal model, 2B effective params (Gemma license)
    Gemma3nE2B,
    /// Gemma 3n E4B - Google's multimodal model, 4B effective params (Gemma license)
    Gemma3nE4B,
    /// Phi-4 Mini - Microsoft's latest small model (MIT)
    Phi4Mini,
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
            Self::Gemma3nE2B => "gemma-3n-E2B-it-Q4_K_M.gguf",
            Self::Gemma3nE4B => "gemma-3n-E4B-it-Q4_K_M.gguf",
            Self::Phi4Mini => "Phi-4-mini-instruct-Q4_K_M.gguf",
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
            Self::Gemma3nE2B => Some("unsloth/gemma-3n-E2B-it-GGUF"),
            Self::Gemma3nE4B => Some("unsloth/gemma-3n-E4B-it-GGUF"),
            Self::Phi4Mini => Some("lmstudio-community/Phi-4-mini-instruct-GGUF"),
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
            Self::Gemma3nE2B => "Gemma 3n E2B (Multimodal)",
            Self::Gemma3nE4B => "Gemma 3n E4B (Multimodal)",
            Self::Phi4Mini => "Phi-4 Mini 3.8B",
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
            Self::Gemma3nE2B => 1.8,  // E2B = 2B effective params
            Self::Gemma3nE4B => 3.2,  // E4B = 4B effective params
            Self::Phi4Mini => 2.4,
            Self::Phi2 => 1.6,
            Self::Custom(_) => 0.0,
        }
    }

    /// Whether this model supports multimodal input (images/screenshots)
    pub fn is_multimodal(&self) -> bool {
        matches!(self, Self::Gemma3nE2B | Self::Gemma3nE4B)
    }

    /// Get the recommended inference backend for this model
    /// Models with architectures not supported by mistral.rs use llama.cpp
    pub fn backend(&self) -> LlmBackend {
        match self {
            // Supported by mistral.rs
            Self::Qwen3_1_7B => LlmBackend::MistralRs,
            Self::Qwen3_4B => LlmBackend::MistralRs,
            Self::Gemma2_2B => LlmBackend::MistralRs,
            Self::Phi2 => LlmBackend::MistralRs,

            // Require llama.cpp (architecture not supported by mistral.rs)
            Self::SmolLM3_3B => LlmBackend::LlamaCpp,   // 'smollm3' architecture
            Self::Gemma3nE2B => LlmBackend::LlamaCpp,   // 'gemma3n' architecture
            Self::Gemma3nE4B => LlmBackend::LlamaCpp,   // 'gemma3n' architecture
            Self::Phi4Mini => LlmBackend::LlamaCpp,     // 'phi4' architecture

            // Custom models default to llama.cpp (more architecture coverage)
            Self::Custom(_) => LlmBackend::LlamaCpp,
        }
    }

    /// Get all available models (excluding Custom)
    pub fn all_models() -> Vec<LlmModel> {
        vec![
            Self::Qwen3_1_7B,
            Self::Qwen3_4B,
            Self::SmolLM3_3B,
            Self::Gemma2_2B,
            Self::Gemma3nE2B,
            Self::Gemma3nE4B,
            Self::Phi4Mini,
        ]
    }

    /// Get benchmark-recommended models for dictation formatting
    /// Now includes all models thanks to switchable backends (mistral.rs + llama.cpp)
    pub fn benchmark_models() -> Vec<LlmModel> {
        vec![
            Self::Qwen3_1_7B,   // Fast baseline (mistral.rs)
            Self::Qwen3_4B,     // Higher quality (mistral.rs)
            Self::SmolLM3_3B,   // HuggingFace efficient model (llama.cpp)
            Self::Gemma3nE2B,   // Google's multimodal (llama.cpp)
            Self::Phi4Mini,     // Microsoft's latest (llama.cpp)
        ]
    }
}

/// Supported Whisper model sizes
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum WhisperModel {
    Tiny,
    #[default]
    Base,
    Small,
    Medium,
    /// Large V3 - highest accuracy, slower
    LargeV3,
    /// Large V3 Turbo - 6x faster than V3, near-same accuracy
    LargeV3Turbo,
    /// Distil Large V3 - distilled for speed, ~3.5% WER
    DistilLargeV3,
}

impl WhisperModel {
    pub fn filename(&self) -> &str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::Base => "ggml-base.bin",
            Self::Small => "ggml-small.bin",
            Self::Medium => "ggml-medium.bin",
            Self::LargeV3 => "ggml-large-v3.bin",
            Self::LargeV3Turbo => "ggml-large-v3-turbo.bin",
            Self::DistilLargeV3 => "ggml-distil-large-v3.bin",
        }
    }

    pub fn url(&self) -> &str {
        match self {
            Self::Tiny => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            Self::Base => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            Self::Small => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            Self::Medium => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            Self::LargeV3 => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
            Self::LargeV3Turbo => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
            Self::DistilLargeV3 => "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Tiny => "Whisper Tiny (39M)",
            Self::Base => "Whisper Base (74M)",
            Self::Small => "Whisper Small (244M)",
            Self::Medium => "Whisper Medium (769M)",
            Self::LargeV3 => "Whisper Large V3 (1.5B)",
            Self::LargeV3Turbo => "Whisper Large V3 Turbo (809M)",
            Self::DistilLargeV3 => "Distil-Whisper Large V3 (756M)",
        }
    }

    pub fn size_mb(&self) -> u32 {
        match self {
            Self::Tiny => 75,
            Self::Base => 142,
            Self::Small => 466,
            Self::Medium => 1533,
            Self::LargeV3 => 3094,
            Self::LargeV3Turbo => 1624,
            Self::DistilLargeV3 => 1510,
        }
    }

    pub fn expected_wer(&self) -> f32 {
        match self {
            Self::Tiny => 7.5,
            Self::Base => 5.0,
            Self::Small => 4.2,
            Self::Medium => 3.5,
            Self::LargeV3 => 2.9,
            Self::LargeV3Turbo => 3.0,
            Self::DistilLargeV3 => 3.5,
        }
    }

    /// Get all available Whisper models
    pub fn all_models() -> Vec<WhisperModel> {
        vec![
            Self::Tiny,
            Self::Base,
            Self::Small,
            Self::Medium,
            Self::LargeV3,
            Self::LargeV3Turbo,
            Self::DistilLargeV3,
        ]
    }

    /// Get benchmark-recommended models (good coverage of size/quality tradeoffs)
    pub fn benchmark_models() -> Vec<WhisperModel> {
        vec![
            Self::Base,           // Current default
            Self::LargeV3Turbo,   // Best speed/accuracy
            Self::DistilLargeV3,  // Distilled alternative
        ]
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
    /// Pipeline mode: traditional STT+LLM or consolidated single-model
    #[serde(default)]
    pub pipeline_mode: PipelineMode,
    /// Consolidated model selection (used when pipeline_mode is Consolidated)
    #[serde(default)]
    pub consolidated_model: ConsolidatedModel,
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
    /// VLM model selection (None = no VLM, use LLM only)
    #[serde(default)]
    pub vlm_model: Option<VlmModel>,
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
            pipeline_mode: PipelineMode::default(),
            consolidated_model: ConsolidatedModel::default(),
            stt_engine: SttEngine::default(),
            whisper_model: WhisperModel::default(),
            moonshine_model: MoonshineModel::default(),
            llm_model: LlmModel::default(),
            vlm_model: None,
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
    pub const PIPELINE_MODE: &str = "VOICEFLOW_PIPELINE_MODE";
    pub const CONSOLIDATED_MODEL: &str = "VOICEFLOW_CONSOLIDATED_MODEL";
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
        // Pipeline mode
        if let Ok(val) = env::var(env_vars::PIPELINE_MODE) {
            match val.to_lowercase().replace("-", "_").as_str() {
                "stt_plus_llm" | "traditional" | "default" => self.pipeline_mode = PipelineMode::SttPlusLlm,
                "consolidated" => self.pipeline_mode = PipelineMode::Consolidated,
                _ => tracing::warn!("Unknown pipeline mode from env: {}", val),
            }
        }

        // Consolidated model
        if let Ok(val) = env::var(env_vars::CONSOLIDATED_MODEL) {
            match val.to_lowercase().replace(['-', '.'], "_").as_str() {
                "qwen3_asr_0_6b" | "qwen3_asr_0.6b" | "0.6b" | "0_6b" => {
                    self.consolidated_model = ConsolidatedModel::Qwen3Asr0_6B;
                }
                "qwen3_asr_1_7b" | "qwen3_asr_1.7b" | "1.7b" | "1_7b" => {
                    self.consolidated_model = ConsolidatedModel::Qwen3Asr1_7B;
                }
                _ => tracing::warn!("Unknown consolidated model from env: {}", val),
            }
        }

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
            match val.to_lowercase().replace("-", "_").as_str() {
                "tiny" => self.whisper_model = WhisperModel::Tiny,
                "base" => self.whisper_model = WhisperModel::Base,
                "small" => self.whisper_model = WhisperModel::Small,
                "medium" => self.whisper_model = WhisperModel::Medium,
                "large_v3" | "largev3" => self.whisper_model = WhisperModel::LargeV3,
                "large_v3_turbo" | "largev3turbo" | "turbo" => self.whisper_model = WhisperModel::LargeV3Turbo,
                "distil_large_v3" | "distillargev3" | "distil" => self.whisper_model = WhisperModel::DistilLargeV3,
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

    /// Get directory for a consolidated model
    pub fn consolidated_model_dir(&self) -> Result<PathBuf> {
        Ok(Self::models_dir()?.join(self.consolidated_model.dir_name()))
    }

    /// Check if the configured consolidated model files are downloaded
    pub fn consolidated_model_downloaded(&self) -> bool {
        self.consolidated_model_downloaded_for(&self.consolidated_model)
    }

    /// Check if a specific consolidated model's files are downloaded
    pub fn consolidated_model_downloaded_for(&self, model: &ConsolidatedModel) -> bool {
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

    /// Check if a specific VLM model's files are downloaded
    pub fn vlm_model_downloaded_for(&self, model: &VlmModel) -> bool {
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

    /// Check if the pipeline is in consolidated mode
    pub fn is_consolidated_mode(&self) -> bool {
        self.pipeline_mode == PipelineMode::Consolidated
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
    fn test_consolidated_model_properties() {
        let model_0_6b = ConsolidatedModel::Qwen3Asr0_6B;
        assert_eq!(model_0_6b.display_name(), "Qwen3-ASR 0.6B");
        assert_eq!(model_0_6b.dir_name(), "qwen3-asr-0.6b");
        assert!(!model_0_6b.required_files().is_empty());

        let model_1_7b = ConsolidatedModel::Qwen3Asr1_7B;
        assert_eq!(model_1_7b.display_name(), "Qwen3-ASR 1.7B");
        assert_eq!(model_1_7b.dir_name(), "qwen3-asr-1.7b");
        assert!(model_1_7b.size_gb() > model_0_6b.size_gb());

        assert_eq!(ConsolidatedModel::all_models().len(), 2);
    }

    #[test]
    fn test_pipeline_mode() {
        let config = Config::default();
        assert_eq!(config.pipeline_mode, PipelineMode::SttPlusLlm);
        assert!(!config.is_consolidated_mode());

        let mut config = Config::default();
        config.pipeline_mode = PipelineMode::Consolidated;
        assert!(config.is_consolidated_mode());
    }

    #[test]
    fn test_env_var_names() {
        // Ensure all env var names are unique and properly prefixed
        let vars = [
            env_vars::PIPELINE_MODE,
            env_vars::CONSOLIDATED_MODEL,
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
