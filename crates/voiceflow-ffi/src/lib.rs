//! C FFI bindings for VoiceFlow - for Swift/macOS app integration
//!
//! Build: cargo build --release -p voiceflow-ffi
//! This generates a dylib/staticlib that can be linked from Swift

use std::ffi::{c_char, c_float, CStr, CString};
use std::ptr;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::runtime::Runtime;
use voiceflow_core::{Config, Pipeline, PipelineMode, ConsolidatedModel, VlmModel, runtime as core_runtime};

// =============================================================================
// Global Singleton Tokio Runtime
// =============================================================================

/// Singleton Tokio runtime shared across all FFI calls.
/// This prevents the memory leak caused by creating a new runtime per LLM call.
static RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    log_debug("Creating global Tokio runtime");
    let rt = Arc::new(Runtime::new().expect("Failed to create Tokio runtime"));

    // Register with voiceflow-core so LlmEngine can use it
    core_runtime::register_runtime(Arc::clone(&rt));
    log_debug("Runtime registered with voiceflow-core");

    rt
});

/// Memory tracking for debugging
static PEAK_MEMORY_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Get a reference to the global runtime
pub fn get_runtime() -> &'static Arc<Runtime> {
    &RUNTIME
}

/// Force initialization of the global runtime (call early in app lifecycle)
pub fn ensure_runtime_initialized() {
    let _ = &*RUNTIME;
}

/// Write debug log to file (since macOS GUI apps don't have stderr)
fn log_debug(msg: &str) {
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/voiceflow_debug.log")
    {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(file, "[{}] {}", timestamp, msg);
    }
}

/// Opaque handle to the VoiceFlow pipeline
pub struct VoiceFlowHandle {
    /// Pipeline is None in consolidated mode (Swift/MLX handles inference)
    pipeline: Option<Pipeline>,
    /// Cached config to avoid repeated file parsing
    config: Config,
}

/// Result struct returned to foreign callers
#[repr(C)]
pub struct VoiceFlowResult {
    pub success: bool,
    pub formatted_text: *mut c_char,
    pub raw_transcript: *mut c_char,
    pub error_message: *mut c_char,
    pub transcription_ms: u64,
    pub llm_ms: u64,
    pub total_ms: u64,
}

/// Set up ONNX Runtime library path for macOS app bundle
fn setup_onnx_runtime_path() {
    // Only set if not already set
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        return;
    }

    // Try to find libonnxruntime.dylib relative to the executable
    if let Ok(exe_path) = std::env::current_exe() {
        // App bundle structure: VoiceFlow.app/Contents/MacOS/VoiceFlow
        // Library is at: VoiceFlow.app/Contents/Frameworks/libonnxruntime.dylib
        if let Some(macos_dir) = exe_path.parent() {
            let frameworks_dir = macos_dir.parent().map(|p| p.join("Frameworks"));
            if let Some(frameworks) = frameworks_dir {
                let ort_path = frameworks.join("libonnxruntime.dylib");
                if ort_path.exists() {
                    log_debug(&format!("Setting ORT_DYLIB_PATH to {:?}", ort_path));
                    std::env::set_var("ORT_DYLIB_PATH", &ort_path);
                    return;
                }
            }
        }
    }

    log_debug("Could not find bundled libonnxruntime.dylib, will use system path");
}

/// Initialize the VoiceFlow pipeline
///
/// # Safety
/// config_path must be a valid null-terminated string or null for default
#[no_mangle]
pub unsafe extern "C" fn voiceflow_init(config_path: *const c_char) -> *mut VoiceFlowHandle {
    log_debug("voiceflow_init called");

    // Initialize the global Tokio runtime early
    ensure_runtime_initialized();
    log_debug("Global Tokio runtime initialized");

    // Set up ONNX Runtime path before any ort calls
    setup_onnx_runtime_path();

    // Wrap everything in catch_unwind to prevent panics from unwinding across FFI boundary
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let config_str = if config_path.is_null() {
            None
        } else {
            match CStr::from_ptr(config_path).to_str() {
                Ok(s) => Some(s),
                Err(_) => return ptr::null_mut(),
            }
        };

        let config = match Config::load(config_str) {
            Ok(c) => {
                log_debug(&format!("Config loaded: STT={:?}", c.stt_engine));
                c
            },
            Err(e) => {
                log_debug(&format!("Failed to load config: {}", e));
                return ptr::null_mut();
            }
        };

        // In consolidated mode, skip pipeline creation (Swift/MLX handles inference)
        if config.is_consolidated_mode() {
            log_debug("Consolidated mode - skipping pipeline creation (MLX Swift handles inference)");
            log_debug("voiceflow_init complete - returning handle (consolidated mode)");
            return Box::into_raw(Box::new(VoiceFlowHandle { pipeline: None, config }));
        }

        log_debug("Creating pipeline (loading ONNX models - this may take a while)...");
        let pipeline = match Pipeline::new(&config) {
            Ok(p) => {
                log_debug("Pipeline created successfully");
                p
            },
            Err(e) => {
                log_debug(&format!("Failed to create pipeline: {}", e));
                return ptr::null_mut();
            }
        };

        log_debug("voiceflow_init complete - returning handle");
        Box::into_raw(Box::new(VoiceFlowHandle { pipeline: Some(pipeline), config }))
    }));

    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            log_debug(&format!("PANIC caught in voiceflow_init: {}", msg));
            ptr::null_mut()
        }
    }
}

/// Process audio samples and return formatted text
///
/// # Safety
/// - handle must be a valid pointer from voiceflow_init
/// - audio_data must point to audio_len floats (16kHz mono PCM)
/// - context can be null
#[no_mangle]
pub unsafe extern "C" fn voiceflow_process(
    handle: *mut VoiceFlowHandle,
    audio_data: *const c_float,
    audio_len: usize,
    context: *const c_char,
) -> VoiceFlowResult {
    log_debug(&format!("voiceflow_process called with {} samples", audio_len));

    if handle.is_null() || audio_data.is_null() {
        log_debug("ERROR - Invalid handle or audio data");
        return error_result("Invalid handle or audio data");
    }

    // Store raw pointers for use in closure
    let handle_ptr = handle;
    let audio_ptr = audio_data;
    let context_ptr = context;

    // Wrap in catch_unwind to prevent panics from unwinding across FFI boundary
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let handle = &mut *handle_ptr;
        let audio = std::slice::from_raw_parts(audio_ptr, audio_len);

        // Log audio stats
        let audio_duration = audio_len as f32 / 16000.0;
        let max_val = audio.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        log_debug(&format!("Audio duration: {:.2}s, max amplitude: {:.4}", audio_duration, max_val));

        let context_str = if context_ptr.is_null() {
            None
        } else {
            match CStr::from_ptr(context_ptr).to_str() {
                Ok(s) => Some(s),
                Err(_) => None,
            }
        };

        let pipeline = match handle.pipeline.as_mut() {
            Some(p) => p,
            None => {
                log_debug("ERROR - Pipeline not available (consolidated mode). Use MLX Swift for inference.");
                return error_result("Pipeline not available in consolidated mode. Use MLX Swift for inference.");
            }
        };

        log_debug("Calling pipeline.process()...");
        match pipeline.process(audio, context_str) {
            Ok(result) => {
                log_debug(&format!("Success! Raw transcript: '{}'", result.raw_transcript));
                log_debug(&format!("Formatted text: '{}'", result.formatted_text));
                VoiceFlowResult {
                    success: true,
                    formatted_text: CString::new(result.formatted_text)
                        .map(|s| s.into_raw())
                        .unwrap_or(ptr::null_mut()),
                    raw_transcript: CString::new(result.raw_transcript)
                        .map(|s| s.into_raw())
                        .unwrap_or(ptr::null_mut()),
                    error_message: ptr::null_mut(),
                    transcription_ms: result.timings.transcription_ms,
                    llm_ms: result.timings.llm_formatting_ms,
                    total_ms: result.timings.total_ms,
                }
            },
            Err(e) => {
                log_debug(&format!("ERROR - pipeline.process failed: {}", e));
                error_result(&e.to_string())
            },
        }
    }));

    match result {
        Ok(vf_result) => vf_result,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            log_debug(&format!("PANIC caught in voiceflow_process: {}", msg));
            error_result(&format!("Internal error: {}", msg))
        }
    }
}

/// Free a VoiceFlowResult's strings
///
/// # Safety
/// Only call this once per result
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_result(result: VoiceFlowResult) {
    if !result.formatted_text.is_null() {
        let _ = CString::from_raw(result.formatted_text);
    }
    if !result.raw_transcript.is_null() {
        let _ = CString::from_raw(result.raw_transcript);
    }
    if !result.error_message.is_null() {
        let _ = CString::from_raw(result.error_message);
    }
}

/// Cleanup and free the handle
///
/// # Safety
/// Only call this once per handle
#[no_mangle]
pub unsafe extern "C" fn voiceflow_destroy(handle: *mut VoiceFlowHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Get the library version
#[no_mangle]
pub extern "C" fn voiceflow_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

fn error_result(msg: &str) -> VoiceFlowResult {
    VoiceFlowResult {
        success: false,
        formatted_text: ptr::null_mut(),
        raw_transcript: ptr::null_mut(),
        error_message: CString::new(msg)
            .map(|s| s.into_raw())
            .unwrap_or(ptr::null_mut()),
        transcription_ms: 0,
        llm_ms: 0,
        total_ms: 0,
    }
}

/// Model info struct for FFI
#[repr(C)]
pub struct ModelInfo {
    pub id: *mut c_char,
    pub display_name: *mut c_char,
    pub filename: *mut c_char,
    pub size_gb: c_float,
    pub is_downloaded: bool,
}

/// Get the models directory path
#[no_mangle]
pub extern "C" fn voiceflow_models_dir() -> *mut c_char {
    match Config::models_dir() {
        Ok(path) => CString::new(path.to_string_lossy().to_string())
            .map(|s| s.into_raw())
            .unwrap_or(ptr::null_mut()),
        Err(_) => ptr::null_mut(),
    }
}

/// Get the number of available models
#[no_mangle]
pub extern "C" fn voiceflow_model_count() -> usize {
    use voiceflow_core::config::LlmModel;
    LlmModel::all_models().len()
}

/// Get model info by index
///
/// # Safety
/// index must be < voiceflow_model_count()
#[no_mangle]
pub unsafe extern "C" fn voiceflow_model_info(index: usize) -> ModelInfo {
    use voiceflow_core::config::LlmModel;

    let models = LlmModel::all_models();
    if index >= models.len() {
        return ModelInfo {
            id: ptr::null_mut(),
            display_name: ptr::null_mut(),
            filename: ptr::null_mut(),
            size_gb: 0.0,
            is_downloaded: false,
        };
    }

    let model = &models[index];
    let models_dir = Config::models_dir().ok();
    let is_downloaded = models_dir
        .map(|dir| dir.join(model.filename()).exists())
        .unwrap_or(false);

    let id_str = match model {
        LlmModel::Qwen3_1_7B => "qwen3-1.7b",
        LlmModel::Qwen3_4B => "qwen3-4b",
        LlmModel::SmolLM3_3B => "smollm3-3b",
        LlmModel::Gemma2_2B => "gemma2-2b",
        LlmModel::Gemma3nE2B => "gemma3n-e2b",
        LlmModel::Gemma3nE4B => "gemma3n-e4b",
        LlmModel::Phi4Mini => "phi4-mini",
        LlmModel::Phi2 => "phi-2",
        LlmModel::Custom(_) => "custom",
    };

    ModelInfo {
        id: CString::new(id_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        display_name: CString::new(model.display_name()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        filename: CString::new(model.filename()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        size_gb: model.size_gb(),
        is_downloaded,
    }
}

/// Free model info strings
///
/// # Safety
/// Only call once per ModelInfo
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_model_info(info: ModelInfo) {
    if !info.id.is_null() {
        let _ = CString::from_raw(info.id);
    }
    if !info.display_name.is_null() {
        let _ = CString::from_raw(info.display_name);
    }
    if !info.filename.is_null() {
        let _ = CString::from_raw(info.filename);
    }
}

/// Free a C string returned by other functions
///
/// # Safety
/// Only call once per string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

/// Get the current model ID from config
#[no_mangle]
pub extern "C" fn voiceflow_current_model() -> *mut c_char {
    use voiceflow_core::config::LlmModel;

    let config = Config::load(None).unwrap_or_default();
    let id_str = match config.llm_model {
        LlmModel::Qwen3_1_7B => "qwen3-1.7b",
        LlmModel::Qwen3_4B => "qwen3-4b",
        LlmModel::SmolLM3_3B => "smollm3-3b",
        LlmModel::Gemma2_2B => "gemma2-2b",
        LlmModel::Gemma3nE2B => "gemma3n-e2b",
        LlmModel::Gemma3nE4B => "gemma3n-e4b",
        LlmModel::Phi4Mini => "phi4-mini",
        LlmModel::Phi2 => "phi-2",
        LlmModel::Custom(_) => "custom",
    };

    CString::new(id_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Set the current model in config (requires restart to take effect)
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_set_model(model_id: *const c_char) -> bool {
    use voiceflow_core::config::LlmModel;

    if model_id.is_null() {
        return false;
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match id_str {
        "qwen3-1.7b" => LlmModel::Qwen3_1_7B,
        "qwen3-4b" => LlmModel::Qwen3_4B,
        "smollm3-3b" => LlmModel::SmolLM3_3B,
        "gemma2-2b" => LlmModel::Gemma2_2B,
        "phi-2" => LlmModel::Phi2,
        _ => return false,
    };

    let mut config = Config::load(None).unwrap_or_default();
    config.llm_model = model;
    config.save(None).is_ok()
}

/// Get the HuggingFace download URL for a model
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_model_download_url(model_id: *const c_char) -> *mut c_char {
    use voiceflow_core::config::LlmModel;

    if model_id.is_null() {
        return ptr::null_mut();
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let model = match id_str {
        "qwen3-1.7b" => LlmModel::Qwen3_1_7B,
        "qwen3-4b" => LlmModel::Qwen3_4B,
        "smollm3-3b" => LlmModel::SmolLM3_3B,
        "gemma2-2b" => LlmModel::Gemma2_2B,
        "phi-2" => LlmModel::Phi2,
        _ => return ptr::null_mut(),
    };

    if let Some(repo) = model.hf_repo() {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo,
            model.filename()
        );
        CString::new(url).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
    } else {
        ptr::null_mut()
    }
}

// =============================================================================
// STT Engine Management
// =============================================================================

/// Get the current STT engine ("whisper", "moonshine", or "qwen3-asr")
#[no_mangle]
pub extern "C" fn voiceflow_current_stt_engine() -> *mut c_char {
    use voiceflow_core::config::SttEngine;

    let config = Config::load(None).unwrap_or_default();
    let engine_str = match config.stt_engine {
        SttEngine::Whisper => "whisper",
        SttEngine::Moonshine => "moonshine",
        SttEngine::Qwen3Asr => "qwen3-asr",
    };

    CString::new(engine_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Set the current STT engine ("whisper", "moonshine", or "qwen3-asr")
///
/// # Safety
/// engine_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_set_stt_engine(engine_id: *const c_char) -> bool {
    use voiceflow_core::config::SttEngine;

    if engine_id.is_null() {
        return false;
    }

    let engine_str = match CStr::from_ptr(engine_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let engine = match engine_str {
        "whisper" => SttEngine::Whisper,
        "moonshine" => SttEngine::Moonshine,
        "qwen3-asr" => SttEngine::Qwen3Asr,
        _ => return false,
    };

    let mut config = Config::load(None).unwrap_or_default();
    config.stt_engine = engine;
    config.save(None).is_ok()
}

/// Get the current Moonshine model ("tiny" or "base")
#[no_mangle]
pub extern "C" fn voiceflow_current_moonshine_model() -> *mut c_char {
    use voiceflow_core::config::MoonshineModel;

    let config = Config::load(None).unwrap_or_default();
    let model_str = match config.moonshine_model {
        MoonshineModel::Tiny => "tiny",
        MoonshineModel::Base => "base",
    };

    CString::new(model_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Set the current Moonshine model ("tiny" or "base")
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_set_moonshine_model(model_id: *const c_char) -> bool {
    use voiceflow_core::config::MoonshineModel;

    if model_id.is_null() {
        return false;
    }

    let model_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match model_str {
        "tiny" => MoonshineModel::Tiny,
        "base" => MoonshineModel::Base,
        _ => return false,
    };

    let mut config = Config::load(None).unwrap_or_default();
    config.moonshine_model = model;
    config.save(None).is_ok()
}

/// Moonshine model info struct for FFI
#[repr(C)]
pub struct MoonshineModelInfo {
    pub id: *mut c_char,
    pub display_name: *mut c_char,
    pub size_mb: u32,
    pub is_downloaded: bool,
}

/// Get the number of available Moonshine models
#[no_mangle]
pub extern "C" fn voiceflow_moonshine_model_count() -> usize {
    2 // Tiny and Base
}

/// Get Moonshine model info by index
///
/// # Safety
/// index must be < voiceflow_moonshine_model_count()
#[no_mangle]
pub unsafe extern "C" fn voiceflow_moonshine_model_info(index: usize) -> MoonshineModelInfo {
    use voiceflow_core::config::MoonshineModel;

    let model = match index {
        0 => MoonshineModel::Tiny,
        1 => MoonshineModel::Base,
        _ => return MoonshineModelInfo {
            id: ptr::null_mut(),
            display_name: ptr::null_mut(),
            size_mb: 0,
            is_downloaded: false,
        },
    };

    let config = Config::load(None).unwrap_or_default();
    let is_downloaded = config.moonshine_model_downloaded_for(&model);

    let id_str = match model {
        MoonshineModel::Tiny => "tiny",
        MoonshineModel::Base => "base",
    };

    MoonshineModelInfo {
        id: CString::new(id_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        display_name: CString::new(model.display_name()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        size_mb: model.size_mb(),
        is_downloaded,
    }
}

/// Free Moonshine model info strings
///
/// # Safety
/// Only call once per MoonshineModelInfo
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_moonshine_model_info(info: MoonshineModelInfo) {
    if !info.id.is_null() {
        let _ = CString::from_raw(info.id);
    }
    if !info.display_name.is_null() {
        let _ = CString::from_raw(info.display_name);
    }
}

/// Check if a Moonshine model is downloaded
///
/// # Safety
/// model_id must be a valid null-terminated string ("tiny" or "base")
#[no_mangle]
pub unsafe extern "C" fn voiceflow_moonshine_model_downloaded(model_id: *const c_char) -> bool {
    use voiceflow_core::config::MoonshineModel;

    if model_id.is_null() {
        return false;
    }

    let model_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match model_str {
        "tiny" => MoonshineModel::Tiny,
        "base" => MoonshineModel::Base,
        _ => return false,
    };

    let config = Config::load(None).unwrap_or_default();
    config.moonshine_model_downloaded_for(&model)
}

/// Get the Moonshine models directory path
#[no_mangle]
pub extern "C" fn voiceflow_moonshine_models_dir() -> *mut c_char {
    let config = Config::load(None).unwrap_or_default();
    match config.moonshine_model_dir() {
        Ok(path) => {
            // Return parent directory (models dir)
            if let Some(parent) = path.parent() {
                CString::new(parent.to_string_lossy().to_string())
                    .map(|s| s.into_raw())
                    .unwrap_or(ptr::null_mut())
            } else {
                ptr::null_mut()
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

// =============================================================================
// Pipeline Mode and Consolidated Model Management
// =============================================================================

/// Get the current pipeline mode ("stt-plus-llm" or "consolidated")
#[no_mangle]
pub extern "C" fn voiceflow_current_pipeline_mode() -> *mut c_char {
    let config = Config::load(None).unwrap_or_default();
    let mode_str = match config.pipeline_mode {
        PipelineMode::SttPlusLlm => "stt-plus-llm",
        PipelineMode::Consolidated => "consolidated",
    };

    CString::new(mode_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Set the pipeline mode ("stt-plus-llm" or "consolidated")
///
/// # Safety
/// mode must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_set_pipeline_mode(mode: *const c_char) -> bool {
    if mode.is_null() {
        return false;
    }

    let mode_str = match CStr::from_ptr(mode).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let pipeline_mode = match mode_str {
        "stt-plus-llm" | "traditional" => PipelineMode::SttPlusLlm,
        "consolidated" => PipelineMode::Consolidated,
        _ => return false,
    };

    let mut config = Config::load(None).unwrap_or_default();
    config.pipeline_mode = pipeline_mode;
    config.save(None).is_ok()
}

/// Get the current consolidated model ID
#[no_mangle]
pub extern "C" fn voiceflow_current_consolidated_model() -> *mut c_char {
    let config = Config::load(None).unwrap_or_default();
    let id_str = match config.consolidated_model {
        ConsolidatedModel::Qwen3Asr0_6B => "qwen3-asr-0.6b",
        ConsolidatedModel::Qwen3Asr1_7B => "qwen3-asr-1.7b",
    };

    CString::new(id_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Set the consolidated model
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_set_consolidated_model(model_id: *const c_char) -> bool {
    if model_id.is_null() {
        return false;
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match id_str {
        "qwen3-asr-0.6b" => ConsolidatedModel::Qwen3Asr0_6B,
        "qwen3-asr-1.7b" => ConsolidatedModel::Qwen3Asr1_7B,
        _ => return false,
    };

    let mut config = Config::load(None).unwrap_or_default();
    config.consolidated_model = model;
    config.save(None).is_ok()
}

/// Consolidated model info struct for FFI
#[repr(C)]
pub struct ConsolidatedModelInfo {
    pub id: *mut c_char,
    pub display_name: *mut c_char,
    pub dir_name: *mut c_char,
    pub size_gb: c_float,
    pub is_downloaded: bool,
}

/// Get the number of available consolidated models
#[no_mangle]
pub extern "C" fn voiceflow_consolidated_model_count() -> usize {
    ConsolidatedModel::all_models().len()
}

/// Get consolidated model info by index
///
/// # Safety
/// index must be < voiceflow_consolidated_model_count()
#[no_mangle]
pub unsafe extern "C" fn voiceflow_consolidated_model_info(index: usize) -> ConsolidatedModelInfo {
    let models = ConsolidatedModel::all_models();
    if index >= models.len() {
        return ConsolidatedModelInfo {
            id: ptr::null_mut(),
            display_name: ptr::null_mut(),
            dir_name: ptr::null_mut(),
            size_gb: 0.0,
            is_downloaded: false,
        };
    }

    let model = &models[index];
    let config = Config::load(None).unwrap_or_default();
    let is_downloaded = config.consolidated_model_downloaded_for(model);

    let id_str = match model {
        ConsolidatedModel::Qwen3Asr0_6B => "qwen3-asr-0.6b",
        ConsolidatedModel::Qwen3Asr1_7B => "qwen3-asr-1.7b",
    };

    ConsolidatedModelInfo {
        id: CString::new(id_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        display_name: CString::new(model.display_name()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        dir_name: CString::new(model.dir_name()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        size_gb: model.size_gb(),
        is_downloaded,
    }
}

/// Free consolidated model info strings
///
/// # Safety
/// Only call once per ConsolidatedModelInfo
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_consolidated_model_info(info: ConsolidatedModelInfo) {
    if !info.id.is_null() {
        let _ = CString::from_raw(info.id);
    }
    if !info.display_name.is_null() {
        let _ = CString::from_raw(info.display_name);
    }
    if !info.dir_name.is_null() {
        let _ = CString::from_raw(info.dir_name);
    }
}

/// Check if a consolidated model is downloaded
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_consolidated_model_downloaded(model_id: *const c_char) -> bool {
    if model_id.is_null() {
        return false;
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match id_str {
        "qwen3-asr-0.6b" => ConsolidatedModel::Qwen3Asr0_6B,
        "qwen3-asr-1.7b" => ConsolidatedModel::Qwen3Asr1_7B,
        _ => return false,
    };

    let config = Config::load(None).unwrap_or_default();
    config.consolidated_model_downloaded_for(&model)
}

/// Get the directory path for a consolidated model
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_consolidated_model_dir(model_id: *const c_char) -> *mut c_char {
    if model_id.is_null() {
        return ptr::null_mut();
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let model = match id_str {
        "qwen3-asr-0.6b" => ConsolidatedModel::Qwen3Asr0_6B,
        "qwen3-asr-1.7b" => ConsolidatedModel::Qwen3Asr1_7B,
        _ => return ptr::null_mut(),
    };

    match Config::models_dir() {
        Ok(dir) => {
            let model_dir = dir.join(model.dir_name());
            CString::new(model_dir.to_string_lossy().to_string())
                .map(|s| s.into_raw())
                .unwrap_or(ptr::null_mut())
        }
        Err(_) => ptr::null_mut(),
    }
}

// =============================================================================
// VLM (Vision-Language Model) Management
// =============================================================================

/// VLM model info struct for FFI
#[repr(C)]
pub struct VlmModelInfo {
    pub id: *mut c_char,
    pub display_name: *mut c_char,
    pub dir_name: *mut c_char,
    pub size_gb: c_float,
    pub is_downloaded: bool,
}

/// Get the number of available VLM models
#[no_mangle]
pub extern "C" fn voiceflow_vlm_model_count() -> usize {
    VlmModel::all_models().len()
}

/// Get VLM model info by index
///
/// # Safety
/// index must be < voiceflow_vlm_model_count()
#[no_mangle]
pub unsafe extern "C" fn voiceflow_vlm_model_info(index: usize) -> VlmModelInfo {
    let models = VlmModel::all_models();
    if index >= models.len() {
        return VlmModelInfo {
            id: ptr::null_mut(),
            display_name: ptr::null_mut(),
            dir_name: ptr::null_mut(),
            size_gb: 0.0,
            is_downloaded: false,
        };
    }

    let model = &models[index];
    let config = Config::load(None).unwrap_or_default();
    let is_downloaded = config.vlm_model_downloaded_for(model);

    let id_str = match model {
        VlmModel::JinaVlm => "jina-vlm",
        VlmModel::Qwen3VL2B => "qwen3-vl-2b",
    };

    VlmModelInfo {
        id: CString::new(id_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        display_name: CString::new(model.display_name()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        dir_name: CString::new(model.dir_name()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        size_gb: model.size_gb(),
        is_downloaded,
    }
}

/// Free VLM model info strings
///
/// # Safety
/// Only call once per VlmModelInfo
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_vlm_model_info(info: VlmModelInfo) {
    if !info.id.is_null() {
        let _ = CString::from_raw(info.id);
    }
    if !info.display_name.is_null() {
        let _ = CString::from_raw(info.display_name);
    }
    if !info.dir_name.is_null() {
        let _ = CString::from_raw(info.dir_name);
    }
}

/// Check if a VLM model is downloaded
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_vlm_model_downloaded(model_id: *const c_char) -> bool {
    if model_id.is_null() {
        return false;
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match id_str {
        "jina-vlm" => VlmModel::JinaVlm,
        "qwen3-vl-2b" => VlmModel::Qwen3VL2B,
        _ => return false,
    };

    let config = Config::load(None).unwrap_or_default();
    config.vlm_model_downloaded_for(&model)
}

/// Get the HuggingFace repo for a VLM model
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_vlm_model_hf_repo(model_id: *const c_char) -> *mut c_char {
    if model_id.is_null() {
        return ptr::null_mut();
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let model = match id_str {
        "jina-vlm" => VlmModel::JinaVlm,
        "qwen3-vl-2b" => VlmModel::Qwen3VL2B,
        _ => return ptr::null_mut(),
    };

    CString::new(model.hf_repo()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Get the number of required files for a VLM model
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_vlm_model_file_count(model_id: *const c_char) -> usize {
    if model_id.is_null() {
        return 0;
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let model = match id_str {
        "jina-vlm" => VlmModel::JinaVlm,
        "qwen3-vl-2b" => VlmModel::Qwen3VL2B,
        _ => return 0,
    };

    model.required_files().len()
}

/// Get a required file name for a VLM model by index
///
/// # Safety
/// model_id must be a valid null-terminated string, index must be < file_count
#[no_mangle]
pub unsafe extern "C" fn voiceflow_vlm_model_file_name(model_id: *const c_char, index: usize) -> *mut c_char {
    if model_id.is_null() {
        return ptr::null_mut();
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let model = match id_str {
        "jina-vlm" => VlmModel::JinaVlm,
        "qwen3-vl-2b" => VlmModel::Qwen3VL2B,
        _ => return ptr::null_mut(),
    };

    let files = model.required_files();
    if index >= files.len() {
        return ptr::null_mut();
    }

    CString::new(files[index]).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Get the directory path for a VLM model
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_vlm_model_dir(model_id: *const c_char) -> *mut c_char {
    if model_id.is_null() {
        return ptr::null_mut();
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let model = match id_str {
        "jina-vlm" => VlmModel::JinaVlm,
        "qwen3-vl-2b" => VlmModel::Qwen3VL2B,
        _ => return ptr::null_mut(),
    };

    match Config::models_dir() {
        Ok(dir) => {
            let model_dir = dir.join(model.dir_name());
            CString::new(model_dir.to_string_lossy().to_string())
                .map(|s| s.into_raw())
                .unwrap_or(ptr::null_mut())
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Get the current VLM model ID from config, or null if none selected
#[no_mangle]
pub extern "C" fn voiceflow_current_vlm_model() -> *mut c_char {
    let config = Config::load(None).unwrap_or_default();
    match config.vlm_model {
        Some(VlmModel::JinaVlm) => CString::new("jina-vlm").map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        Some(VlmModel::Qwen3VL2B) => CString::new("qwen3-vl-2b").map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        None => ptr::null_mut(),
    }
}

/// Set the current VLM model in config (requires restart to take effect)
/// Pass null to clear the VLM selection (revert to LLM-only mode)
///
/// # Safety
/// model_id must be a valid null-terminated string or null to clear
#[no_mangle]
pub unsafe extern "C" fn voiceflow_set_vlm_model(model_id: *const c_char) -> bool {
    let mut config = Config::load(None).unwrap_or_default();

    if model_id.is_null() {
        config.vlm_model = None;
        return config.save(None).is_ok();
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match id_str {
        "jina-vlm" => VlmModel::JinaVlm,
        "qwen3-vl-2b" => VlmModel::Qwen3VL2B,
        _ => return false,
    };

    config.vlm_model = Some(model);
    config.save(None).is_ok()
}

// =============================================================================
// Post-Processing
// =============================================================================

/// Apply post-processing to text (tokenization fix, voice commands, spelled words, replacements).
/// Used by consolidated mode where MLX Swift handles inference but Rust handles post-processing.
///
/// # Safety
/// text must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_post_process_text(text: *const c_char) -> *mut c_char {
    use voiceflow_core::prosody::{
        fix_tokenization_artifacts, replace_voice_commands,
        concatenate_spelled_words_aggressive, ReplacementDictionary,
        remove_filler_words,
    };

    if text.is_null() {
        return ptr::null_mut();
    }

    let input = match CStr::from_ptr(text).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return ptr::null_mut(),
    };

    // Apply the same post-processing pipeline as the traditional mode
    let mut result = fix_tokenization_artifacts(&input);
    result = remove_filler_words(&result);
    result = replace_voice_commands(&result);
    result = concatenate_spelled_words_aggressive(&result);

    let replacements = ReplacementDictionary::load_default();
    result = replacements.apply(&result);

    CString::new(result).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Process pre-transcribed text through post-processing + LLM formatting.
/// Used when an external STT engine (e.g. Qwen3-ASR Python daemon) provides the raw transcript
/// and we want the traditional pipeline's LLM formatting applied.
///
/// Returns a VoiceFlowResult with formatted_text and raw_transcript.
///
/// # Safety
/// - handle must be a valid pointer from voiceflow_init
/// - text must be a valid null-terminated string
/// - context can be null
#[no_mangle]
pub unsafe extern "C" fn voiceflow_format_text(
    handle: *mut VoiceFlowHandle,
    text: *const c_char,
    context: *const c_char,
) -> VoiceFlowResult {
    log_debug("voiceflow_format_text called");

    if handle.is_null() || text.is_null() {
        log_debug("ERROR - Invalid handle or text");
        return error_result("Invalid handle or text");
    }

    let handle_ptr = handle;
    let text_ptr = text;
    let context_ptr = context;

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let handle = &mut *handle_ptr;
        let input_text = match CStr::from_ptr(text_ptr).to_str() {
            Ok(s) => s,
            Err(_) => return error_result("Invalid UTF-8 text"),
        };

        let context_str = if context_ptr.is_null() {
            None
        } else {
            match CStr::from_ptr(context_ptr).to_str() {
                Ok(s) => Some(s),
                Err(_) => None,
            }
        };

        let pipeline = match handle.pipeline.as_mut() {
            Some(p) => p,
            None => {
                log_debug("ERROR - Pipeline not available. Cannot format text without pipeline.");
                return error_result("Pipeline not available. Initialize in traditional mode to use LLM formatting.");
            }
        };

        log_debug(&format!("Calling pipeline.process_text() with: '{}'", &input_text[..input_text.len().min(80)]));
        match pipeline.process_text(input_text, context_str) {
            Ok(result) => {
                log_debug(&format!("Success! Formatted text: '{}'", result.formatted_text));
                VoiceFlowResult {
                    success: true,
                    formatted_text: CString::new(result.formatted_text)
                        .map(|s| s.into_raw())
                        .unwrap_or(ptr::null_mut()),
                    raw_transcript: CString::new(result.raw_transcript)
                        .map(|s| s.into_raw())
                        .unwrap_or(ptr::null_mut()),
                    error_message: ptr::null_mut(),
                    transcription_ms: 0,
                    llm_ms: result.timings.llm_formatting_ms,
                    total_ms: result.timings.total_ms,
                }
            }
            Err(e) => {
                log_debug(&format!("ERROR - process_text failed: {}", e));
                error_result(&e.to_string())
            }
        }
    }));

    match result {
        Ok(vf_result) => vf_result,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            log_debug(&format!("PANIC caught in voiceflow_format_text: {}", msg));
            error_result(&format!("Internal error: {}", msg))
        }
    }
}

/// Check if the current STT engine is external (handled outside Rust pipeline)
#[no_mangle]
pub extern "C" fn voiceflow_is_external_stt() -> bool {
    let config = Config::load(None).unwrap_or_default();
    config.stt_engine.is_external()
}

/// Check if the current config is in consolidated mode
#[no_mangle]
pub extern "C" fn voiceflow_is_consolidated_mode() -> bool {
    let config = Config::load(None).unwrap_or_default();
    config.is_consolidated_mode()
}

// =============================================================================
// Memory Management and Cleanup Functions
// =============================================================================

/// Unload all models from memory (LLM and STT)
/// Call this before app termination or when switching models
///
/// # Safety
/// handle must be a valid pointer from voiceflow_init
#[no_mangle]
pub unsafe extern "C" fn voiceflow_unload_models(handle: *mut VoiceFlowHandle) {
    if handle.is_null() {
        return;
    }

    log_debug("voiceflow_unload_models called");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let handle = &mut *handle;
        if let Some(ref mut pipeline) = handle.pipeline {
            pipeline.unload_all();
            log_debug("All models unloaded");
        } else {
            log_debug("No pipeline to unload (consolidated mode)");
        }
    }));

    if let Err(e) = result {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "Unknown panic".to_string()
        };
        log_debug(&format!("PANIC in voiceflow_unload_models: {}", msg));
    }
}

/// Reset the LLM engine, allowing re-initialization on next use
///
/// # Safety
/// handle must be a valid pointer from voiceflow_init
#[no_mangle]
pub unsafe extern "C" fn voiceflow_reset_llm(handle: *mut VoiceFlowHandle) {
    if handle.is_null() {
        return;
    }

    log_debug("voiceflow_reset_llm called");
    let handle = &mut *handle;
    if let Some(ref mut pipeline) = handle.pipeline {
        pipeline.reset_llm();
    }
}

/// Get approximate memory usage in bytes
/// This is a rough estimate based on tracked allocations
#[no_mangle]
pub extern "C" fn voiceflow_memory_usage() -> usize {
    // Try to get process memory info on macOS
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;

        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: u64,
            system_time: u64,
            policy: i32,
            suspend_count: i32,
        }

        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: i32,
                task_info_out: *mut MachTaskBasicInfo,
                task_info_count: *mut u32,
            ) -> i32;
        }

        const MACH_TASK_BASIC_INFO: i32 = 20;
        const MACH_TASK_BASIC_INFO_COUNT: u32 = 10;

        unsafe {
            let mut info = MaybeUninit::<MachTaskBasicInfo>::uninit();
            let mut count = MACH_TASK_BASIC_INFO_COUNT;

            let result = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                info.as_mut_ptr(),
                &mut count,
            );

            if result == 0 {
                let info = info.assume_init();
                return info.resident_size as usize;
            }
        }
    }

    // Fallback: return tracked peak
    PEAK_MEMORY_BYTES.load(Ordering::Relaxed)
}

/// Memory info struct for FFI
#[repr(C)]
pub struct MemoryInfo {
    pub resident_bytes: usize,
    pub virtual_bytes: usize,
    pub peak_bytes: usize,
}

/// Get detailed memory information
#[no_mangle]
pub extern "C" fn voiceflow_memory_info() -> MemoryInfo {
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;

        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: u64,
            system_time: u64,
            policy: i32,
            suspend_count: i32,
        }

        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: i32,
                task_info_out: *mut MachTaskBasicInfo,
                task_info_count: *mut u32,
            ) -> i32;
        }

        const MACH_TASK_BASIC_INFO: i32 = 20;
        const MACH_TASK_BASIC_INFO_COUNT: u32 = 10;

        unsafe {
            let mut info = MaybeUninit::<MachTaskBasicInfo>::uninit();
            let mut count = MACH_TASK_BASIC_INFO_COUNT;

            let result = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                info.as_mut_ptr(),
                &mut count,
            );

            if result == 0 {
                let info = info.assume_init();

                // Update peak tracking
                let current = info.resident_size as usize;
                PEAK_MEMORY_BYTES.fetch_max(current, Ordering::Relaxed);

                return MemoryInfo {
                    resident_bytes: info.resident_size as usize,
                    virtual_bytes: info.virtual_size as usize,
                    peak_bytes: info.resident_size_max as usize,
                };
            }
        }
    }

    MemoryInfo {
        resident_bytes: 0,
        virtual_bytes: 0,
        peak_bytes: PEAK_MEMORY_BYTES.load(Ordering::Relaxed),
    }
}

/// Force a garbage collection hint
/// On Rust, this mainly drops any cached allocator memory
#[no_mangle]
pub extern "C" fn voiceflow_force_gc() {
    log_debug("voiceflow_force_gc called");

    // Trigger a malloc_trim on Linux, or similar cleanup
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        unsafe {
            malloc_trim(0);
        }
    }

    // On macOS, we can't force system GC, but we can log memory state
    #[cfg(target_os = "macos")]
    {
        let info = voiceflow_memory_info();
        log_debug(&format!(
            "Memory after GC hint: resident={}MB, virtual={}MB, peak={}MB",
            info.resident_bytes / 1024 / 1024,
            info.virtual_bytes / 1024 / 1024,
            info.peak_bytes / 1024 / 1024
        ));
    }
}

/// Prepare for app shutdown - ensures clean resource release
/// Call this before NSApp.terminate() for clean shutdown
#[no_mangle]
pub extern "C" fn voiceflow_prepare_shutdown() {
    log_debug("voiceflow_prepare_shutdown called");

    // Log final memory state
    let info = voiceflow_memory_info();
    log_debug(&format!(
        "Final memory state: resident={}MB, peak={}MB",
        info.resident_bytes / 1024 / 1024,
        info.peak_bytes / 1024 / 1024
    ));

    log_debug("Shutdown preparation complete");
}
