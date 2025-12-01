//! C FFI bindings for VoiceFlow - for Swift/macOS app integration
//!
//! Build: cargo build --release -p voiceflow-ffi
//! This generates a dylib/staticlib that can be linked from Swift

use std::ffi::{c_char, c_float, CStr, CString};
use std::ptr;

use voiceflow_core::{Config, Pipeline};

/// Opaque handle to the VoiceFlow pipeline
pub struct VoiceFlowHandle {
    pipeline: Pipeline,
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

/// Initialize the VoiceFlow pipeline
///
/// # Safety
/// config_path must be a valid null-terminated string or null for default
#[no_mangle]
pub unsafe extern "C" fn voiceflow_init(config_path: *const c_char) -> *mut VoiceFlowHandle {
    let config_str = if config_path.is_null() {
        None
    } else {
        match CStr::from_ptr(config_path).to_str() {
            Ok(s) => Some(s),
            Err(_) => return ptr::null_mut(),
        }
    };

    let config = match Config::load(config_str) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load config: {}", e);
            return ptr::null_mut();
        }
    };

    let pipeline = match Pipeline::new(&config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to create pipeline: {}", e);
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(VoiceFlowHandle { pipeline }))
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
    if handle.is_null() || audio_data.is_null() {
        return error_result("Invalid handle or audio data");
    }

    let handle = &mut *handle;
    let audio = std::slice::from_raw_parts(audio_data, audio_len);

    let context_str = if context.is_null() {
        None
    } else {
        match CStr::from_ptr(context).to_str() {
            Ok(s) => Some(s),
            Err(_) => None,
        }
    };

    match handle.pipeline.process(audio, context_str) {
        Ok(result) => VoiceFlowResult {
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
        },
        Err(e) => error_result(&e.to_string()),
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
