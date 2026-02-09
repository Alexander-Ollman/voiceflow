//! Global Tokio runtime management for VoiceFlow
//!
//! This module provides a way to register and use a shared Tokio runtime
//! across all async operations in voiceflow-core. This prevents memory leaks
//! from creating multiple runtimes per LLM call.

use once_cell::sync::OnceCell;
use std::sync::Arc;
use tokio::runtime::{Handle, Runtime};

/// Global runtime storage - set once by FFI layer, used by all async code
static GLOBAL_RUNTIME: OnceCell<Arc<Runtime>> = OnceCell::new();

/// Register a Tokio runtime for use by voiceflow-core
///
/// This should be called once during FFI initialization. If not called,
/// async code will create temporary runtimes (less efficient but still works).
pub fn register_runtime(runtime: Arc<Runtime>) {
    let _ = GLOBAL_RUNTIME.set(runtime);
    tracing::debug!("Global Tokio runtime registered with voiceflow-core");
}

/// Get the registered runtime, if available
pub fn get_runtime() -> Option<&'static Arc<Runtime>> {
    GLOBAL_RUNTIME.get()
}

/// Get a runtime handle, preferring the registered global runtime
///
/// This function tries, in order:
/// 1. The registered global runtime
/// 2. The current tokio runtime handle (if running in async context)
/// 3. Returns None (caller should create a temporary runtime)
pub fn get_runtime_handle() -> Option<Handle> {
    // First, try the registered global runtime
    if let Some(rt) = GLOBAL_RUNTIME.get() {
        return Some(rt.handle().clone());
    }

    // Second, try to get the current async context's runtime
    if let Ok(handle) = Handle::try_current() {
        return Some(handle);
    }

    None
}

/// Run an async block using the best available runtime
///
/// This is the primary way to run async code from sync contexts in voiceflow-core.
/// It uses the registered runtime if available, otherwise creates a temporary one.
pub fn block_on<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T> + Send,
    T: Send,
{
    // Try registered global runtime first
    if let Some(rt) = GLOBAL_RUNTIME.get() {
        // If we're already inside a tokio runtime context (e.g. called from a
        // tokio worker thread or a thread that has an active Handle), we can't
        // nest block_on. Spawn a dedicated thread instead.
        if Handle::try_current().is_ok() {
            let rt = Arc::clone(rt);
            return std::thread::scope(|s| {
                s.spawn(move || {
                    rt.block_on(future)
                }).join().expect("Thread panicked in block_on fallback")
            });
        }
        return rt.block_on(future);
    }

    // Try current runtime handle — we're in an async context without a global runtime
    if Handle::try_current().is_ok() {
        return std::thread::scope(|s| {
            s.spawn(|| {
                let rt = Runtime::new().expect("Failed to create fallback runtime");
                rt.block_on(future)
            }).join().expect("Thread panicked")
        });
    }

    // No runtime available - create temporary one
    // This should rarely happen after proper FFI initialization
    tracing::warn!("No global runtime available, creating temporary runtime");
    let rt = Runtime::new().expect("Failed to create temporary runtime");
    rt.block_on(future)
}
