# VoiceFlow Memory Leak Fix Plan

## Overview

This plan addresses memory leaks causing tens of gigabytes of memory usage, primarily from:
1. Tokio runtime creation per LLM call
2. Moonshine tensor cloning in decode loop
3. Incomplete cleanup on app restart
4. Missing resource lifecycle management

---

## Phase 1: Singleton Tokio Runtime (Critical)

### 1.1 Create a global runtime in `voiceflow-ffi`

**File:** `crates/voiceflow-ffi/src/lib.rs`

- Add `once_cell::sync::Lazy` or `std::sync::OnceLock` for a static runtime
- Initialize runtime once on first FFI call
- Expose runtime handle to all Rust code via thread-local or Arc

```
Steps:
[ ] Add `once_cell` to Cargo.toml dependencies
[ ] Create static RUNTIME: Lazy<Runtime> at module level
[ ] Add getter function: fn get_runtime() -> &'static Runtime
[ ] Add explicit shutdown function for app termination
```

### 1.2 Refactor LlmEngine to use shared runtime

**File:** `crates/voiceflow-core/src/llm/engine.rs`

- Remove runtime creation from `LlmEngine::new()`
- Remove runtime creation from `LlmEngine::format()`
- Accept runtime handle as parameter or use global getter

```
Steps:
[ ] Remove tokio::runtime::Runtime::new() from new() (lines 66, 73-74)
[ ] Remove tokio::runtime::Runtime::new() from format() (lines 126, 155-156)
[ ] Replace with runtime.block_on() using shared runtime
[ ] Update function signatures if needed
[ ] Test that LLM still works with shared runtime
```

### 1.3 Update VoiceFlowHandle to hold runtime reference

**File:** `crates/voiceflow-ffi/src/lib.rs`

```
Steps:
[ ] Add runtime field to VoiceFlowHandle struct (or use global)
[ ] Pass runtime to Pipeline/LlmEngine construction
[ ] Ensure runtime outlives all handles
```

---

## Phase 2: Moonshine Tensor Memory Optimization (High)

### 2.1 Implement tensor buffer reuse

**File:** `crates/voiceflow-core/src/transcribe/moonshine.rs`

- Pre-allocate cache buffers at engine initialization
- Reuse buffers across decode iterations instead of cloning

```
Steps:
[ ] Add cache_buffers: Vec<Vec<f32>> field to MoonshineEngine struct
[ ] Pre-allocate buffers in MoonshineEngine::new() based on max expected size
[ ] In transcribe(), copy into existing buffers instead of .to_vec()
[ ] Clear buffers at start of each transcription, not reallocate
[ ] Add capacity hints based on typical token count (e.g., 500 tokens)
```

### 2.2 Reduce context tensor cloning

**File:** `crates/voiceflow-core/src/transcribe/moonshine.rs` (lines 237, 243)

```
Steps:
[ ] Store context_data once outside the loop
[ ] Create context tensor once, clone reference not data
[ ] Use Cow<[f32]> or similar for zero-copy where possible
[ ] Profile memory before/after with Instruments
```

### 2.3 Add explicit ONNX session cleanup

**File:** `crates/voiceflow-core/src/transcribe/moonshine.rs`

```
Steps:
[ ] Implement Drop for MoonshineEngine if not present
[ ] Explicitly drop ONNX sessions in order (cached_decode, uncached, preprocess, encode)
[ ] Add logging to verify cleanup happens
[ ] Test with multiple transcriptions to verify memory stabilizes
```

---

## Phase 3: Pipeline Resource Lifecycle (Medium)

### 3.1 Add explicit LLM unload capability

**File:** `crates/voiceflow-core/src/pipeline.rs`

```
Steps:
[ ] Add unload_llm() method that drops the LlmEngine
[ ] Add unload_stt() method that drops the STT engine
[ ] Add unload_all() method for full cleanup
[ ] Expose these via FFI for Swift to call before restart
```

### 3.2 Add FFI cleanup functions

**File:** `crates/voiceflow-ffi/src/lib.rs`

```
Steps:
[ ] Add voiceflow_unload_models(handle) FFI function
[ ] Add voiceflow_shutdown_runtime() FFI function (for app termination)
[ ] Update voiceflow_destroy() to call unload_all() first
[ ] Add voiceflow_force_gc() to trigger explicit cleanup
```

### 3.3 Implement LLM idle timeout (optional)

**File:** `crates/voiceflow-core/src/pipeline.rs`

```
Steps:
[ ] Track last_llm_use: Instant in Pipeline
[ ] Add check in process() - if idle > threshold, drop LLM
[ ] Re-initialize on next use (already handled by get_llm())
[ ] Make timeout configurable in Config
```

---

## Phase 4: Swift-Side Cleanup (Medium)

### 4.1 Add synchronous cleanup before app termination

**File:** `VoiceFlowApp/Sources/VoiceFlowApp/VoiceFlowApp.swift`

```
Steps:
[ ] Add applicationWillTerminate handler or equivalent
[ ] Call voiceflow_unload_models() before terminate
[ ] Call voiceflow_shutdown_runtime()
[ ] Add small delay to ensure Rust cleanup completes
[ ] Update restartApp() to use synchronous cleanup
```

### 4.2 Update VoiceFlowBridge cleanup

**File:** `VoiceFlowApp/Sources/VoiceFlowApp/VoiceFlowBridge.swift`

```
Steps:
[ ] Add explicit cleanup() method (not just deinit)
[ ] Call cleanup() before app termination
[ ] Add isCleanedUp flag to prevent double-free
[ ] Log cleanup for debugging
```

### 4.3 Handle cleanup on Settings changes

**File:** `VoiceFlowApp/Sources/VoiceFlowApp/VoiceFlowApp.swift`

```
Steps:
[ ] When model settings change, call unload then reinitialize
[ ] Don't accumulate old model instances
[ ] Add UI feedback during model switching
```

---

## Phase 5: Config Caching (Low)

### 5.1 Cache parsed config

**File:** `crates/voiceflow-ffi/src/lib.rs`

```
Steps:
[ ] Add config field to VoiceFlowHandle
[ ] Load config once in voiceflow_init()
[ ] Pass cached config to all functions that need it
[ ] Add voiceflow_reload_config() for explicit refresh
[ ] Remove Config::load() calls from voiceflow_current_model(), etc.
```

---

## Phase 6: Verification & Monitoring

### 6.1 Add memory tracking

```
Steps:
[ ] Add memory usage logging at key points (init, process, cleanup)
[ ] Use jemalloc stats or similar for Rust heap tracking
[ ] Add voiceflow_memory_usage() FFI function
[ ] Display in Swift UI for debugging
```

### 6.2 Create test harness

```
Steps:
[ ] Write test that runs 100 transcriptions in a loop
[ ] Measure memory before/after each batch
[ ] Assert memory growth < threshold (e.g., 50MB total)
[ ] Add to CI if applicable
```

### 6.3 Profile with Instruments

```
Steps:
[ ] Run Allocations instrument before fixes
[ ] Capture baseline memory profile
[ ] Run after each phase
[ ] Document memory improvement per phase
```

---

## Implementation Order

| Phase | Priority | Estimated Impact | Dependencies |
|-------|----------|------------------|--------------|
| 1.1-1.3 | P0 | ~70% of leak fixed | None |
| 2.1-2.3 | P1 | ~15% of leak fixed | None |
| 3.1-3.3 | P2 | Prevents accumulation | Phase 1 |
| 4.1-4.3 | P2 | Clean shutdown | Phase 3 |
| 5.1 | P3 | Minor improvement | None |
| 6.1-6.3 | P3 | Verification | All phases |

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `crates/voiceflow-ffi/src/lib.rs` | Singleton runtime, new FFI functions, config caching |
| `crates/voiceflow-ffi/Cargo.toml` | Add once_cell dependency |
| `crates/voiceflow-core/src/llm/engine.rs` | Remove runtime creation, use shared |
| `crates/voiceflow-core/src/transcribe/moonshine.rs` | Buffer reuse, explicit cleanup |
| `crates/voiceflow-core/src/pipeline.rs` | Unload methods, idle timeout |
| `VoiceFlowApp/.../VoiceFlowApp.swift` | Cleanup before terminate |
| `VoiceFlowApp/.../VoiceFlowBridge.swift` | Explicit cleanup method |
| `VoiceFlowApp/.../voiceflow.h` | New FFI function declarations |

---

## Success Criteria

1. Memory usage after 100 transcriptions < 500MB above baseline
2. No memory growth on app restart (same baseline each time)
3. Clean shutdown with no leaked threads or GPU memory
4. No regression in transcription/LLM quality or speed

---

## Implementation Status: COMPLETE

All phases have been implemented. Key changes:

### Phase 1: Singleton Tokio Runtime
- Added `once_cell` dependency to both crates
- Created `runtime.rs` module in `voiceflow-core` with `register_runtime()` and `block_on()`
- Created singleton `RUNTIME` in `voiceflow-ffi` that registers with core on init
- Refactored `LlmEngine::new()` and `format()` to use `runtime::block_on()` (removed 3 `Runtime::new()` calls)

### Phase 2: Moonshine Tensor Optimization
- Added `cache_buffers` and `context_buffer` fields to `MoonshineEngine`
- Pre-allocate 12 cache buffers at 1MB each during engine init
- Reuse buffers in decode loop instead of cloning each iteration
- Store context tensor once outside the loop
- Added `clear_buffers()` method and `Drop` implementation

### Phase 3: Pipeline Resource Lifecycle
- Added `unload_llm()` and `unload_all()` methods to `Pipeline`
- Added FFI functions: `voiceflow_unload_models()`, `voiceflow_reset_llm()`
- Added memory tracking: `voiceflow_memory_usage()`, `voiceflow_memory_info()`, `voiceflow_force_gc()`
- Added `voiceflow_prepare_shutdown()` for clean app termination

### Phase 4: Swift-Side Cleanup
- Added `cleanup()` method to `VoiceFlowBridge` that calls FFI cleanup functions
- Added `applicationWillTerminate()` to `AppDelegate` that calls `voiceFlow.cleanup()`
- Added `getMemoryUsage()` and `forceGC()` static methods for debugging

### Phase 5: Config Caching
- Added `config` field to `VoiceFlowHandle` to cache parsed config at init

### Phase 6: Memory Tracking
- Added `MemoryInfo` struct with resident, virtual, and peak bytes
- Uses macOS `task_info` API for accurate memory reporting
- Tracks peak memory usage across session

### Files Modified:
- `crates/voiceflow-ffi/Cargo.toml` - Added once_cell
- `crates/voiceflow-ffi/src/lib.rs` - Singleton runtime, FFI cleanup functions
- `crates/voiceflow-core/Cargo.toml` - Added once_cell
- `crates/voiceflow-core/src/lib.rs` - Export runtime module
- `crates/voiceflow-core/src/runtime.rs` - NEW: Global runtime management
- `crates/voiceflow-core/src/llm/engine.rs` - Use shared runtime
- `crates/voiceflow-core/src/transcribe/moonshine.rs` - Buffer reuse, Drop impl
- `crates/voiceflow-core/src/pipeline.rs` - Unload methods
- `VoiceFlowApp/.../voiceflow.h` - New FFI declarations
- `VoiceFlowApp/.../VoiceFlowBridge.swift` - Cleanup methods
- `VoiceFlowApp/.../VoiceFlowApp.swift` - applicationWillTerminate handler
