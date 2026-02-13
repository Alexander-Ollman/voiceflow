# VoiceFlow Performance Improvements - Scope of Work

## Overview
Reduce transcription latency and improve overall performance for faster, more responsive voice-to-text experience.

**Baseline**: ~1.5 second total latency (Whisper + LLM)

**Goal**: Sub-500ms response time for optimal user experience

---

## Phase 1: Streaming Transcription (High Impact)

### 1.1 Real-time Transcription During Recording
- Implement streaming Whisper transcribe as audio is being captured
- Display live text in a floating window while user holds Option+Space
- Update text dynamically as more speech is recognized
- Allow user to stop recording early if transcription appears correct

**Files**: `crates/voiceflow-core/src/transcribe/mod.rs`, `VoiceFlowApp/Sources/VoiceFlowApp/RecordingOverlay.swift`
**Est. Impact**: Reduce perceived latency by ~800ms (user sees progress immediately)
**Priority**: P0

### 1.2 Incremental LLM Formatting
- Apply LLM formatting to partial transcription results
- Update formatted text in real-time without waiting for full release
- Debounce formatting updates (e.g., every 500ms or after pause detection)
- Maintain consistency of final formatted text

**Files**: `crates/voiceflow-core/src/pipeline.rs`, `crates/voiceflow-core/src/llm/engine.rs`
**Est. Impact**: Final formatting completes 300-500ms faster
**Priority**: P1

### 1.3 Live Waveform Visualization
- Display real-time audio waveform in overlay window
- Visual feedback for voice activity detection
- Helps users adjust speaking distance/volume
- Minimal performance overhead using Metal drawing

**Files**: `VoiceFlowApp/Sources/VoiceFlowApp/WaveformView.swift`
**Est. Impact**: Better UX, improves perceived responsiveness
**Priority**: P2

---

## Phase 2: LLM Optimization (High Impact)

### 2.1 LLM Prompt Caching
- Implement KV cache for LLM prompts (for models that support it)
- Cache common prompt patterns (punctuation, bullet points, basic formatting)
- Reuse cached context when consecutive transcriptions are similar
- Configure cache size limits (default: 5-10 cached prompts)

**Files**: `crates/voiceflow-core/src/llm/engine.rs`, `crates/voiceflow-core/src/llm/mistralrs_backend.rs`
**Est. Impact**: 30-40% faster LLM inference for repeated patterns
**Priority**: P0

### 2.2 Quantized LLM Models
- Add 4-bit quantized versions of Qwen3 model
- Implement model selection with quality/speed trade-off
- Auto-select based on input length (simple = quantized, complex = full precision)
- Maintain accuracy benchmarks for quantized models

**Files**: `crates/voiceflow-core/src/config.rs`, `crates/voiceflow-core/src/llm/mod.rs`, `models/`
**Est. Impact**: 50-60% faster LLM inference, 70% memory reduction
**Priority**: P0

### 2.3 Faster LLM Models
- Evaluate and integrate SmolLM3-135M for simple formatting tasks
- Use lightweight model for basic punctuation/formatting
- Fallback to Qwen3 only for complex formatting (lists, quotes, etc.)
- Auto-detect complexity and route to appropriate model

**Files**: `crates/voiceflow-core/src/llm/mod.rs`, `crates/voiceflow-core/src/llm/backend.rs`
**Est. Impact**: 2-3x faster for simple transcriptions
**Priority**: P1

### 2.4 Context-Aware Prompt Optimization
- Shorten LLM prompts based on detected content (no list cues = remove list instructions)
- Generate prompts dynamically based on transcription features (caps, numbers, punctuation)
- Remove unused prompt sections to reduce token count
- Validate formatting quality vs. prompt length

**Files**: `crates/voiceflow-core/src/llm/prompts.rs`, `crates/voiceflow-core/src/context/mod.rs`
**Est. Impact**: 10-20% faster LLM inference
**Priority**: P2

---

## Phase 3: Cold Start Improvements (Medium Impact)

### 3.1 Lazy Model Loading
- Defer model loading until first use
- Show "initializing" status indicator to user
- Pre-load models on app launch (configurable: immediate, on-demand, scheduled)
- Cache model state between app sessions

**Files**: `crates/voiceflow-core/src/pipeline.rs`, `VoiceFlowApp/Sources/VoiceFlowApp/VoiceFlowBridge.swift`
**Est. Impact**: App launch 2-3x faster, first transcription unchanged
**Priority**: P0

### 3.2 Model Warm-up on Background
- Idle warm-up: Initialize models when computer has been idle > 5min
- Scheduled warm-up: Pre-load at predictable usage times (learn user patterns)
- User-initiated warm-up: Settings option to "warm up models now"
- Cancel warm-up if user interrupts

**Files**: `crates/voiceflow-core/src/pipeline.rs`, `VoiceFlowApp/Sources/VoiceFlowApp/VoiceFlowApp.swift`
**Est. Impact**: First transcription after idle < 100ms
**Priority**: P1

### 3.3 Reduced Model Variants for Quick Start
- Load "fast" model variants first (small quantized models)
- Upgrade to full models on second transcription if configured
- Background replacement: Swap in better models after warm-up
- No performance regression for fast-only mode

**Files**: `crates/voiceflow-core/src/config.rs`, `crates/voiceflow-core/src/pipeline.rs`
**Est. Impact**: First transcription 40-50% faster, seamless upgrade later
**Priority**: P2

---

## Phase 4: Audio Processing Optimization (Medium Impact)

### 4.1 Voice Activity Detection (VAD) Speedup
- Optimize pitch-detection algorithm (currently ~50ms overhead)
- Use Metal-accelerated VAD for faster processing
- Reduce VAD buffer sizes for faster response
- False positive tuning to minimize re-recording

**Files**: `crates/voiceflow-core/src/audio/mod.rs`, `crates/voiceflow-core/src/context/detector.rs`
**Est. Impact**: 30-50ms faster per transcription
**Priority**: P1

### 4.2 Parallel Audio Processing Pipeline
- Overlap audio capture, VAD, and pre-processing
- Use tokio tasks for non-blocking audio pipeline stages
- Pipeline: Capture → VAD → Resample → Format → Transcript
- Reduce end-to-end latency by 50-100ms

**Files**: `crates/voiceflow-core/src/audio/mod.rs`, `crates/voiceflow-core/src/pipeline.rs`
**Est. Impact**: 50-100ms faster total latency
**Priority**: P1

### 4.3 GPU-Accelerated Audio Processing
- Implement Metal shaders for resampling and filtering
- Move audio normalization to GPU
- Batch audio chunks for GPU processing
- Fallback to CPU for non-Metal systems

**Files**: `crates/voiceflow-core/src/audio/mod.rs`, `VoiceFlowApp/Sources/VoiceFlowApp/MetalAudioProcessor.swift`
**Est. Impact**: 20-30ms faster, lower CPU usage
**Priority**: P2

---

## Phase 5: Model Inference Optimizations (Medium Impact)

### 5.1 Whisper Model Optimization
- Switch to faster Whisper variant (tiny-base with same accuracy)
- Optimize ONNX runtime configuration for Whisper
- Enable ONNX Runtime session options: graph optimization, parallel execution
- Pre-compile ONNX models for faster runtime initialization

**Files**: `crates/voiceflow-core/src/transcribe/mod.rs`, `crates/voiceflow-core/src/transcribe/whisper.rs`
**Est. Impact**: 100-200ms faster transcription
**Priority**: P0

### 5.2 Batch Processing for Multiple Transcriptions
- Process queued transcriptions in parallel when possible
- Batch small transcriptions for single LLM pass with separator
- Reuse LLM session across multiple formatting requests
- Configurable batch size (default: disabled for immediate feedback)

**Files**: `crates/voiceflow-core/src/pipeline.rs`, `crates/voiceflow-core/src/llm/engine.rs`
**Est. Impact**: 20-30% faster for back-to-back transcriptions
**Priority**: P1

### 5.3 Moonshine ONNX Optimization
- Optimize ONNX Runtime session for Moonshine model
- Enable custom execution providers and optimization levels
- Pre-allocate decoder cache (already partially done)
- Profile and optimize hot paths in decode loop

**Files**: `crates/voiceflow-core/src/transcribe/moonshine.rs`
**Est. Impact**: 50-100ms faster
**Priority**: P2

---

## Phase 6: Memory & Resource Optimization (Low Impact)

### 6.1 Memory Pool for Tensors
- Pre-allocate memory pool for model tensors
- Reuse tensor buffers across inference calls
- Reduce allocation overhead and fragmentation
- Configurable pool size based on available RAM

**Files**: `crates/voiceflow-core/src/transcribe/mod.rs`, `crates/voiceflow-core/src/llm/engine.rs`
**Est. Impact**: 10-20ms faster, lower memory pressure
**Priority**: P2

### 6.2 LLM Session Reuse
- Keep LLM session alive between transcriptions
- Implement idle timeout with configurable keep-alive
- Warm session on app launch if user likely to transcribe soon
- Graceful fallback on session failure

**Files**: `crates/voiceflow-core/src/llm/engine.rs`
**Est. Impact**: 30-50ms faster for consecutive uses
**Priority**: P2

### 6.3 Optimized Configuration Caching
- Cache compiled configuration and model metadata
- Validate cache consistency on load
- Invalidate cache on model updates
- Reduce config parse overhead

**Files**: `crates/voiceflow-core/src/config.rs`, `crates/voiceflow-ffi/src/lib.rs`
**Est. Impact**: <10ms, minor but trivial to implement
**Priority**: P3

---

## Implementation Order

| Phase | Priority | Est. Latency Reduction | Est. Effort |
|-------|----------|------------------------|-------------|
| 1.1 | P0 | 800ms (perceived) | 3 weeks |
| 1.2 | P1 | 300-500ms | 2 weeks |
| 2.1 | P0 | 150-200ms | 1 week |
| 2.2 | P0 | 200-300ms | 2 weeks |
| 2.3 | P1 | 400-600ms (simple) | 2 weeks |
| 2.4 | P2 | 50-100ms | 1 week |
| 3.1 | P0 | N/A (launch speed) | 1 week |
| 3.2 | P1 | ~400ms (first after idle) | 1 week |
| 3.3 | P2 | 200-300ms (first use) | 1 week |
| 4.1 | P1 | 30-50ms | 1 week |
| 4.2 | P1 | 50-100ms | 2 weeks |
| 4.3 | P2 | 20-30ms | 1.5 weeks |
| 5.1 | P0 | 100-200ms | 1 week |
| 5.2 | P1 | 50-100ms (batch) | 1 week |
| 5.3 | P2 | 50-100ms | 1 week |
| 6.1-6.3 | P2-P3 | 20-50ms | 1-2 weeks |

**Total Estimated Effort**: ~26 weeks (6 months)

---

## Success Metrics

### Primary Metrics
1. **P50 Latency**: < 500ms from key release to formatted text
2. **P90 Latency**: < 800ms from key release to formatted text
3. **App Launch**: < 1 second to ready state (no models loaded)
4. **First Transcription**: < 800ms after cold start

### Secondary Metrics
1. **Memory Usage**: Baseline < 300MB idle, < 1GB peak
2. **CPU Usage**: < 20% average during transcription
3. **GPU Usage**: < 50% sustained during inference
4. **Accuracy**: No regression in transcription/formatting quality (>95% of baseline)

### User Experience Metrics
1. **Perceived Latency**: User sees progress within 200ms of key press
2. **Error Rate**: < 1% transcription failures
3. **Reliability**: 99%+ success rate on consecutive uses

---

## Testing Plan

### Performance Benchmarks
1. Create benchmark suite with realistic audio samples (10s, 30s, 60s durations)
2. Measure latency at each pipeline stage (capture, transcribe, format)
3. Profile with Instruments before and after each phase
4. Track regression across commits

### Load Testing
1. 100 consecutive transcriptions to verify memory stability
2. Stress test with 5 transcriptions per second for 30 seconds
3. Test with varying audio quality (clear, noisy, background speech)
4. Validate thermal throttling behavior under sustained use

### A/B Testing
1. Compare streaming vs. non-streaming UX (user preference study)
2. A/B test quantized vs. full-precision LLM models
3. Survey users on perceived responsiveness

---

## Risk & Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Streaming accuracy lower than batch | Quality regression | Medium | Verify quality before release, offer fallback |
| Quantized models lose accuracy | User complaints | Medium | Quality benchmarks, auto-promote on errors |
| Increased memory usage | Performance degradation | Low | Memory profiling, size limits on cache |
| Complexity increases bugs | Maintenance overhead | Medium | Comprehensive testing, incremental rollout |
| Metal support issues | Platform incompatibility | Low | Fallback to CPU, test on target devices |

---

## Deliverables

1. **Streaming transcription** with live overlay UI
2. **Quantized LLM models** integrated with auto-selection
3. **Prompt caching** system with configurable limits
4. **Model warm-up system** with idle detection
5. **Optimized audio pipeline** with V2M support
6. **Performance benchmark suite** for regression testing
7. **Documentation**: performance tuning guide, architecture updates
8. **Migration guide** for existing users (if behavior changes)

---

## Notes

- Phases can be implemented in parallel where possible (e.g., 4.2 and 5.1)
- Some P2 items can be deferred to post-launch based on user feedback
- Consider feature flags for gradual rollout of risky changes
- Maintain backward compatibility for existing configurations