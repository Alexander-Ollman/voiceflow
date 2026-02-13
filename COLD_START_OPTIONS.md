# VoiceFlow Cold Start Strategies

## Current State

### Architecture Overview
VoiceFlow currently uses a **mixed approach** for model loading:

| Component | Loading Strategy | Init Time | When It Happens |
|-----------|------------------|-----------|-----------------|
| Tokio Runtime | Eager (singleton) | ~50ms | App launch |
| Config | Eager | ~10ms | App launch |
| **STT Engine** | **Eager** | **500-2000ms** | App launch **(bottleneck)** |
| **LLM Engine** | Lazy | **2000-5000ms** | First transcription |
| Cache buffers | Eager (within STT) | ~100ms | App launch |
| UI/Bridge | Eager | ~100ms | App launch |

### Current Timing Breakdown

```
App Launch (t=0) → Background initialization task starts
  ├─ Create Tokio runtime:     50ms
  ├─ Load config:               10ms
  ├─ Create Pipeline:
  │   └─ Load STT Engine:     500-2000ms  ← BLOCKS APP READINESS
  │       ├─ Whisper or
  │       └─ Moonshine (4 ONNX files)
  └─ Bridge ready:             ~560-2160ms

User presses ⌥+Space (first use)
  ├─ Start recording
  ├─ Check if LLM loaded → NO (first use)
  └─ Load LLM:             2000-5000ms  ← BLOCKS FIRST TRANSCRIPTION

First transcription complete:      3000-7000ms from app launch
```

**Problems:**
1. STT blocks app readiness - users can't "launch and wait"
2. LLM blocks first transcription - bad experience for immediate use
3. Models reload every launch - no persistence between sessions
4. No progress feedback - users see frozen UI

---

## Cold Start Strategies

### Strategy 1: Fully Lazy Loading (Eager → Lazy)

**Concept:** Defer ALL model loading until first use, making app launch nearly instant.

**Implementation:**
```rust
// Before (current)
pub fn new(config: &Config) -> Result<Self> {
    let stt = SttEngine::new(config)?; // Blocks app
    let pipeline = Pipeline { stt, llm: None, config };
    Ok(pipeline)
}

// After (lazy)
pub fn new(config: &Config) -> Result<Self> {
    let pipeline = Pipeline {
        stt: None,  // Not loaded yet
        llm: None,
        config,
    };
    Ok(pipeline)
}

pub fn get_stt(&mut self) -> Result<&mut SttEngine> {
    if self.stt.is_none() {
        tracing::info!("Loading STT engine on-demand");
        self.stt = Some(SttEngine::new(&self.config)?);
    }
    Ok(self.stt.as_mut().unwrap())
}
```

**Flow:**
```
App Launch (t=0) → 100ms → Ready
User presses ⌥+Space
  ├─ "Loading models..." overlay appears
  ├─ Start loading STT: 500-2000ms
  ├─ Start audio capture
  └─ Process audio
```

**Pros:**
- Fastest app launch (~100ms)
- Simplifies code (both engines use same pattern)
- No background complexity
- Low memory footprint until needed

**Cons:**
- First transcription still blocked by model loading
- Worse UX for immediate use (no gradual warm-up)
- No benefit for back-to-back uses (still same initial delay)

**Estimated Impact:**
- App launch: 2100ms → 100ms (95% faster)
- First transcription: Still 3000-7000ms (no change)

---

### Strategy 2: Background Warm-up After Launch (Delayed Eager)

**Concept:** Launch app fast, then start loading models in background after short delay.

**Implementation:**
```swift
// VoiceFlowApp.swift
func initialize() async {
    let newHandle = voiceflow_init(nil)
    self.handle = newHandle
    self.isInitialized = true
    
    // Start warm-up after 5-second delay
    Task {
        try? await Task.sleep(nanoseconds: 5_000_000_000)
        if handle != nil {
            await warmUpModels()
        }
    }
}

func warmUpModels() async {
    // Start loading both engines in background
    voiceflow_start_warmup(handle) { progress in
        // Update UI with progress
    }
}
```

```rust
// voiceflow-ffi/src/lib.rs
pub unsafe extern "C" fn voiceflow_start_warmup(
    handle: *mut VoiceFlowHandle,
    callback: extern "C" fn(*const c_char)
) -> bool {
    let pipeline = &mut (*handle).pipeline;
    
    // Spawn background warm-up task
    tokio::spawn(async move {
        // Warm STT
        if let Err(e) = pipeline.warm_up_stt().await {
            tracing::error!("STT warm-up failed: {}", e);
        }
        
        callback(c"STT warm-up complete".as_ptr());
        
        // Warm LLM
        if let Err(e) = pipeline.warm_up_llm().await {
            tracing::error!("LLM warm-up failed: {}", e);
        }
        
        callback(c"LLM warm-up complete".as_ptr());
    });
    
    true
}
```

**Flow:**
```
App Launch (t=0) → 100ms → Ready
[BEGIN background warm-up after 5s_delay]
  └─ Warm-up: 2500-7000ms total
      ├─ STT: 500-2000ms (t=5000-7000)
      └─ LLM: 2000-5000ms (t=5500-7500)

User presses ⌥+Space at t=3000
  ├─ Warm-up in progress (maybe STT done, LLM loading)
  └─ Block on remaining load: ~2000-4500ms

User presses ⌥+Space at t=8000
  ├─ Both models ready
  └─ Process immediately
```

**Pros:**
- Fast app launch
- Automatic warm-up reduces delay for patient users
- Can be cancelled if user quits early
- Configurable delay (learn from usage patterns)

**Cons:**
- Wastes resources if user quits before warm-up
- Fixed delay doesn't adapt to user behavior
- Still blocks first use if user is quick

**Estimated Impact:**
- App launch: 2100ms → 100ms
- First transcription (patient): 0ms (if wait >7s)
- First transcription (quick): 1500-3000ms (partial warm-up)

---

### Strategy 3: Activity-Based Warm-up (Predictive)

**Concept:** Warm up models when user is likely to use them, based on activity patterns.

**Implementation:**
```swift
class ActivityMonitor {
    var lastActivity: Date
    var isIdle: Bool { Date().timeIntervalSince(lastActivity) > 300 }
    
    func startMonitoring() {
        let tracker = NSWorkspace.shared.notificationCenter
        tracker.addObserver(forName: NSWorkspace.screensDidWakeNotification, ...)
        tracker.addObserver(forName: NSWorkspace.didActivateApplicationNotification, ...)
        
        // Periodically check idle time
        Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { _ in
            if self.isIdle && !modelsWarmedUp {
                self.warmUpModels()
            }
        }
    }
}
```

```rust
// Learn usage patterns
struct UsagePattern {
    let hour: u8
    let dayOfWeek: u8
    let minutesSinceLaunch: u16
    
    fn predict_next_use(&self) -> Seconds {
        // Use simple time-based prediction
        // Or implement ML model for better prediction
    }
}

fn decide_when_to_warmup(pattern: &UsagePattern) -> Duration {
    if pattern.in_predicted_window() {
        Duration::from_secs(0) // Warm now
    } else if pattern.is_idle() {
        Duration::from_secs(60) // Give it a minute
    } else {
        Duration::from_secs(300) // Default 5 min
    }
}
```

**Flow:**
```
App Launch (t=0) → 100ms → Ready
[Monitor activity]

User opens TextEdit at t=120 → High likelihood of typing
  └─ Trigger warm-up immediately

User goes idle at t=450 → 5 minutes of no activity
  └─ Trigger warm-up

Morning routine detected (9:00 AM every day)
  └─ Pre-warm at 8:55 AM

User presses ⌥+Space at unknown time
  └─ If not warmed: block on load
  └─ If warmed: immediate response
```

**Pros:**
- Adapts to user behavior
- Minimizes wasted resources (only warms when needed)
- Can predictively warm before user even thinks to use it
- Improves over time as it learns patterns

**Cons:**
- Complex implementation (monitoring, prediction, learning)
- May miss unpredictable usage
- Requires privacy considerations (activity tracking)
- Initial learning period with poor predictions

**Estimated Impact:**
- App launch: 2100ms → 100ms
- First transcription (predicted): 0ms
- First transcription (unpredicted): 2500-7000ms (no warm-up)

---

### Strategy 4: Progressive Loading (Streaming Init)

**Concept:** Load models in stages, allowing partial functionality early.

**Implementation:**
```rust
pub struct SttEngine {
    preprocess: Option<Preprocessor>,  // Load first (~100ms)
    encoder: Option<Encoder>,          // Load second (~300ms)
    decoder: Option<Decoder>,          // Load third (~1000ms)
}

impl SttEngine {
    async fn load_progressive() -> Result<Self> {
        let mut engine = SttEngine { preprocess: None, encoder: None, decoder: None };
        
        // Stage 1: Preprocessing (needed for audio capture)
        engine.preprocess = Some(Preprocessor::new()?);
        callback(LoadingStage::Preprocess, 0.2);  // 20% complete
        
        // Stage 2: Encoder (can transcribe first tokens)
        engine.encoder = Some(Encoder::new()?);
        callback(LoadingStage::Encoder, 0.5);  // 50% complete
        
        // Stage 3: Decoder (full transcription)
        engine.decoder = Some(Decoder::new()?);
        callback(LoadingStage::Complete, 1.0);  // 100% complete
        
        Ok(engine)
    }
}

// Can start transcription with partial model
pub fn can_start_transcript(&self) -> bool {
    self.preprocess.is_some() && self.encoder.is_some()
}

pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
    // Use available stages, show "processing..." if decoder not ready
    match &self.decoder {
        Some(decoder) => {
            let encoded = self.encoder.as_ref().unwrap().encode(audio)?;
            decoder.decode(encoded)
        }
        None => {
            // Show "Loading decoder..."
            Err(Error::ModelNotReady)
        }
    }
}
```

**Flow:**
```
App Launch (t=0) → 100ms → Ready
User presses ⌥+Space at t=200
  ├─ Start model loading (progressive)
  ├─ Preprocess ready at t=300 (100ms)
  │   └─ Start capturing audio
  ├─ Encoder ready at t=600 (300ms)
  │   └─ Can begin encoding while decoder loads
  ├─ Decoder ready at t=1600 (1000ms)
  │   └─ Full transcription available
  └─ Paste result at t=1600

Progress shown: "Loading... 20%" → "50%" → "100%"
```

**Pros:**
- Can start working earlier (audio capture, preprocessing)
- Visual progress feedback improves perceived speed
- Gracious degradation (partial functionality until full load)
- Overlaps model load with other operations

**Cons:**
- Most complexity of all strategies
- Need to handle "not ready" states gracefully
- May confuse users if behavior changes during load
- Limited benefit if decoder is last stage (can't complete without it)

**Estimated Impact:**
- App launch: 2100ms → 100ms
- First transcription: 1400ms (partial overlap)
  - Start audio: 100ms after key press
  - Finish: 300ms earlier than full block
- Perceived latency: Better due to progress feedback

---

### Strategy 5: Session Persistence (Warm Restart)

**Concept:** Save model state between app launches and resume from cache.

**Implementation:**
```rust
pub struct ModelPersistence {
    cache_dir: PathBuf,
    session_id: String,
}

impl ModelPersistence {
    fn save_session(&self, engine: &Engine) -> Result<()> {
        // Serialize model weights and state
        let state = engine.serialize()?;
        let path = self.cache_dir.join(format!("{}.cache", self.session_id));
        std::fs::write(path, state)?;
        Ok(())
    }
    
    fn load_session(&self, session_id: &str) -> Result<Option<Vec<u8>>> {
        let path = self.cache_dir.join(format!("{}.cache", session_id));
        if path.exists() {
            Ok(Some(std::fs::read(path)?))
        } else {
            Ok(None)
        }
    }
    
    fn validate_cache(&self, cached: &[u8], model_file: &Path) -> bool {
        // Verify cache matches current model version
        let hash = compute_model_hash(model_file)?;
        let stored_hash = extract_hash_from_cache(cached)?;
        hash == stored_hash
    }
}

// Use memory-mapped files for zero-copy loading
pub fn load_model_with_cache(path: &Path) -> Result<Arc<Mmap>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };  // Fast, no copy
    Ok(Arc::new(mmap))
}
```

**Flow:**
```
First App Launch
  └─ Load models from disk: 2500-7000ms
  └─ Save to cache: ~100ms

User closes app (session saved)

Next App Launch
  └─ Check cache: ~10ms
  └─ Validate cache matches model: ~5ms
  └─ Load from MMAP (partial): 100-500ms
  └─ Ready in: 115-515ms

Subsequent uses: Instant (already in memory)
```

**Pros:**
- Massive speedup for repeated launches
- Uses memory mapping (no duplicate copies)
- Transparent to user (automatic)
- Works well for users who frequently close/reopen

**Cons:**
- Uses extra disk space (~2-4GB for all models)
- Cache invalidation on model updates (maintenance overhead)
- First launch still slow (need to build cache)
- Security concerns (persisting model weights)

**Estimated Impact:**
- First launch: 3000-7000ms
- Subsequent launches: 115-515ms (90-98% faster)
- First transcription: Instant (models already loaded)

---

### Strategy 6: Hybrid Approach (Recommended Combination)

**Concept:** Combine multiple strategies for optimal behavior.

**Recommended Hybrid:**

1. **Fully lazy load** at startup (Strategy 1) for fast launch
2. **Background warm-up after 5s** (Strategy 2) for automatic pre-load
3. **Simple activity-based trigger** (Strategy 3 lite, no ML)
4. **Progressive loading** (Strategy 4) for STT only (simple win)
5. **Optional session persistence** (Strategy 5, for power users)

**Implementation:**

```rust
pub struct Pipeline {
    stt: Option<LazySttEngine>,  // Lazy init
    llm: Option<LazyLlmEngine>,  // Lazy init
    config: Config,
    persistence: Option<ModelPersistence>,
    warmup_token: Option<CancellationToken>,
}

impl Pipeline {
    // Startup: NO models loaded
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Pipeline {
            stt: None,
            llm: None,
            config: config.clone(),
            persistence: config.persist().then(load_persistence),
            warmup_token: None,
        })
    }
    
    // Background warm-up (start after user delay or activity)
    pub fn start_warmpup(&mut self) -> CancellationToken {
        let token = CancellationToken::new();
        self.warmup_token = Some(token.clone());
        
        tokio::spawn(async move {
            // Try persistent cache first
            if let Some(session) = persistence.load_cache().await? {
                return;  // Already warm
            }
            
            // Progressive STT load
            stt.load_progressive(token.clone()).await?;
            
            // Full LLM load
            llm.load(token.clone()).await?;
        });
        
        token
    }
    
    // User-initiated use
    pub fn get_stt(&mut self) -> Result<&mut SttEngine> {
        if self.stt.is_none() {
            // If warm-up started, wait for it
            // Otherwise, load now
            self.stt = Some(SttEngine::load_progressive()?);
        }
        Ok(self.stt.as_mut().unwrap())
    }
}

// Swift integration
class VoiceFlowBridge {
    func initialize() async {
        let newHandle = voiceflow_init(nil)  // Fast, no models
        self.handle = newHandle
        
        // Start warm-up after 3s (shorter delay + activity trigger)
        Task {
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            self.warmupToken = self.startWarmup()
        }
    }
    
    // Activity-based trigger (simple: user switches to text editor)
    func onAppFocus(appName: String) {
        if isTextEditor(appName) && shouldWarmup() {
            // Cancel pending warm-up, start now
            warmupToken?.cancel()
            voiceflow_warmup_now(handle)
        }
    }
    
    // Recording start
    func startRecording() {
        // Check if models ready
        if !voiceflow_models_ready(handle) {
            showLoadingOverlay()
        }
        // Start audio capture regardless
    }
}
```

**Flow:**

```
App Launch (t=0)
  └─ Launch: 100ms (no models)

[Background warm-up starts at t=3s]
  └─ Load STT progressively: 600-1500ms (t=3600-4500)
  └─ Load LLM fully: 2000-5000ms (t=5000-8000)

Scenario A: User is patient
  User presses ⌥+Space at t=10000
  ├─ Models already warm
  └─ Transcribe immediately: 200ms

Scenario B: User switches to TextEdit at t=500
  ├─ Detect text editor → Trigger immediate warm-up
  ├─ Models warm: 2500-7000ms (t=3000-7500)
  └─ User starts recording at t=8000: Ready immediately

Scenario C: User is quick
  User presses ⌥+Space at t=1500
  ├─ Models not ready
  ├─ Show "Loading models... 20%" → "100%"
  ├─ Progressive load: 600-1500ms
  └─ Complete: 2100-3000ms (still faster than full block)

Scenario D: User uses persistence
  App quit and cached
  Next app launch: 115-515ms to be ready
  ⌥+Space use: Instant
```

**Pros:**
- Fastest app launch among all options
- Automatic warm-up for most users
- Activity aware (triggers on text editor use)
- Graceful degradation with progressive loading
- Optional persistence for power users
- Adaptive behavior (learns user patterns over time)

**Cons:**
- Most complex implementation
- More code to maintain
- Slightly higher memory usage (optional warm-up)
- Requires careful cancellation handling

**Estimated Impact:**

| Scenario | Time | Improvement |
|----------|------|-------------|
| App Launch | 100ms | 95% faster |
| First use (patient, >10s) | 200ms | 90% faster |
| First use (quick, text editor) | ~500ms | 80% faster |
| First use (quick, no warm-up) | 2100-3000ms | 50% faster |
| Subsequent uses (same session) | 200ms | Best case |
| Next app launch (with persistence) | 115-515ms | 95% faster |

---

## Recommended Strategy: Hybrid Approach

### Summary

I recommend **Strategy 6 (Hybrid)** as it provides the best balance of speed, user experience, and maintainability.

### Why Hybrid Wins

1. **Fastest launch**: 100ms vs 2100ms (current) - matches Strategy 1
2. **Automatic warm-up**: Users who wait 3s get instant response - matches Strategy 2
3. **Activity aware**: Triggers on text editor focus - better than fixed delay
4. **Progressive loading**: Can capture audio while models load - better UX
5. **Opt-in persistence**: Power users get 5x faster re-launch -Strategy 5

### Implementation Plan (Phased)

**Phase 1: Lazy Loading** (Week 1)
- Move STT to lazy initialization
- App launches in 100ms
- Test that audio capture starts even without models

**Phase 2: Background Warm-up** (Week 2)
- Add 3-second delayed warm-up
- Progress callbacks to Swift
- Cancellation support

**Phase 3: Progressive STT** (Week 3)
- Implement staged loading for SttEngine
- Show progress in overlay
- Allow audio capture during load

**Phase 4: Activity Trigger** (Week 4)
- Detect text editor app switches
- Immediate warm-up on focus change
- Simple heuristic (no ML needed initially)

**Phase 5: Optional Persistence** (Week 5-6)
- Add session caching with MMAP
- Cache validation
- User preference toggle

**Phase 6: Learning & Tuning** (Week 7)
- Log warm-up effectiveness
- Adjust delays based on actual usage
- A/B test different warm-up timings

### Configuration Options

```toml
[coldstart]
# When to start background warm-up (seconds)
warmup_delay = 3

# Enable activity-based triggers (text editor detection)
activity_trigger = true

# Enable session persistence (cache between launches)
persistence.enabled = true
persistence.max_size_mb = 4096

# Progressive loading preferences
stt.progressive = true
llm.progressive = false  # LLM is slower to load, less beneficial
```

### Fallback Behavior

Always have a safe fallback:

```rust
pub fn ensure_models_ready(&mut self) -> Result<()> {
    // Cancel any pending warm-up
    self.cancel_warmup();
    
    // Force load now (blocking)
    self.get_stt()?;
    self.get_llm()?;
    
    Ok(())
}
```

---

## Comparison Matrix

| Strategy | Launch Speed | First Use Speed | Complexity | Memory | Disk | User Experience |
|----------|--------------|-----------------|------------|--------|------|-----------------|
| **Current** | 2100ms | 3000-7000ms | Low | 1GB | 0 | Poor (long blocks) |
| **1. Lazy** | 100ms | 3000-7000ms | Low | 1GB | 0 | Poor (slower first use) |
| **2. Delayed** | 100ms | 0-3000ms | Medium | 1GB | 0 | Good (adapts) |
| **3. Predictive** | 100ms | 0-7000ms | High | 1GB | 0 | Great (when works) |
| **4. Progressive** | 100ms | 1400-3000ms | High | 1GB | 0 | Good (visible progress) |
| **5. Persistence** | 115-515ms | Instant | Medium | 2GB | 4GB | Excellent (after first) |
| **6. Hybrid (Recommended)** | 100ms | 200-3000ms | High | 1-2GB | 0-4GB | Excellent (adaptive) |

### Key Metrics

| Metric | Current | With Hybrid |
|--------|---------|-------------|
| App Launch | 2100ms | 100ms |
| First Use (patient) | 3000ms | 200ms |
| First Use (quick) | 5000ms | 2100ms |
| Re-launch | 3000ms | 115ms |
| Avg First Use | 5000ms | ~800ms |

**Overall improvement**: 84% faster average cold start experience

---

## Implementation Notes

### Rust Changes

1. **Pipeline struct**: Add lazy fields, warm-up token
2. **SttEngine**: Add `load_progressive()` method
3. **LlmEngine**: Add `load()` with cancellation support
4. **Persistence module**: For MMAP caching
5. **Progress callbacks**: Expose via FFI

### Swift Changes

1. **Initialization**: Remove blocking wait, start background warm-up
2. **Activity monitor**: Detect text editor switches
3. **Loading overlay**: Show progress during first use
4. **Bridge methods**: `voiceflow_start_warmup`, `voiceflow_cancel_warmup`
5. **Settings UI**: Add cold start preferences

### Testing Strategy

1. **Cold start benchmarks**: Measure all scenarios
2. **Activity detection**: Verify triggers work
3. **Memory profiling**: Ensure no leaks
4. **User testing**: UX validation of progress indicators
5. **Regression tests**: Maintain transcription quality

---

## Conclusion

The **Hybrid Strategy** provides the best用户体验by:
- Launching instantly (100ms)
- Adapting to user behavior (3s warm-up + activity triggers)
- Showing progress (progressive loading)
- Caching for power users (optional persistence)
- Failing gracefully (force load if needed)

This approach reduces cold start latency from **3000-7000ms to 200-3000ms depending on timing**, with an **84% average improvement** in the user experience.