# Moonshine Timestamp Gap Solution Scope

## Problem Statement

Moonshine ONNX models don't provide word-level timestamps, which breaks pause-based prosody analysis. This limits punctuation detection to only voice commands and pitch analysis when using Moonshine.

## Current State

| Feature | Whisper | Moonshine |
|---------|---------|-----------|
| Transcription | Yes | Yes |
| Word Timestamps | Yes | No |
| Pause Analysis | Yes | No |
| Pitch Analysis | Yes | Yes |
| Voice Commands | Yes | Yes |
| Speed | ~800ms | ~150ms |

## Proposed Solutions

### Option 1: Energy-Based Pause Detection (Recommended)

Detect pauses directly from audio energy levels without relying on word timestamps.

**Approach**:
```rust
pub struct EnergyPauseDetector {
    window_size_ms: u32,      // 20-50ms analysis window
    hop_size_ms: u32,         // 10-25ms hop between windows
    silence_threshold: f32,    // RMS below this = silence
    min_pause_ms: u32,        // Minimum pause duration
}

impl EnergyPauseDetector {
    pub fn detect_pauses(&self, audio: &[f32]) -> Vec<PauseRegion> {
        // 1. Compute RMS energy per window
        // 2. Find regions below threshold
        // 3. Merge consecutive silent windows
        // 4. Filter by minimum duration
        // 5. Return pause regions with start/end times
    }
}
```

**Pros**:
- Works with any STT engine
- Simple to implement
- Low computational cost

**Cons**:
- Less precise than word-aligned timestamps
- May miss pauses in noisy audio
- Cannot distinguish between words

**Implementation Steps**:
1. Add `EnergyPauseDetector` to `prosody/`
2. Modify `analyze_prosody()` to use energy detection when timestamps unavailable
3. Map pause regions to approximate word boundaries

### Option 2: Forced Alignment Post-Processing

Use a separate forced alignment model to align transcript to audio.

**Approach**:
```rust
pub struct ForcedAligner {
    model: WhisperEngine,  // Use Whisper tiny just for alignment
}

impl ForcedAligner {
    pub fn align(&self, audio: &[f32], transcript: &str) -> Vec<WordTimestamp> {
        // Run Whisper with word timestamps enabled
        // Match against provided transcript
    }
}
```

**Pros**:
- Accurate word-level timestamps
- Can use smaller/faster Whisper model

**Cons**:
- Adds latency (~200-400ms)
- Requires loading second model
- Negates some of Moonshine's speed advantage

### Option 3: Hybrid Approach

Use energy detection for initial punctuation, refine with optional alignment.

**Approach**:
1. Primary: Energy-based pause detection (fast)
2. Optional: Forced alignment for high-confidence cases
3. User can toggle "high accuracy" mode

### Option 4: Accept the Trade-off

Document that Moonshine trades accuracy for speed, and pause analysis is unavailable.

**Approach**:
- Update UI to show "Pause analysis unavailable with Moonshine"
- Rely more heavily on LLM for punctuation decisions
- Enhance prompt with note about punctuation uncertainty

## Recommendation

**Implement Option 1 (Energy-Based)** as the primary solution:

1. Simple and fast
2. No additional model dependencies
3. Good enough for most use cases
4. Can be enhanced later with Option 2 if needed

## Implementation Plan

### Phase 1: Energy Pause Detector

```rust
// New file: crates/voiceflow-core/src/prosody/energy_pauses.rs

/// Detected pause region in audio
pub struct PauseRegion {
    pub start_ms: u64,
    pub end_ms: u64,
    pub duration_ms: u64,
}

/// Detect pauses using audio energy analysis
pub fn detect_pauses_from_energy(
    audio: &[f32],
    sample_rate: u32,
    config: &PauseDetectionConfig,
) -> Vec<PauseRegion> {
    // Implementation
}

/// Configuration for energy-based pause detection
pub struct PauseDetectionConfig {
    pub window_ms: u32,          // Default: 25
    pub hop_ms: u32,             // Default: 10
    pub silence_threshold: f32,  // Default: 0.01
    pub min_pause_ms: u32,       // Default: 150
    pub max_pause_ms: u32,       // Default: 2000
}
```

### Phase 2: Integration

Modify `prosody/mod.rs`:

```rust
pub fn analyze_prosody(
    audio: &[f32],
    word_timestamps: Option<&[(String, i64, i64)]>,
) -> ProsodyHints {
    let pause_hints = if let Some(timestamps) = word_timestamps {
        // Use word-aligned pause analysis (Whisper)
        analyze_pauses(timestamps)
    } else {
        // Fall back to energy-based detection (Moonshine)
        let regions = detect_pauses_from_energy(audio, 16000, &Default::default());
        convert_regions_to_hints(regions)
    };
    // ...
}
```

### Phase 3: UI Updates

- Add tooltip explaining accuracy difference
- Consider showing confidence indicator

## Files to Modify

1. `crates/voiceflow-core/src/prosody/energy_pauses.rs` (new)
2. `crates/voiceflow-core/src/prosody/mod.rs`
3. `VoiceFlowApp/Sources/.../SettingsView.swift` (tooltip)

## Estimated Effort

- Phase 1: 3-4 hours
- Phase 2: 1-2 hours
- Phase 3: 1 hour

## Success Metrics

- Pause detection accuracy within 80% of Whisper timestamps
- No measurable latency impact (<10ms)
- Works reliably across different speaking speeds
