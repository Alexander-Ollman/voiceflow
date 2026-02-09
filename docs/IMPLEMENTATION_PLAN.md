# VoiceFlow v2: Wispr Flow-Quality Local Dictation

## Executive Summary

This plan outlines how to achieve Wispr Flow-level dictation quality with **100% local inference** - no cloud, no subscription, no data leaving the device.

**Target metrics:**
- End-to-end latency: <500ms (vs Wispr's 700ms cloud)
- Accuracy: 95%+ on clean speech (matching Wispr)
- Memory footprint: <2GB total (vs Wispr's 800MB + cloud)
- Privacy: Complete - nothing leaves the device

---

## Research Findings Summary

### What Makes Wispr Flow Excellent

1. **Context-conditioned ASR** - STT model conditioned on screen context and user history
2. **Fine-tuned Llama** - <200ms inference for transcript cleanup
3. **Screenshot context** - Understands active app, adjusts formatting
4. **Learns from corrections** - RL policies trained on user edits
5. **Cloud GPU infrastructure** - TensorRT-LLM optimized, massive compute

### What We Can Replicate Locally

| Wispr Feature | Local Implementation | Feasibility |
|---------------|---------------------|-------------|
| Fast STT | Moonshine (already have) | ✅ Done |
| Context-conditioned ASR | Custom vocabulary per app | ✅ Achievable |
| LLM formatting | Fine-tuned SmolLM2/Gemma | ✅ Achievable |
| Screenshot context | Gemma 3n E2B (multimodal) | ✅ Achievable |
| Learn from corrections | Local LoRA fine-tuning | ⚠️ Advanced |
| <200ms LLM inference | MLX + quantization | ✅ Achievable |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VoiceFlow v2 Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        INPUT LAYER                                   │   │
│  │  Audio ──► VAD ──► Moonshine STT ──► Raw Transcript                 │   │
│  │                                           │                          │   │
│  │  Screenshot ──► (Optional) ──────────────┼───────────┐              │   │
│  │                                           │           │              │   │
│  │  Active App ──► Context Detector ────────┼───────────┤              │   │
│  └───────────────────────────────────────────┼───────────┼──────────────┘   │
│                                              │           │                   │
│                                              ▼           ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     PROCESSING TIERS                                 │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ TIER 1: Rule-Based (No additional ML)              ~50ms    │   │   │
│  │  │  • Tokenization artifact fix                                 │   │   │
│  │  │  • Voice commands (period, comma, new line)                  │   │   │
│  │  │  • Spelled word concatenation (S M L → SML)                  │   │   │
│  │  │  • Number/date/time formatting (rule-based)         ◄── NEW │   │   │
│  │  │  • Currency/phone formatting                        ◄── NEW │   │   │
│  │  │  • Filler word removal                              ◄── NEW │   │   │
│  │  │  • User dictionary replacements                              │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ TIER 2: Lightweight ML (Punctuation)               ~100ms   │   │   │
│  │  │  • ELECTRA-small punctuation restoration            ◄── NEW │   │   │
│  │  │  • Truecasing (capitalization)                      ◄── NEW │   │   │
│  │  │  • Prosody-based sentence boundaries                         │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ TIER 3: Fine-tuned LLM (Text Quality)              ~200ms   │   │   │
│  │  │  • SmolLM2-1.7B or Gemma-2B (fine-tuned)            ◄── NEW │   │   │
│  │  │  • Context-aware formatting                                  │   │   │
│  │  │  • Grammar correction                                        │   │   │
│  │  │  • Homophone resolution                                      │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ TIER 4: Multimodal Assistant (Optional)            ~300ms   │   │   │
│  │  │  • Gemma 3n E2B with vision                         ◄── NEW │   │   │
│  │  │  • Screenshot understanding                                  │   │   │
│  │  │  • Tool calling / actions                                    │   │   │
│  │  │  • Deep context awareness                                    │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                              │                              │
│                                              ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      PERSONALIZATION LAYER                           │   │
│  │  • User correction history                              ◄── NEW     │   │
│  │  • Per-app preferences                                  ◄── NEW     │   │
│  │  • Custom vocabulary learning                           ◄── NEW     │   │
│  │  • Local LoRA adaptation (advanced)                     ◄── FUTURE  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                              │                              │
│                                              ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        OUTPUT LAYER                                  │   │
│  │  Formatted Text ──► Clipboard / Direct Input                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Enhanced Rule-Based Processing (2-3 weeks of work)

**Goal:** Reduce LLM dependency by handling common patterns with rules.

#### 1.1 Number Formatting Module

```rust
// crates/voiceflow-core/src/prosody/number_formatting.rs

/// Convert spoken numbers to digits
/// "twenty three" → "23"
/// "three point five" → "3.5"
/// "one thousand two hundred" → "1,200"

pub fn format_numbers(text: &str) -> String {
    // Word-to-number mapping
    // Handle compound numbers
    // Handle decimals
    // Handle ordinals (first → 1st)
}
```

**Patterns to handle:**
- Cardinals: one, two, ... twenty-one, ... one hundred
- Ordinals: first, second, third → 1st, 2nd, 3rd
- Decimals: "three point one four" → 3.14
- Large numbers: "two million" → 2,000,000
- Negative: "negative five" → -5

#### 1.2 Date/Time Formatting

```rust
// crates/voiceflow-core/src/prosody/datetime_formatting.rs

/// "January fifteenth twenty twenty three" → "January 15, 2023"
/// "three thirty PM" → "3:30 PM"
/// "quarter past four" → "4:15"
```

**Patterns:**
- Dates: month + day + optional year
- Times: hour + minutes + AM/PM
- Relative: "tomorrow", "next week" (context-dependent)
- Durations: "two hours and thirty minutes" → "2h 30m"

#### 1.3 Currency Formatting

```rust
// crates/voiceflow-core/src/prosody/currency_formatting.rs

/// "fifty dollars" → "$50"
/// "twenty five cents" → "$0.25"
/// "five ninety nine" (in price context) → "$5.99"
```

#### 1.4 Phone Number Formatting

```rust
// crates/voiceflow-core/src/prosody/phone_formatting.rs

/// "five five five one two three four five six seven" → "555-123-4567"
/// Area code detection
/// International format support
```

#### 1.5 Filler Word Removal

```rust
// crates/voiceflow-core/src/prosody/filler_removal.rs

/// Remove: um, uh, er, like (when filler), basically, literally
/// Context-aware: don't remove "like" in "I like pizza"
```

**Deliverables:**
- [ ] `number_formatting.rs` with comprehensive number handling
- [ ] `datetime_formatting.rs` with date/time patterns
- [ ] `currency_formatting.rs` with multi-currency support
- [ ] `phone_formatting.rs` with format detection
- [ ] `filler_removal.rs` with context-aware removal
- [ ] Unit tests for each module
- [ ] Integration into pipeline.rs

---

### Phase 2: Lightweight Punctuation Model (2 weeks of work)

**Goal:** Add punctuation without a full LLM, using a specialized lightweight model.

#### 2.1 Model Selection

**Recommended:** ELECTRA-small based punctuation restoration

| Model | Size | Latency | Accuracy |
|-------|------|---------|----------|
| ELECTRA-small | ~13MB | ~50ms | 90%+ F1 |
| DistilBERT-punct | ~66MB | ~100ms | 92%+ F1 |
| BertPunc | ~110MB | ~150ms | 94%+ F1 |

#### 2.2 Implementation

```rust
// crates/voiceflow-core/src/ml/punctuation_model.rs

pub struct PunctuationRestorer {
    model: ElectraSmall,
    tokenizer: WordPieceTokenizer,
}

impl PunctuationRestorer {
    /// Restore punctuation with minimal latency
    /// Input: "hello how are you doing today"
    /// Output: "Hello, how are you doing today?"
    pub fn restore(&self, text: &str) -> String {
        // Tokenize
        // Run inference (batch of 4 words for streaming)
        // Insert punctuation tokens
        // Return formatted text
    }
}
```

#### 2.3 Training Data

Create training dataset from:
1. LibriSpeech transcripts (with ground truth punctuation)
2. Common Crawl filtered text
3. Domain-specific corpora (code comments, emails, chat)

#### 2.4 Integration Options

**Option A: Rust-native ONNX**
```toml
# Cargo.toml
ort = "2.0"  # ONNX Runtime for Rust
```

**Option B: Candle (Hugging Face's Rust ML)**
```toml
candle-core = "0.4"
candle-transformers = "0.4"
```

**Deliverables:**
- [ ] Evaluate ELECTRA-small vs DistilBERT for punctuation
- [ ] Export chosen model to ONNX
- [ ] Implement Rust inference wrapper
- [ ] Integrate into Tier 2 of pipeline
- [ ] Benchmark latency on target hardware

---

### Phase 3: Fine-tuned Formatting LLM (3-4 weeks of work)

**Goal:** Create a purpose-built LLM for dictation formatting that's faster and more accurate than generic models.

#### 3.1 Base Model Selection

| Model | Params | VRAM | Inference (MLX) | Quality |
|-------|--------|------|-----------------|---------|
| SmolLM2-1.7B | 1.7B | ~1.2GB | ~80 tok/s | Good |
| Gemma-2B | 2B | ~1.5GB | ~60 tok/s | Better |
| Qwen2.5-1.5B | 1.5B | ~1.1GB | ~90 tok/s | Good |
| Phi-3-mini | 3.8B | ~2.5GB | ~40 tok/s | Best |

**Recommendation:** Start with **SmolLM2-1.7B** for speed, upgrade to **Gemma-2B** if quality insufficient.

#### 3.2 Fine-tuning Strategy

**Task:** Dictation transcript → Formatted text

**Training data sources:**
1. **Synthetic generation** - Use GPT-4/Claude to generate (raw, formatted) pairs
2. **LibriSpeech** - Real STT output → ground truth text
3. **User corrections** - From VoiceFlow usage (with consent)

**Dataset format:**
```json
{
  "input": "um so i was thinking like maybe we could meet at three thirty tomorrow at the coffee shop on main street",
  "output": "I was thinking maybe we could meet at 3:30 tomorrow at the coffee shop on Main Street.",
  "context": "email"
}
```

**LoRA Configuration:**
```python
lora_config = LoraConfig(
    r=4,                    # Rank 4 sufficient for formatting
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

#### 3.3 Training Pipeline

```bash
# 1. Generate synthetic training data
python scripts/generate_training_data.py \
  --source librispeech \
  --output data/dictation_pairs.jsonl \
  --samples 10000

# 2. Fine-tune with LoRA
python scripts/finetune_lora.py \
  --base-model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --data data/dictation_pairs.jsonl \
  --output models/voiceflow-formatter \
  --epochs 3 \
  --batch-size 4

# 3. Merge LoRA weights
python scripts/merge_lora.py \
  --base-model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --lora models/voiceflow-formatter \
  --output models/voiceflow-formatter-merged

# 4. Quantize for deployment
python scripts/quantize.py \
  --model models/voiceflow-formatter-merged \
  --output models/voiceflow-formatter-q4 \
  --bits 4
```

#### 3.4 Inference Optimization

**Target:** <200ms for 100 tokens

**Optimizations:**
1. **Quantization:** Q4_K_M reduces model size 4x, minimal quality loss
2. **MLX backend:** ~230 tok/s on M2 Ultra, ~80 tok/s on M1
3. **Speculative decoding:** Use smaller model for draft tokens
4. **KV cache:** Reuse for streaming transcription
5. **Batch processing:** Process multiple sentences together

```rust
// crates/voiceflow-core/src/llm/optimized_engine.rs

pub struct OptimizedLLM {
    model: QuantizedModel,
    kv_cache: KVCache,
    config: InferenceConfig,
}

impl OptimizedLLM {
    pub fn format(&mut self, text: &str, context: &Context) -> String {
        // Prepare prompt with context
        let prompt = self.build_prompt(text, context);

        // Run inference with KV cache
        let output = self.generate_with_cache(&prompt);

        // Post-process
        self.clean_output(output)
    }
}
```

**Deliverables:**
- [ ] Create synthetic training dataset (10K+ pairs)
- [ ] Fine-tune SmolLM2-1.7B with LoRA
- [ ] Benchmark quality vs current Qwen3-4B
- [ ] Quantize to Q4 for deployment
- [ ] Integrate MLX backend for inference
- [ ] Achieve <200ms latency target

---

### Phase 4: Context Awareness (2-3 weeks of work)

**Goal:** Understand the active application and adapt formatting accordingly.

#### 4.1 App Context Detection

```rust
// crates/voiceflow-core/src/context/app_detector.rs

pub enum AppContext {
    Email { client: EmailClient },
    Chat { platform: ChatPlatform },
    Code { language: Language, ide: IDE },
    Document { app: DocApp },
    Terminal,
    Browser { site: Option<String> },
    Unknown,
}

impl AppDetector {
    /// Detect active application using accessibility APIs
    pub fn detect(&self) -> AppContext {
        #[cfg(target_os = "macos")]
        {
            // Use NSWorkspace to get active app
            // Parse window title for additional context
        }
    }
}
```

#### 4.2 Context-Specific Formatting Rules

```rust
// crates/voiceflow-core/src/context/formatters.rs

pub trait ContextFormatter {
    fn format(&self, text: &str) -> String;
    fn get_style(&self) -> FormattingStyle;
}

impl ContextFormatter for EmailContext {
    fn format(&self, text: &str) -> String {
        // Professional tone
        // Proper salutations
        // Signature handling
    }

    fn get_style(&self) -> FormattingStyle {
        FormattingStyle::Formal
    }
}

impl ContextFormatter for SlackContext {
    fn format(&self, text: &str) -> String {
        // Casual tone
        // Emoji support
        // @mention handling
    }

    fn get_style(&self) -> FormattingStyle {
        FormattingStyle::Casual
    }
}

impl ContextFormatter for CodeContext {
    fn format(&self, text: &str) -> String {
        // Preserve technical terms
        // Handle variable names
        // Code-specific punctuation
    }
}
```

#### 4.3 Per-App Vocabulary

```toml
# ~/.config/voiceflow/contexts/slack.toml
[vocabulary]
"lgtm" = "LGTM"
"ptal" = "PTAL"
"wfh" = "WFH"

[style]
tone = "casual"
emoji = true
```

**Deliverables:**
- [ ] Implement app detection for macOS
- [ ] Create context-specific formatters
- [ ] Add per-app vocabulary support
- [ ] Test with common apps (Slack, Mail, VS Code, Terminal)

---

### Phase 5: Gemma 3n Multimodal Integration (4-6 weeks of work)

**Goal:** Enable screenshot understanding for Wispr Flow-level context awareness.

#### 5.1 Why Gemma 3n

- **E2B model:** Effective 2B parameters, fits in ~2GB RAM
- **Multimodal:** Native vision + text understanding
- **On-device optimized:** Designed for mobile/laptop deployment
- **Tool use:** Built-in function calling capability

#### 5.2 Screenshot Context Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  Screenshot Context Flow                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Capture Screenshot (on voice activation)                │
│     └─► Capture active window only (privacy)                │
│                                                             │
│  2. Resize for Efficiency                                   │
│     └─► 512x512 (Gemma 3n optimal resolution)              │
│                                                             │
│  3. Generate Context Description                            │
│     └─► "User is composing email in Gmail to john@..."     │
│     └─► "User is writing code in VS Code, Python file"     │
│     └─► "User is in Slack #engineering channel"            │
│                                                             │
│  4. Condition Formatting                                    │
│     └─► Pass context to LLM formatter                      │
│     └─► Adjust tone, vocabulary, style                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 5.3 Implementation

```rust
// crates/voiceflow-core/src/multimodal/gemma_vision.rs

pub struct GemmaVisionContext {
    model: Gemma3nE2B,
    image_processor: MobileNetV5,
}

impl GemmaVisionContext {
    pub fn analyze_screenshot(&self, image: &Image) -> ContextDescription {
        // Resize to 512x512
        let processed = self.image_processor.process(image);

        // Generate context description
        let prompt = "Describe what application the user is using and what they appear to be doing. Be concise.";

        let description = self.model.generate(prompt, Some(processed));

        ContextDescription::parse(&description)
    }

    pub fn format_with_context(
        &self,
        transcript: &str,
        screenshot: &Image
    ) -> String {
        let context = self.analyze_screenshot(screenshot);

        let prompt = format!(
            "Context: {}\n\nFormat this dictation appropriately:\n{}",
            context.description,
            transcript
        );

        self.model.generate(&prompt, None)
    }
}
```

#### 5.4 Privacy Controls

```rust
// crates/voiceflow-core/src/multimodal/privacy.rs

pub struct PrivacyConfig {
    /// Never capture screenshots
    pub disable_screenshots: bool,

    /// Blur sensitive areas (detected passwords, credit cards)
    pub blur_sensitive: bool,

    /// Only capture when explicitly enabled per-session
    pub require_explicit_consent: bool,

    /// Apps to never capture
    pub blacklist: Vec<String>,
}
```

**Deliverables:**
- [ ] Integrate Gemma 3n E2B model
- [ ] Implement screenshot capture (macOS)
- [ ] Create vision-to-context pipeline
- [ ] Add privacy controls and app blacklist
- [ ] Benchmark end-to-end latency
- [ ] Test across different app contexts

---

### Phase 6: Personalization Engine (3-4 weeks of work)

**Goal:** Learn from user corrections to improve over time, entirely locally.

#### 6.1 Correction Tracking

```rust
// crates/voiceflow-core/src/personalization/corrections.rs

pub struct CorrectionStore {
    db: SqliteConnection,
}

impl CorrectionStore {
    /// Store a user correction
    pub fn record_correction(
        &self,
        original: &str,
        corrected: &str,
        context: &AppContext,
        timestamp: DateTime<Utc>,
    ) {
        // Store the correction pair
        // Track frequency of similar corrections
        // Associate with context
    }

    /// Get learned corrections for context
    pub fn get_corrections(&self, context: &AppContext) -> Vec<Correction> {
        // Return corrections relevant to current context
        // Prioritize by frequency
    }
}
```

#### 6.2 Learning Strategies

**Strategy 1: Dictionary Learning**
- Track repeated corrections
- After N occurrences, add to personal dictionary
- Simple, deterministic, fast

**Strategy 2: Pattern Matching**
- Identify correction patterns (e.g., always capitalize "iPhone")
- Generate rules from patterns
- Apply rules in Tier 1 processing

**Strategy 3: Local LoRA Adaptation (Advanced)**
- Periodically fine-tune LoRA weights with user corrections
- Requires background processing
- Most powerful but resource-intensive

```rust
// crates/voiceflow-core/src/personalization/adapter.rs

pub struct PersonalAdapter {
    base_model: SmolLM2,
    personal_lora: Option<LoraWeights>,
    correction_store: CorrectionStore,
}

impl PersonalAdapter {
    /// Periodically update LoRA weights with user corrections
    pub async fn adapt(&mut self) {
        let corrections = self.correction_store.get_recent(100);

        if corrections.len() >= 50 {
            // Fine-tune LoRA on corrections
            let new_weights = self.train_lora(&corrections).await;
            self.personal_lora = Some(new_weights);
        }
    }
}
```

#### 6.3 User Vocabulary Learning

```rust
// crates/voiceflow-core/src/personalization/vocabulary.rs

pub struct VocabularyLearner {
    /// Words the user types frequently that STT misses
    learned_words: HashSet<String>,

    /// Word → preferred spelling mapping
    spelling_preferences: HashMap<String, String>,
}

impl VocabularyLearner {
    /// Learn from clipboard content the user pastes after dictation
    pub fn learn_from_paste(&mut self, dictated: &str, pasted: &str) {
        // Diff the texts
        // Extract new vocabulary
        // Store spelling preferences
    }
}
```

**Deliverables:**
- [ ] Implement SQLite-based correction storage
- [ ] Add correction tracking to pipeline
- [ ] Implement dictionary learning from corrections
- [ ] Add pattern detection for automatic rules
- [ ] (Optional) Implement local LoRA adaptation
- [ ] Create vocabulary learning from user behavior

---

## Model & Resource Summary

### Memory Budget (Target: <2GB total)

| Component | Memory | Notes |
|-----------|--------|-------|
| Moonshine STT | ~100MB | Base model + buffers |
| Punctuation Model | ~15MB | ELECTRA-small ONNX |
| Formatting LLM (Q4) | ~1.2GB | SmolLM2-1.7B quantized |
| Gemma 3n E2B (Optional) | ~1.5GB | Multimodal, replaces LLM |
| Context/Personalization | ~50MB | SQLite + caches |
| **Total (Tier 3)** | **~1.4GB** | Without multimodal |
| **Total (Tier 4)** | **~1.7GB** | With Gemma 3n |

### Latency Budget (Target: <500ms)

| Stage | Target | Notes |
|-------|--------|-------|
| VAD + Audio | 50ms | Already optimized |
| Moonshine STT | 150ms | Current performance |
| Tier 1 (Rules) | 10ms | Pure Rust, fast |
| Tier 2 (Punctuation) | 50ms | ELECTRA-small |
| Tier 3 (LLM) | 200ms | SmolLM2 Q4 on MLX |
| Post-processing | 10ms | Cleanup |
| **Total** | **~470ms** | Under 500ms target |

### Quality Targets

| Metric | Current | Target | Wispr Flow |
|--------|---------|--------|------------|
| Raw WER | 3.77% | 3.5% | ~3% |
| Formatted WER | 3.96% | 3.3% | ~2.5% |
| LLM Regression | +0.19% | -0.2% | -0.5% |
| User Satisfaction | Good | Excellent | Excellent |

---

## Implementation Timeline

```
Month 1: Foundation
├── Week 1-2: Phase 1 - Rule-based enhancements
│   └── Number, date, currency, phone formatting
├── Week 3-4: Phase 2 - Punctuation model
│   └── ELECTRA integration, benchmarking

Month 2: Core ML
├── Week 5-6: Phase 3 - Fine-tuned LLM
│   └── Training data generation, LoRA fine-tuning
├── Week 7-8: Phase 3 continued
│   └── Quantization, MLX integration, latency optimization

Month 3: Intelligence
├── Week 9-10: Phase 4 - Context awareness
│   └── App detection, context-specific formatting
├── Week 11-12: Phase 5 - Gemma 3n (optional)
│   └── Multimodal integration, screenshot pipeline

Month 4: Polish
├── Week 13-14: Phase 6 - Personalization
│   └── Correction tracking, vocabulary learning
├── Week 15-16: Testing & Optimization
│   └── End-to-end testing, performance tuning
```

---

## Success Metrics

### Quantitative

1. **Latency:** End-to-end < 500ms (p95)
2. **Accuracy:** WER improvement of 0.5%+ over raw STT
3. **Memory:** Total footprint < 2GB
4. **Regression:** LLM should never make output worse

### Qualitative

1. **Context awareness:** Correctly adapts to Email vs Slack vs Code
2. **Learning:** Noticeably improves with usage
3. **Reliability:** No crashes, graceful degradation
4. **Privacy:** Zero data transmission (verifiable)

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM too slow | High | Smaller model, better quantization |
| LLM quality insufficient | High | Fine-tuning, better training data |
| Memory exceeds budget | Medium | Lazy loading, model swapping |
| Punctuation model inaccurate | Medium | Fall back to LLM for punctuation |
| Context detection unreliable | Low | Fall back to generic formatting |
| LoRA adaptation unstable | Low | Use simpler dictionary learning |

---

## Next Steps

1. **Immediate:** Start Phase 1 implementation (rule-based enhancements)
2. **This week:** Benchmark ELECTRA-small punctuation model
3. **This month:** Generate training data for LLM fine-tuning
4. **Decision point:** Choose between SmolLM2 and Gemma-2B after benchmarks

---

## References

### Research Papers
- [Moonshine: Speech Recognition for Live Transcription](https://arxiv.org/html/2410.15608v2)
- [MobileASR: On-device Learning for Voice Personalization](https://arxiv.org/abs/2306.09384)
- [Lightweight Online Punctuation Restoration](https://www.sciencedirect.com/science/article/abs/pii/S0167639325000846)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### Models
- [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Gemma 3n E2B](https://huggingface.co/google/gemma-3n-E2B-it)
- [BertPunc](https://github.com/nkrnrnk/BertPunc)

### Tools
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform inference
