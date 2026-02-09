# VoiceFlow Evaluation Plan: Do We Need Fine-Tuning?

## Objective

Determine whether the current architecture can match Wispr Flow quality with rule-based improvements alone, or if LLM fine-tuning is necessary.

**Decision criteria:**
- If rule-based achieves <3.5% formatted WER → No fine-tuning needed
- If rule-based achieves 3.5-4.0% → Fine-tuning optional (quality vs effort trade-off)
- If rule-based still >4.0% → Fine-tuning likely necessary

---

## Evaluation Framework

### 1. Baseline Measurements (Current State)

**Already collected:**
- Raw WER: 3.77% (Moonshine on LibriSpeech test-clean)
- LLM-Formatted WER: 3.96% (Qwen3-4B)
- LLM Regression: +0.19% (LLM makes it worse)

**Need to collect:**
- [ ] Wispr Flow benchmark (if possible)
- [ ] Breakdown by error category
- [ ] Latency measurements per stage

### 2. Error Taxonomy

Categorize all errors to understand what each solution tier can fix:

| Error Type | Example | Fix Tier |
|------------|---------|----------|
| **STT Errors** | "how tell" vs "hotel" | Unfixable (STT limit) |
| **Missing Punctuation** | "hello how are you" | Tier 2 (punctuation model) |
| **Capitalization** | "john went to paris" | Tier 2 (truecasing) |
| **Number Format** | "twenty three" vs "23" | Tier 1 (rules) |
| **Filler Words** | "um", "like", "basically" | Tier 1 (rules) |
| **Word Merging** | "facefight" vs "face fight" | Prompt fix / unfixable |
| **Archaic Modernization** | "yea" → "yes" | Prompt fix (preserve) |
| **Grammar Correction** | "he don't" → "he doesn't" | Tier 3 (LLM only) |
| **Homophone Errors** | "there" vs "their" | Tier 3 (LLM only) |
| **Context Formatting** | Email vs Slack tone | Tier 4 (context-aware) |

### 3. Test Datasets

| Dataset | Purpose | Size | Notes |
|---------|---------|------|-------|
| LibriSpeech test-clean | Baseline accuracy | 2620 samples | Already using |
| LibriSpeech test-other | Noisy/accented speech | 2939 samples | Harder test |
| Custom dictation set | Real-world dictation | TBD | Need to create |
| Number/date heavy | Rule-based evaluation | TBD | Need to create |

---

## Evaluation Phases

### Phase 1: Deep Dive on Current Failures (This Week)

**Goal:** Understand exactly where errors come from.

#### 1.1 Run Extended Analysis

```bash
# Run 1000 samples with detailed logging
cargo run --release -p voiceflow-cli -- eval \
  --samples 1000 \
  --dataset librispeech-test-clean \
  --output /tmp/eval_detailed.json \
  --log-all-errors
```

#### 1.2 Categorize All Errors

Create a script to categorize errors:

```rust
// Pseudo-code for error categorization
enum ErrorCategory {
    SttMistranscription,    // STT heard wrong word entirely
    MissingPunctuation,     // No period, comma, question mark
    WrongCapitalization,    // lowercase when should be upper
    NumberNotFormatted,     // "twenty" instead of "20"
    FillerNotRemoved,       // "um", "uh" still present
    LlmWordMerge,           // LLM merged words incorrectly
    LlmAddedWords,          // LLM hallucinated content
    LlmRemovedWords,        // LLM deleted spoken content
    LlmModernized,          // LLM changed archaic to modern
    GrammarError,           // Grammar issue LLM could fix
    HomophoneError,         // Wrong word, same sound
    Other,
}

fn categorize_error(reference: &str, raw: &str, formatted: &str) -> Vec<ErrorCategory> {
    // Compare reference vs raw → STT errors
    // Compare raw vs formatted → LLM changes
    // Classify each difference
}
```

#### 1.3 Generate Error Report

Output format:
```
=== VoiceFlow Error Analysis Report ===

Total Samples: 1000
Total Errors: 156

Error Distribution:
  STT Mistranscription:     45 (28.8%)  ← Cannot fix
  Missing Punctuation:      32 (20.5%)  ← Tier 2 can fix
  Wrong Capitalization:     28 (17.9%)  ← Tier 2 can fix
  Number Not Formatted:     12 (7.7%)   ← Tier 1 can fix
  Filler Not Removed:        8 (5.1%)   ← Tier 1 can fix
  LLM Word Merge:           15 (9.6%)   ← Prompt fix
  LLM Added Words:           6 (3.8%)   ← Prompt fix
  LLM Removed Words:         4 (2.6%)   ← Prompt fix
  Grammar Error:             3 (1.9%)   ← LLM only
  Homophone Error:           2 (1.3%)   ← LLM only
  Other:                     1 (0.6%)

Fixable by Tier 1 (rules):     20 errors → WER improvement: ~0.15%
Fixable by Tier 2 (punct):     60 errors → WER improvement: ~0.45%
Fixable by prompt changes:     25 errors → WER improvement: ~0.19%
Requires LLM intelligence:      5 errors → WER improvement: ~0.04%
Unfixable (STT limit):         45 errors → Cannot improve

Projected WER after improvements:
  Current:     3.96%
  After Tier 1: 3.81%
  After Tier 2: 3.36%
  After prompt: 3.17%
  Theoretical min: 3.13% (STT limit)
```

### Phase 2: Ablation Study (Week 2)

**Goal:** Measure impact of each component independently.

#### 2.1 Test Configurations

| Config | STT | Tier 1 | Tier 2 | LLM | Purpose |
|--------|-----|--------|--------|-----|---------|
| A | Moonshine | OFF | OFF | OFF | Pure STT baseline |
| B | Moonshine | ON | OFF | OFF | Rule-based only |
| C | Moonshine | ON | ON | OFF | + Punctuation model |
| D | Moonshine | ON | OFF | ON | Current (rules + LLM) |
| E | Moonshine | ON | ON | ON | Full pipeline |

#### 2.2 Implementation for Ablation

```rust
// Add to eval.rs
pub struct EvalConfig {
    pub enable_tier1_rules: bool,      // Number/date/filler formatting
    pub enable_tier2_punct: bool,      // Punctuation model
    pub enable_llm: bool,              // LLM formatting
    pub llm_model: Option<String>,     // Which LLM
}

pub fn run_ablation(configs: Vec<EvalConfig>, samples: usize) -> AblationReport {
    // Run each config on same samples
    // Compare results
}
```

#### 2.3 Expected Output

```
=== Ablation Study Results ===

Config A (STT only):
  WER: 3.77%  Latency: 180ms

Config B (STT + Tier 1):
  WER: 3.62%  Latency: 185ms  Improvement: -0.15%

Config C (STT + Tier 1 + Tier 2):
  WER: 3.20%  Latency: 235ms  Improvement: -0.57%

Config D (STT + Tier 1 + LLM):
  WER: 3.96%  Latency: 850ms  Improvement: +0.19% (REGRESSION)

Config E (Full pipeline):
  WER: 3.15%  Latency: 900ms  Improvement: -0.62%

Conclusion: Tier 2 punctuation model provides most value.
            LLM currently causes regression.
            Fine-tuning decision: [NEEDED/NOT NEEDED]
```

### Phase 3: Wispr Flow Comparison (Week 2-3)

**Goal:** Benchmark against the target we're trying to match.

#### 3.1 Create Comparison Test Set

Since we can't run Wispr on LibriSpeech programmatically:

1. **Record 50 dictation samples** covering:
   - Simple sentences
   - Numbers and dates
   - Technical terms
   - Email-style content
   - Code-related content

2. **Process each through:**
   - VoiceFlow (current)
   - Wispr Flow (manual)
   - macOS Dictation (baseline)

3. **Compare outputs**

#### 3.2 Qualitative Evaluation

For each sample, rate (1-5):
- Accuracy (words correct)
- Formatting (punctuation, capitalization)
- Naturalness (reads well)
- Context-appropriateness (matches intended use)

#### 3.3 Comparison Matrix

```
=== Wispr Flow Comparison ===

Sample: "I need to schedule a meeting for January fifteenth at three thirty pm with John"

Expected: "I need to schedule a meeting for January 15 at 3:30 PM with John."

Results:
  VoiceFlow:      "I need to schedule a meeting for January fifteenth at three thirty pm with John"
  Wispr Flow:     "I need to schedule a meeting for January 15th at 3:30 PM with John."
  macOS Dictation: "I need to schedule a meeting for January 15 at 3:30 PM with John"

Scores (Accuracy / Formatting / Natural / Context):
  VoiceFlow:       5/2/3/3 = 3.25
  Wispr Flow:      5/5/5/5 = 5.00
  macOS Dictation: 5/4/4/4 = 4.25

Gap to Wispr: 1.75 points (mostly formatting)
```

### Phase 4: Rule-Based Prototype (Week 3)

**Goal:** Implement Tier 1 improvements and measure actual impact.

#### 4.1 Quick Implementations

| Feature | Effort | Expected Impact |
|---------|--------|-----------------|
| Number formatting | 2 days | -0.10% WER |
| Date/time formatting | 1 day | -0.03% WER |
| Filler removal | 0.5 day | -0.02% WER |
| Currency formatting | 0.5 day | -0.01% WER |
| **Total Tier 1** | **4 days** | **-0.16% WER** |

#### 4.2 Punctuation Model Test

| Task | Effort | Expected Impact |
|------|--------|-----------------|
| Integrate ELECTRA-small | 2 days | -0.40% WER |
| Benchmark latency | 0.5 day | Target <50ms |
| **Total Tier 2** | **2.5 days** | **-0.40% WER** |

#### 4.3 Decision Point

After implementing Tier 1 + Tier 2:

```
Current WER:        3.96%
Projected WER:      3.40%
Wispr Flow target:  ~2.5-3.0%

Gap remaining:      0.40-0.90%

If gap < 0.5%:  → Rule-based approach sufficient
If gap > 0.5%:  → Consider fine-tuning
```

---

## Deliverables

### Week 1
- [ ] Error categorization script
- [ ] Detailed error analysis report (1000 samples)
- [ ] Error taxonomy documentation

### Week 2
- [ ] Ablation study implementation
- [ ] Ablation results report
- [ ] Wispr Flow comparison test set (50 samples)

### Week 3
- [ ] Tier 1 rule-based implementations
- [ ] ELECTRA punctuation model integration
- [ ] Final evaluation with all improvements
- [ ] **GO/NO-GO decision on fine-tuning**

---

## Success Criteria

### Fine-tuning NOT needed if:
1. Formatted WER < 3.5% with rule-based approach
2. LLM regression eliminated (formatted WER ≤ raw WER)
3. Qualitative scores within 0.5 points of Wispr Flow
4. Latency < 500ms end-to-end

### Fine-tuning NEEDED if:
1. Formatted WER still > 3.5% after rule-based improvements
2. Significant quality gap vs Wispr Flow on real dictation
3. Specific error categories require semantic understanding
4. User testing shows dissatisfaction with output quality

---

## Quick Start Commands

```bash
# 1. Run baseline evaluation
cargo run --release -p voiceflow-cli -- eval --samples 1000

# 2. Run with detailed error logging (need to implement)
cargo run --release -p voiceflow-cli -- eval --samples 1000 --categorize-errors

# 3. Run ablation study (need to implement)
cargo run --release -p voiceflow-cli -- eval --ablation --samples 500

# 4. Generate error report
cargo run --release -p voiceflow-cli -- eval --report /tmp/error_report.md
```

---

## Timeline

```
Week 1: Analysis
├── Day 1-2: Implement error categorization
├── Day 3-4: Run analysis, generate report
└── Day 5: Document findings

Week 2: Ablation + Comparison
├── Day 1-2: Implement ablation configs
├── Day 3: Run ablation study
├── Day 4-5: Create Wispr comparison test set
└── Day 5: Qualitative evaluation

Week 3: Prototype + Decision
├── Day 1-2: Implement Tier 1 rules
├── Day 3-4: Integrate punctuation model
├── Day 5: Final evaluation
└── Day 5: GO/NO-GO decision document
```
