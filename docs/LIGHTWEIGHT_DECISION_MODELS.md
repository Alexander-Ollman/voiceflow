---
title: Lightweight Decision Models
icon: 🧠
order: 30
tags: [architecture, inference, performance]
---

# Lightweight decision models

## Recommendation

Keep deterministic routing first. Trial a small classifier for fixed decisions,
starting with app classification, then redo-versus-new in shadow mode. Retain
Bonsai for writing. A classifier returns a label or score; it cannot supply the
replacement sentence that the existing redo endpoint also generates.

This is a code-informed proposal, not a trained model or a measured speedup.
No inference behavior or model defaults change in this update.

## Where to use each approach

| Task | Current implementation | Proposed approach |
|---|---|---|
| Voice command / mode routing | Rust `text_normalize/intent_classifier.rs` matches explicit phrases | Keep rules. Add a classifier only if a labeled evaluation shows missed paraphrases matter. |
| App / website writing style | Curated mappings plus `PersonaClassifier.swift`; unmapped apps use three LLM votes | Exact rules first; a compact classifier proposes a built-in category; unknown/custom styles abstain or use on-demand LLM classification. |
| Redo versus new dictation | `RedoSignal` gates calls; `structured_edit.rs::assess_redo` returns decision AND replacement text | A pair classifier predicts append/redo/uncertain. Keep field, window, recency and exact insertion guards. Generate text separately only after a confident redo. |
| Explicit change-X-to-Y edits | Intent classifier extracts `anchor_hint`; structured edits resolve a target | Prefer deterministic replacement when the anchor is exact and unique. Ambiguous edits still need generation or clarification. |
| Snippets, dictionary replacements, number/voice-command normalization | Existing deterministic processing | Keep rules and direct lookup. No new model. |
| Punctuation, contextual spelling and sentence cleanup | Bonsai generation | Keep generation for now. A dedicated punctuation/tagging model is a separate experiment with quality tradeoffs. |
| Compose, rewrite, summarize, reply, semantic edits | Bonsai generates new text | Keep a generator; a decision classifier cannot replace these behaviors. |
| Visible-text extraction | Existing OCR plus optional generative context helpers | Use existing OCR for literal extraction. Do not replace OCR with an intent classifier. |

## Small first experiment

Use app classification as the first production candidate because it proposes
preferences rather than modifying a user's text. Compare a simple bag-of-words
or character n-gram linear baseline with a Create ML text classifier, deployed
through Core ML. Apple documents [Create ML text classification](https://developer.apple.com/documentation/createml/creating-a-text-classifier-model)
and [Core ML deployment](https://developer.apple.com/documentation/coreml).
These frameworks provide a deployment route; they do not establish this app's
accuracy, memory use, or latency.

Predict stable built-in category IDs plus `unknown`, not arbitrary persona
names. Map a category onto a currently available style. User-created personas
are open-ended: a fixed-label classifier cannot infer a new class without new
training or a separate retrieval/ranking approach. Explicit user assignments
always win. Cache results by bundle ID / normalized hostname and classifier
version so routine dictation never reclassifies the application.

## Redo requires a separate decision contract

Candidate inputs: new transcript, previous insertion, normalized token overlap,
length ratio, correction phrases and elapsed time. The output contains only
`append`, `redo` or `uncertain` and calibrated scores. The current API also
produces replacement text; splitting that API is required before substituting
a classifier. Word-level repairs such as “store” after “I went to the stoor”
still need reconstruction. Low confidence defaults to append; uncertain inputs
must never automatically destroy earlier text.

The classifier must not choose a target or authorize insertion. Keep exact
app/window/field identity, anchored recent insertion, focus revalidation and
verified writing outside model control. The local runtime redesign already
provides stronger target contracts in `VoiceFlowDomain/Runtime/TargetWriting.swift`
and `Runtime/TargetWriter.swift`; carry these forward when integrating it.

## What actually reduces memory

Skipping a decision request reduces inference work. It does not release the
resident Bonsai weights or its KV cache while `llama-server` remains running.
`LlamaServerManager.swift` currently uses a 16,384-token context; `VoiceFlowApp.swift`
starts and prewarms the server at launch. Replacing persona classification alone
will barely affect normal dictation because that classification is an explicit
batch action, not a call on every utterance.

A materially lighter mode needs an end-to-end no-generator Dictate path:
Parakeet → existing deterministic normalization/dictionary/snippets → verified
insertion. It would omit generative polishing and semantic repairs. Start Bonsai
only when advanced formatting, Compose or Edit needs it; unload it after an idle
period. Audit both the Swift external-ASR path and Rust `pipeline.rs`, which can
initialize its LLM lazily, before promising that basic Dictate never starts it.
Do not silently downgrade formatting preferences. Offer the tradeoff explicitly.

Reducing context length is another experiment, not an automatic win: vocabulary,
field context and long Compose output require token budgeting. Measure actual
resident and Metal memory rather than equating the 1.1 GB model file with RAM.
Apple's [model compression guidance](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
covers storage, latency and runtime-memory tradeoffs for Core ML models; it is
not evidence that moving Bonsai to Core ML is necessary or beneficial here.

## Evaluation before switching behavior

1. Build a consented or synthetic labeled set, with a held-out split by app,
   utterance family and session. Include ordinary repeated phrases, short repairs,
   ambiguous app names, custom styles, and multilingual/code-heavy negatives.
   Do not export private dictations or train on user history implicitly.
2. Compare rules-only, current Bonsai and the classifier. Score precision/recall,
   calibration, abstention rate, and especially false redo replacements.
3. Measure warm and cold p50/p95 latency, model size, process RSS, Metal memory,
   generator request count and cold-start cost on the same Apple Silicon Mac.
   Include an end-to-end basic Dictate run with the generator stopped.
4. Run the redo classifier in opt-in local shadow mode first. It records aggregate
   disagreement and timings, never changes paste behavior, and stores no raw text
   by default. A model's self-reported confidence is not calibration evidence.
5. Choose rollout thresholds from held-out precision and cost of mistakes; keep
   rollback to current routing. Require no target-safety regression before use.

Prioritize waveform CPU savings now, classification experiments next, then a
separately scoped lazy-generator / lightweight Dictate mode for larger memory
savings. Do not add a second transformer permanently beside Bonsai and assume
the total application is lighter.
