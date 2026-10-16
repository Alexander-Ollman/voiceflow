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
Bonsai for writing. A whole-utterance classifier returns a label or score; it cannot supply the
replacement sentence that the existing redo endpoint also generates. Token-level
edit taggers can reconstruct corrected text, as detailed in the follow-up below.

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
| Punctuation, contextual spelling and sentence cleanup | Bonsai generation | Evaluate punctuation/case heads and GECToR-style edit taggers; see the researched variants below. Retain generation until dictation-specific comparisons pass. |
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

## Follow-up research: models that actually format and correct text

Checked October 15, 2026. A whole-utterance intent classifier only chooses an
action, but a **token-level edit tagger** can change text by choosing an edit for
each token. That makes the earlier classifier/generator split incomplete for
formatting. These candidates have not been benchmarked or integrated here.

| Candidate | What it can do | Fit for VoiceFlow | Main limitation |
|---|---|---|---|
| NVIDIA NeMo punctuation/capitalization | Joint token classification for punctuation and case | First formatting-only baseline | Limited punctuation labels; cannot repair semantic mishears or apply arbitrary writing styles. |
| GECToR, single encoder / distilled student | Predict keep, delete, insertion, replacement and grammatical transforms | Best edit-tagger candidate for conservative grammar cleanup | Fixed edit vocabulary and possibly several passes; names/code require protection and ASR-specific evaluation. |
| Google FELIX | Tag, reorder and insert using a masked language model | More flexible correction/reordering experiment | Two model stages and a training/export project, not a ready native formatter. |
| FastCorrect / FastCorrect 2 / SoftCorrect | Non-autoregressive ASR error detection/correction | Relevant research for phonetic misrecognitions | Released examples are Chinese-focused; multi-candidate variants need ASR hypotheses our current API does not return. |
| Community `jbochi/coedit-small` | Instruction-based grammar, fluency, formality and paraphrasing with FLAN-T5-small | A small generation baseline to compare against taggers | Autoregressive; thin model documentation and no demonstrated VoiceFlow/name-preservation quality. |
| Official `grammarly/coedit-large` | Instruction-based text revision, 770M parameters | Useful research reference | Model card is CC-BY-NC-4.0; do not treat it as an unrestricted shipping replacement. |

### Punctuation and capitalization

NVIDIA's documented model attaches two classification heads to a BERT-like
encoder, predicting punctuation after each word and whether it should be
capitalized. Default punctuation covers comma, period and question mark. This is
much narrower than free-form writing and is a plausible first Basic Dictate
formatter. [NVIDIA architecture and inference documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/nlp/punctuation_and_capitalization.html)

Treat the pretrained BERT checkpoint as a baseline; a smaller/distilled encoder
would need separate training and evaluation. Do not assume a ready DistilBERT
checkpoint exists with equivalent quality. Model artifact terms are separate
from the NeMo source license. The [NGC model page](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/punctuation_en_bert/-?_lr=1)
points to NGC terms; verify the exact artifact before distribution.

### GECToR and distilled variants

GECToR provides BERT, RoBERTa and XLNet pretrained variants and an edit-tagging
pipeline. It uses token transformations to reconstruct corrected text rather
than freely regenerating every word. For this app, start with one encoder rather
than an ensemble and preserve input when edit confidence is insufficient.
The reference code has an older Python/AllenNLP stack, so shipping that stack
unchanged would undermine the lightweight objective. [Official repository](https://github.com/grammarly/gector)

Grammarly also demonstrated distilling an ensemble into a single GECToR tagger.
This establishes a useful training strategy; it is not proof that an off-the-shelf
small model matches Bonsai on dictated text. [Grammarly's distillation experiments](https://www.grammarly.com/blog/engineering/experimenting-with-gector/)

### FELIX and ASR-specific correctors

FELIX separates token selection/reordering from insertion using a masked language
model. Both stages are non-autoregressive. Its reference implementation is
TensorFlow 2 with BERT-based training examples. It is a more flexible research
route when local edit tags prove too restrictive, but doubles the model-stage
and export work. Published speedups concern the paper's hardware/tasks and are
not Apple Silicon estimates. [Google architecture description](https://research.google/blog/introducing-felix-flexible-text-editing-through-tagging-and-insertion/),
[official implementation](https://github.com/google-research/google-research/tree/master/felix).

FastCorrect predicts edit alignment and corrected output non-autoregressively;
its released pretrained example is Chinese-character based. FastCorrect 2 uses
multiple ASR beam candidates, and SoftCorrect adds a soft error detector and
constrained correction. Our daemon currently returns one transcript, so these
are training/ASR-interface projects rather than drop-in English replacements.
[FastCorrect](https://github.com/microsoft/NeuralSpeech/tree/master/FastCorrect),
[FastCorrect 2](https://github.com/microsoft/NeuralSpeech/tree/master/FastCorrect2),
[SoftCorrect](https://github.com/microsoft/NeuralSpeech/tree/master/SoftCorrect).

### Small instruction-based editors

The community `jbochi/coedit-small` is fine-tuned from FLAN-T5-small on the CoEdIT
dataset and offers examples for grammar, coherence, formality and paraphrasing.
Its card contains evaluation loss and ROUGE scores but leaves important intended
use and training-data descriptions incomplete. Those metrics do not establish
safe preservation of names, numbers, code or dictated intent.
[Author's model card](https://huggingface.co/jbochi/coedit-small).

FLAN-T5-small is listed at 77M parameters with Apache-2.0 licensing. Its raw
weight payload is approximately 154 MB at FP16 or 77 MB at INT8, **before**
quantization metadata, tokenizer, activations, caches and runtime overhead. These
are arithmetic estimates, not measured Mac memory or latency.
[Base model card](https://huggingface.co/google/flan-t5-small).

The official CoEdIT-large checkpoint is 770M and uses CC-BY-NC-4.0. That license
must not be inferred for every independently trained small variant: the
community-small card declares Apache-2.0, and the CoEdIT dataset card also
currently declares Apache-2.0. Check the exact weights and training provenance;
source, base-model, dataset and fine-tuned-weight metadata are distinct.
[Official large model](https://huggingface.co/grammarly/coedit-large),
[dataset card](https://huggingface.co/datasets/grammarly/coedit).

### Proposed comparison

Benchmark three alternatives against current Bonsai and a rules-only control:

1. Punctuation/case tagging for a conservative Basic Dictate mode.
2. A single GECToR-style model for punctuation plus local grammar edits.
3. A small instruction editor as a generation-quality comparison.

Evaluate ASR transcripts from the actual Parakeet version rather than only
written-grammar benchmarks. Parakeet already emits punctuation and case; compare
conditional repair against passing its output through rather than blindly
stripping punctuation and restoring it. Include correctly written text, ambiguous phonetic
mishears, partial sentences, negation, names, IDs, URLs, amounts, dates and code.
Score punctuation/case accuracy, edit precision, unnecessary-edit rate, semantic
preservation and exact preservation of protected spans. Report p50/p95 latency,
peak RSS/Metal memory and cold startup on the same Mac. Retain Bonsai for Compose,
style-heavy rewriting and ambiguous semantic repairs until results justify a
change. No benchmark results are claimed by this research update.

A native Core ML or ONNX implementation could avoid another Python/PyTorch
runtime, but tokenizer parity, edit reconstruction, conversion support and
quantization must all be validated. Existing ONNX packaging does not make a new
model automatically compatible. Two uncompressed encoders can also erase much
of the memory gain over Bonsai's aggressively quantized weights. The useful
product outcome is a fully functioning basic mode with Bonsai unloaded, not
simply adding another resident model alongside it.
