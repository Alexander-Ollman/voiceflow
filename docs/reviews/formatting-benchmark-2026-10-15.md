---
title: Formatting Benchmark and Context Verification
icon: 🔬
order: 32
tags: [inference, benchmark, formatting]
---

# Formatting benchmark and context verification

## Recommendation

Keep Bonsai available for contextual correction and writing. Build a conservative
basic path that preserves already-correct Parakeet output, and trial a small
**edit verifier alongside Bonsai**. Do not replace the formatter with an
unconditional punctuation model, and do not add a permanently resident second
transformer while calling the application lighter.

The experiment is now implemented and measured: seven paths, 43 authored cases,
20 additional synthetic-speech/Parakeet cases, three recorded calls per case.
No production model defaults changed. Candidate weights are not in the repo.
[Reproduction, scoring and adapters](../../benchmarks/formatting/README.md),
[pinned artifacts/environment](../../benchmarks/formatting/environment.json),
[per-case evidence](../../benchmarks/formatting/results).

## Authored text results

Exact matches are agreement with a small set of authored acceptable references,
not an estimate of general semantic correctness. “Clean kept” counts identical
output on the 16 already-correct examples. Grammar and punctuation are separate
sets of eight cases each. Sensitive-span failures count cases among 15 containing
names, amounts, identifiers, URLs or negation that must remain exact.

| Path | Clean kept | Punctuation exact | Grammar exact | All exact /43 | Sensitive failures /15 | Warm median / p95 ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pass through | 16/16 | 0/8 | 0/8 | 18 | 0 | <0.01 / <0.01 | 25 |
| Existing rules | 6/16 | 0/8 | 0/8 | 14 | 10 | 0.19 / 5.74 | 35 |
| Raw punctuation/case ONNX | 5/16 | 7/8 | 0/8 | 14 | 11 | 26 / 69 | 798 |
| Guarded punctuation prototype | 16/16 | 7/8 | 0/8 | 25 | 0 | <0.01 / 71 | 841 |
| GECToR RoBERTa-base | 16/16 | 0/8 | 7/8 | 25 | 0 | 76 / 205 | 1,616 |
| Community CoEdIT-small | 16/16 | 5/8 | 6/8 | 29 | 0 | 140 / 211 | 704 |
| Bonsai through native formatter | 7/16 | 6/8 | 7/8 | 26 | 9 | 101 / 176 | 2,604 |

Guarded punctuation's overall median is dominated by bypasses. On the eight
unpunctuated examples its median is **30 ms**. It checks the original text before
normalization, bypasses sensitive patterns and already-formatted sentences, and
rejects lexical changes. It intentionally does not remove fillers, execute voice
commands, fix grammar, or infer whether a terminal-punctuated sentence still
needs commas. Thus 16/16 preservation is partly the gate's deliberate behavior,
not proof of model understanding.

Raw PCS fixed punctuation quickly but damaged names/code when forced to strip
case and punctuation. GECToR's local grammar edits were strong on this tiny set,
but “There are a problem…” became “There was a problem…”, changing tense. CoEdIT
changed “The user has write access” to “The user has written access,” an example
of fluent-looking semantic damage that clean-text tests alone miss. One authored
Bonsai case varied across the three calls; other stored paths were stable.
Bonsai also failed both authored vocabulary-hint repairs (`cube control` →
`kubectl`, `post grass` → `Postgres`), so this run does not establish reliable
use of terminology context.

### A preprocessing problem precedes model selection

The cached production deterministic path changed `3:30 PM` to `3: 30 PM`,
`$5,000` to `$5, 000`, `v2.7.1` to `v2. 7. 1`, `userID` to `user ID`, and
`VoiceFlow` to `Voice Flow`. The Bonsai path retained several of those problems.
These are observed outputs from the binary identified in the environment file;
a freshly rebuilt Rust binary has not been evaluated here.

Before a production basic mode, protect URLs, numbers, code and mixed-case terms
**before** tokenization repair/spacing. Audit these transformations at each stage,
then evaluate the rebuilt normalizer. A verifier after generation can flag damage,
but it cannot excuse a lossy preprocessing step. This experiment records the
regressions without silently changing application behavior.

## Parakeet output results

Twenty selected fixtures were spoken by macOS Samantha, converted to 16 kHz mono
PCM and transcribed by the actual Parakeet source path with the cached v2 model.
The daemon had its own socket, PID file and log. No private recordings or history
were read. These are clean synthetic recordings, not evidence about accents,
noise, the other tester's capture failure, or production speech accuracy.

| Path | Exact /20 | Grammar exact /8 | Punctuation exact /8 | Sensitive failures /3 | Warm median / p95 ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| Parakeet pass through | 8 | 1 | 5 | 0 | <0.01 / <0.01 | 24 |
| Existing rules | 7 | 1 | 5 | 1 | 0.18 / 0.24 | 34 |
| Raw punctuation/case ONNX | 9 | 1 | 7 | 1 | 62 / 101 | 848 |
| Guarded punctuation prototype | 8 | 1 | 5 | 0 | <0.01 / 0.01 | 578 |
| GECToR | 13 | 6 | 5 | 0 | 103 / 216 | 1,603 |
| CoEdIT-small | 16 | 7 | 7 | 0 | 139 / 705 | 719 |
| Bonsai/native formatter | 15 | 8 | 6 | 1 | 141 / 285 | 2,598 |

Parakeet already supplied case and terminal punctuation in all 20 examples, so
the guarded punctuation path bypassed them all. It saved work but missed internal
commas. Synthetic ASR itself changed one erroneous sentence to the intended
correct wording; that is why pass-through already scores 1/8 on grammar. The
“preserve” subset here describes the spoken source, not necessarily perfect ASR;
clean input preservation and reference agreement must not be conflated.

## Footprint, latency and licensing limits

Measured on an M4 Max with 128 GiB unified RAM, macOS 15.6. Encoders used CPU/four
threads; Bonsai used Metal with a 16,384-token context and q8_0 KV. This compares
configured local paths, not equal compute budgets. The raw ONNX payload is
about 200 MiB, GECToR safetensors 488 MiB, and CoEdIT safetensors 294 MiB. The
Bonsai quantized file is about 1.1 GiB. Runtime memory exceeds those file sizes.

RSS includes Python/Torch and, for Bonsai, its child server. It is not an isolated
model measurement or a promise for native Core ML/ONNX deployment. The Python
punctuator wrapper imports Torch despite ONNX inference. No ANE/Core ML conversion,
quantization-quality experiment, app-wide battery test, or native memory test was
performed. Samples were sequential, not randomized, and p95 uses only three
calls per case. Cold file caches were not reset.

Authored-run model-load/import times were PCS 4.49 s, GECToR 7.01 s, CoEdIT 4.31 s,
and Bonsai server/native initialization 0.65 s followed by a 4.84 s first inference.
On the later ASR run Bonsai's first inference took 11.24 s. These varying cold
costs reinforce measuring lazy loading separately from warmed calls.

The PCS author declares Apache-2.0. The tested GECToR checkpoint is explicitly
noncommercial and is **research-only**, regardless of its implementation's MIT
license. Community CoEdIT-small declares Apache-2.0 but needs provenance and
artifact review before shipping. The official CoEdIT-large model has different
terms. [PCS checkpoint](https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english),
[GECToR checkpoint](https://huggingface.co/gotutiyan/gector-roberta-base-5k),
[CoEdIT-small checkpoint](https://huggingface.co/jbochi/coedit-small).

## Improving Bonsai with context-aware verification

Treat the useful quantity as **probability that a proposed edit is appropriate
given transcript, candidate and current context**. This does not change Bonsai's
internal confidence or weights. It estimates edit risk externally, selects work
for Bonsai, and can constrain a targeted second pass.

A whole-sentence confidence score hides a suspect product name or dropped list
item. Score changed spans, and assess structure separately:

| Level | Evidence/features | Intended response after validation |
|---|---|---|
| Word/term | ASR span, proposed spelling, nearby words, exact term in active document or user vocabulary, changed ID/amount/negation | Accept supported correction; preserve a fragile term when uncertain |
| Phrase | Aligned changed span, grammatical tagger agreement, local context, edit size, tense/meaning changes | Rank original against candidate; retry only an uncertain span |
| Lists/formatting | Requested style, target field capabilities, item content/order/count, nesting, punctuation, protected spans | Permit structure changes while preserving items; avoid list syntax in single-line fields |
| Whole result | Aggregated edit risks, coverage of uncertain spans, model disagreement, latency budget | Accept, fall back to the source, or request one bounded retry |

Concrete example: context containing `Postgres` supports correcting “post grass”
to that spelling. It does not establish that every edit mentioning `Postgres` is
correct. “write access” becoming “written access” should remain suspicious even
if the revised phrase sounds fluent. Turning “milk eggs bread” into three bullets
can preserve content; dropping “bread” cannot be classified as formatting alone.

The implemented [shadow feature extractor](../../scripts/formatting_bench/shadow_verifier.py)
records changed word spans, context-word matches, numeric/negation changes, lost
identifiers, and list counts/order-preserving lexical equality. It takes no
reference labels or fixture protections as inputs. Six focused tests cover
context support, amounts, negation, identifiers, and valid/invalid list changes.
Checked-in Bonsai feature reports show its current behavior; they do not accept,
reject, or paste edits. It has no learned weights, calibrated probabilities,
semantic entailment test, robust list parser, or application-context capture.

### Model variants worth testing for the verifier

Start with a logistic classifier or small tree model on these features and any
available tagger scores; this avoids another large resident encoder. For a
contextual verifier, compare a compact encoder over `(source, candidate, context)`
with edit-level heads. Distill from a stronger quality-estimation teacher only
with suitable labels/terms. These are proposed architectures, not ready models
shown to work by this benchmark.

GRECO is a directly relevant published approach: estimate correction quality
without a reference and select/combine edits from correction systems. Its paper
also discusses lighter logistic edit selection and contextual edit scorers. This
supports testing the verifier architecture, not assuming a pretrained GRECO
checkpoint is small or appropriate for English dictation/list formatting.
[GRECO paper](https://aclanthology.org/2023.emnlp-main.785/).

Bonsai's emitted `confidence` fields, token likelihoods if exposed, and agreement
between models can be features; none is correctness ground truth. The current
OpenAI-compatible backend does not request or return token log-probabilities.
Research has found verbalized confidence useful on some models/tasks, so it
should be evaluated rather than universally discarded. Those results do not
calibrate this quantized Bonsai formatter.
[Confidence elicitation study](https://arxiv.org/abs/2305.14975).

### Before any automatic gate

Collect labeled corrections and deliberately wrong alternatives, including
unchanged text, context conflicts, uncommon names, tense/negation changes and
list-item loss. Split by speaker/document/domain so related phrases do not leak
across training and held-out evaluation. Fit and calibrate on separate splits;
report accepted-edit precision, harmful edits accepted, coverage/abstention,
per-category risk, calibration error and added latency/memory. Choose thresholds
from the cost of changing meaning, not a self-reported “0.9”. The 63 stored cases
here are an exploratory regression set, not calibration/training evidence.

Use context already authorized by the application, minimize it to relevant text
and terms, and treat document text as evidence rather than instructions. Keep
unvalidated decisions shadow-only. A production lightweight path also needs
Bonsai actually unloaded when unused, plus a tested fallback for uncertain edits.
