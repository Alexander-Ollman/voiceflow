---
title: Decision Model Research and Bonsai Evaluation
order: 34
tags: [research, architecture, inference, performance]
---

# Decision model research and Bonsai evaluation

Recorded October 19, 2026, from research and experiments conducted October 15–16.
This document preserves findings and future options; it does not introduce a model
or change application behavior.

## Current decision

Keep Bonsai. Defer decision-model development and fine-tuning. The next formatter
work, when resumed, should improve the existing Bonsai pipeline, starting with
text preservation and normalization. No additional training or inference run is
authorized by this documentation update.

Decision models remain a credible future way to reduce generator calls and
independently assess proposed corrections. The completed experiments do not yet
establish a calibrated verifier or a production-ready replacement for Bonsai.

## What a decision model could do

A contextual encoder can read both sides of a word and predict whether to keep,
delete, replace, or insert text. GECToR already demonstrates this edit-tagging
approach. A whole-sentence classifier alone cannot reconstruct arbitrary text,
but a token-level tagger can reconstruct text using a defined edit vocabulary.

| Task | Suitable approach | Boundary |
|---|---|---|
| Explicit voice commands, snippets, dictionary substitutions | Rules and direct lookup | Preserve existing deterministic behavior. |
| App or website writing-style category | Small classifier after exact mappings | Predict stable built-in categories; abstain for unknown/custom styles. Cache decisions. |
| Redo versus new dictation | Pair classifier with append/redo/uncertain outputs | Decision is separate from generating replacement text and selecting its target. |
| Punctuation and capitalization | Token classification heads | Narrower than grammar, meaning, and arbitrary list formatting. |
| Local spelling and grammar repair | Contextual edit tagger | Limited edit vocabulary; multiple passes may be needed. |
| Selecting a word or phrase | Candidate ranker with keep/uncertain options | Requires a source of alternatives. |
| Checking Bonsai's proposed edits | Source/context/candidate verifier | Must evaluate meaning preservation as well as grammatical plausibility. |
| Lists and formatting | Classify formatting intent, then deterministic rendering where possible | Arbitrary restructuring may still require generation. |
| Compose, summarize, reply, style-heavy rewriting | Bonsai generation | A fixed-choice classifier does not supply open-ended content. |

For redo, retain app/window/field identity, recent insertion anchors, focus
revalidation, and verified insertion outside model control. A classifier must not
choose an arbitrary target or authorize destructive replacement.

## Model families researched

| Candidate | Mechanism | Finding |
|---|---|---|
| NeMo-style punctuation/case model | Encoder with punctuation and capitalization heads | Fast narrow baseline; insufficient as a general formatter. |
| GECToR and distilled students | Contextual keep/delete/insert/replace/transform tags | Strong candidate architecture for conservative local corrections. Tested weights have noncommercial restrictions. |
| FELIX | Tag/reorder stage plus masked-language-model insertion | More flexible non-autoregressive editing; two stages and substantial training/export work. |
| FastCorrect, FastCorrect 2, SoftCorrect | ASR-specific non-autoregressive correction, candidate selection, and error detection | Relevant to misrecognitions; released examples and ASR interfaces do not establish a drop-in English VoiceFlow replacement. |
| Community CoEdIT-small | Small autoregressive instruction editor | Lowest measured editor RAM, with weaker grammar/fluency scores than the leading alternatives. |
| Laya | Typed contextual choices, scores, and yes/no decisions | Fine-tunable candidate selector/verifier; not benchmarked locally. |
| Small encoder with dedicated token/edit heads | Task-specific detection, selection, and verification | Worth comparing with Laya on identical data; could avoid repeating sentence encoding for each word. |

Primary architecture references:

- [NeMo punctuation/capitalization documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/nlp/punctuation_and_capitalization.html)
- [GECToR implementation and training recipe](https://github.com/grammarly/gector)
- [GECToR distillation experiments](https://www.grammarly.com/blog/engineering/experimenting-with-gector/)
- [FELIX architecture](https://research.google/blog/introducing-felix-flexible-text-editing-through-tagging-and-insertion/) and [implementation](https://github.com/google-research/google-research/tree/master/felix)
- [FastCorrect](https://github.com/microsoft/NeuralSpeech/tree/master/FastCorrect), [FastCorrect 2](https://github.com/microsoft/NeuralSpeech/tree/master/FastCorrect2), and [SoftCorrect](https://github.com/microsoft/NeuralSpeech/tree/master/SoftCorrect)
- [Community CoEdIT-small model card](https://huggingface.co/jbochi/coedit-small)

## Completed benchmark

All seven paths completed 1,000 quality cases. The first 190 cases received three
calls each and the remaining 810 one call: 9,660 calls in total. Quality uses the
first output, never the best repeat. A separate fresh speed pass used 190 cases
in five seeded shuffled rounds across seven paths: 6,650 measured calls.

The suite comprises 43 original authored cases, 357 controlled cases, 200 JFLEG
cases, 200 Cambridge W&I development cases, and 200 synthetic speech clips
transcribed by Parakeet. There are 956 distinct text/context inputs and 411
families. Speech uses four installed voices, with 152 clean clips and 48 clips
with seeded 20 dB Gaussian noise. This is not a real-user dictation study.

### Quality totals

| Path | Exact matches /1,000 | W&I ERRANT F0.5 /100 | JFLEG GLEU /100 | Protected-span failures /129 |
|---|---:|---:|---:|---:|
| Unchanged input | 286 | 0.0 | 38.9 | 0 |
| Existing rules | 223 | 2.1 | 39.2 | 96 |
| Raw punctuation/case | 158 | 19.2 | 41.3 | 96 |
| Guarded punctuation/case | 313 | 5.2 | 39.7 | 0 |
| GECToR | 376 | 49.8 | 51.3 | 0 |
| CoEdIT-small | 396 | 36.7 | 45.4 | 0 |
| Bonsai/native pipeline | 404 | 43.0 | 53.4 | 83 |

Exact agreement is agreement with available references, not semantic accuracy.
W&I and JFLEG scores each cover their selected 200-case subsets, not full public
leaderboards. No weighted composite score was constructed.

Bonsai leads exact agreement and fluency point estimates. GECToR leads grammar
F0.5. Paired family-bootstrap exact differences against Bonsai include zero:
CoEdIT −0.8 percentage points (95% interval −9.55 to +7.45), GECToR −2.8 points
(−12.01 to +5.92). GECToR's W&I F0.5 advantage also has an interval including zero
(−0.16 to +13.73 points). These results do not establish a clear overall winner.

Of 286 inputs already matching an accepted reference, Bonsai moved 140 outside
the reference set, rules 118, GECToR 25, and CoEdIT 29. These are review flags:
reference sets omit legitimate alternatives. Zero observed protected-span
failures is also not a guarantee of general safety.

### Speed, size, and memory

Apple M4 Max, 128 GiB RAM; batch one; normal CPU priority; one benchmark model
loaded at a time. Encoders ran on CPU with four threads. Bonsai ran through the
native pipeline and llama-server/Metal with a 16,384-token context and q8 KV.

| Path | Warm median ms | Warm p95 ms | Weight MiB | Peak process-tree RSS MiB |
|---|---:|---:|---:|---:|
| Unchanged input | <0.01 | <0.01 | — | 28 |
| Existing rules | 0.1 | 0.2 | — | 43 |
| Raw punctuation/case | 21.3 | 70.9 | 199.8 | 862 |
| Guarded punctuation/case | <0.01 | 22.6 | 199.8 | 787 |
| GECToR | 75.9 | 135.2 | 487.9 | 1,660 |
| CoEdIT-small | 91.6 | 216.3 | 293.6 | 743 |
| Bonsai/native pipeline | 270.8 | 808.3 | 1,105.0 | 2,597 |

Guarded punctuation bypassed inference on 820/950 calls. When invoked, its model
took 20.8 ms median and 72.5 ms p95; the near-zero overall median is not neural
inference speed. The model remained loaded.

GECToR's median was about 3.6 times lower than Bonsai's; CoEdIT's about three
times lower. These compare configured pipelines, not equally quantized models
on matched acceleration. GECToR and CoEdIT do not consume Bonsai's contextual
instructions. Changing host load and another resident MLX model affected timing;
all measured rounds were retained. These are formatter-call timings, not full
hotkey-to-paste latency.

| Model | Load/import seconds | First formatter call ms |
|---|---:|---:|
| Punctuation/case | 1.26 | 23.1 |
| GECToR | 6.57 | 33.3 |
| CoEdIT-small | 5.03 | 737.6 |
| Bonsai/native pipeline | 1.08 | 5,446.3 |

Startup includes fresh-process initialization; disk caches were not flushed.
Bonsai has approximately 8.19B parameters quantized to Q1_0, GECToR approximately
128M, and CoEdIT-small approximately 77M. Their weight formats differ. Bonsai's
configured KV buffer alone was 1,224 MiB. RSS includes runtime/startup overhead
and is not an isolated or steady-state model-memory measurement.

The priority experiment initially caused a memory-reserve hold at Bonsai case
983. The final 17 cases completed after the user paused that job. The timing-only
guard subsequently used a 16 GiB estimated available-memory reserve, normal macOS
pressure, and no new swap-outs; the historical quality guard used physical free
memory. No other job was stopped by the benchmark.

### Artifacts and limitations

The benchmark and detailed reports remain in [draft PR #9](https://github.com/Alexander-Ollman/voiceflow/pull/9),
on branch `update/review-open-prs-20261015`. This documentation-only publication
does not merge that PR or its application changes. Evidence at recorded commit
`2a70a9c6c454128257685543a0fa546a50ddacb0`:

- [Detailed expanded report](https://github.com/Alexander-Ollman/voiceflow/blob/2a70a9c6c454128257685543a0fa546a50ddacb0/docs/reviews/expanded-formatting-benchmark-2026-10-15.md)
- [Quality aggregates](https://github.com/Alexander-Ollman/voiceflow/blob/2a70a9c6c454128257685543a0fa546a50ddacb0/benchmarks/formatting/expanded-v2/summary-1000.json)
- [Fresh latency aggregates](https://github.com/Alexander-Ollman/voiceflow/blob/2a70a9c6c454128257685543a0fa546a50ddacb0/benchmarks/formatting/expanded-v2/latency-paused-peer/summary.json)
- [Reproduction instructions](https://github.com/Alexander-Ollman/voiceflow/blob/2a70a9c6c454128257685543a0fa546a50ddacb0/benchmarks/formatting/expanded-v2/README.md)
- [Earlier architecture research](https://github.com/Alexander-Ollman/voiceflow/blob/2a70a9c6c454128257685543a0fa546a50ddacb0/docs/LIGHTWEIGHT_DECISION_MODELS.md)

The original pilot used 63 cases and three repeats; the 190-case checkpoint and
the completed 1,000-case suite supersede it for these comparisons. The constructed
mixture does not estimate production traffic. Confidence intervals used 2,000
seeded family/document bootstrap draws; W&I covers 132 documents. External model
training overlap with public corpora is unknown. Human corpus text, references,
and predictions were retained only in the ignored local cache. Published evidence
is redacted. No private dictation history was used.

## Improving Bonsai later

First fix the lossy normalization shared by rules and the tested native Bonsai
path. Controlled failures include spacing damage to times (`3:30`), amounts
(`$5,000`), URLs, versions (`v2.7.1`), and identifiers (`userID`, `VoiceFlow`).
The 83/129 protected-span failures measure the full pipeline, not isolated LLM
behavior; they must not all be attributed to Bonsai's weights.

Preserve protected spans and existing list structure before regeneration. Then
evaluate contextual prompting, correction restraint, and list rendering against
the frozen suite. Prompt/context sizing, lazy loading, and model unloading are
separate memory experiments with cold-start tradeoffs. No fine-tune is required
to begin addressing normalization defects.

Skipping a generator call saves computation but does not unload its weights or
KV cache. Adding a permanently resident encoder beside Bonsai may increase total
RAM. A future lightweight mode must demonstrate an end-to-end path with Bonsai
actually unloaded and explicitly define the supported formatting behavior.

## Laya and contextual word confidence

Laya provides typed choices, scores, and yes/no decisions, plus supervised
fine-tuning and held-out temperature calibration. At research time its official
documentation listed English/typed-decision checkpoints at 421M parameters and a
multilingual checkpoint at 322M. These are larger parameter counts than the
tested GECToR and CoEdIT-small models; deployment precision and runtime still
determine actual memory. Laya has not been measured on this Mac or this task.
[Official repository](https://github.com/NandhaKishorM/laya),
[fine-tuning documentation](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md).

An example decision is the sentence “Please send there invoices before Friday,”
with alternatives keep `there`, replace with `their`, replace with `they're`, or
uncertain. Contextual selection can choose among those candidates. Candidate
sources could include a dictionary/confusion set, an edit tagger, Bonsai, or ASR
alternatives if exposed by the daemon. The current ASR interface does not supply
an n-best list; do not assume it already provides every useful acoustic feature.

Distinguish three targets:

1. Detection: does a word or span appear incorrect?
2. Selection: which supplied alternative fits the context and intended task?
3. Verification: does a proposed edit improve the text while preserving meaning?

A word's linguistic plausibility is different from confidence that the speaker
said it. Tuesday and Thursday may both fit a sentence. Acoustic evidence, word
confidence, user vocabulary, and surrounding context can help; text alone cannot
recover every intended word. Research on [ELECTRA ASR rescoring/confidence](https://arxiv.org/abs/2110.01857)
and [RED-ACE](https://aclanthology.org/2022.emnlp-main.180/) supports studying
contextual error detection and ASR confidence together, not assuming guaranteed
recovery from grammatical context.

A future verifier could assess Bonsai's changes to words, phrases, punctuation,
and list formatting. Preserve the original when uncertain, or request Bonsai
only when it has not already run. Verification after generation improves control
but does not eliminate the cost of that generation. The existing shadow verifier
is only a feature prototype, not a trained/calibrated decision model.

## Training data and artifact terms

GECToR is a model/training recipe, not one uniquely licensed dataset. Its source
code, checkpoint weights, encoder, and training corpora have distinct terms.
The tested `gotutiyan/gector-roberta-base-5k` checkpoint is noncommercial
research-only; it is an architecture reference, not a shipping recommendation.

| Source | Possible use | Constraint |
|---|---|---|
| Owned or explicitly licensed clean text plus controlled errors | Generate word-choice, punctuation, agreement, and formatting decisions | Include correct inputs and plausible wrong alternatives. |
| Consented audio, Parakeet transcripts, human-corrected references | Learn actual ASR error patterns | Speaker/session splits; explicit consent; no implicit use of private history. |
| Synthetic speech through the actual ASR version | Bootstrap controlled phonetic and noise examples | Does not substitute for real recording diversity. |
| BEA learner corpora: W&I/LOCNESS, FCE, Lang-8, NUCLE | Research grammar supervision | Shared-task page specifies noncommercial use; not unrestricted product-training data. |
| Google C4_200M | Large synthetic correction source | Corruption edits are CC BY 4.0; underlying text rights require separate consideration. |

Dataset references: [BEA corpus descriptions and terms](https://www.cl.cam.ac.uk/research/nl/bea2019st/),
[Google C4_200M repository](https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction).

The community CoEdIT-small card declares Apache-2.0 but has thin provenance;
review the exact artifact before distribution. Official CoEdIT-large uses
CC-BY-NC-4.0. Neither source-code licensing nor a base-model license automatically
establishes the rights for a fine-tuned checkpoint or all its training data.
[Community small model](https://huggingface.co/jbochi/coedit-small),
[official large model](https://huggingface.co/grammarly/coedit-large),
[CoEdIT dataset card](https://huggingface.co/datasets/grammarly/coedit).

Measured checkpoint revisions were punctuation/case
`b26fd1c40e88678859048898218ea4edcc24c84a`, GECToR
`adaac6fb919431fb5a038b1e449055ae638613a4`, and CoEdIT-small
`6ce9822b4ff6e4af86b70f979c890e9e41f04366`; full environment metadata is in the
benchmark artifacts. The punctuation model was
`1-800-BAD-CODE/punctuation_fullstop_truecase_english`.

## Future experiment, if resumed

Convert licensed erroneous/corrected pairs into contextual decisions. Retain
unchanged examples, valid unusual wording, names, numbers, URLs, code, negation,
and ambiguous candidates. Train uncertainty and preservation behavior explicitly.

Compare fine-tuned Laya with a smaller dedicated encoder/head on identical data.
Keep training, calibration, threshold selection, and final evaluation separate.
Split by source document, speaker/session, and template family, not random rows.
Keep the existing 1,000 cases out of training; because they have informed research
choices, add a fresh untouched final test set for stronger confirmation.

Evaluate edit precision/recall, harmful or unnecessary edits, exact protected-span
preservation, meaning and formatting correctness, calibration (Brier score,
reliability plots/ECE), and coverage versus error when abstaining. A predicted
0.95 is not established 95% correctness until measured on relevant held-out data;
temperature scaling does not guarantee calibration under distribution shift.

Measure warm/cold p50/p95, weight size, RSS/Metal memory, generator-call rate, and
end-to-end dictation latency. Run in shadow mode before changing user output.
Validate tokenization, edit reconstruction, native export, and quantization
parity separately. Do not keep both models permanently loaded and infer memory
savings from parameter counts alone.

This experiment is deferred. The current product direction is to improve Bonsai
later, with text preservation as the first prerequisite.
