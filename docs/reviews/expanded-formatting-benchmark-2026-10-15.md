---
title: Expanded Formatting Evaluation
icon: 🔬
order: 33
tags: [benchmark, inference, performance]
---

# Expanded formatting evaluation

## Status

The 190-case stage is complete across seven paths. The expanded suite contains
1,000 cases, and six paths finished all 1,000. Bonsai reached 983/1,000 before the
resource guard stopped it to protect the user’s higher-priority experiment.
**Only 17 speech cases remain. No full1,000-case comparison is claimed yet.**

All800 text cases, including 400 human-annotated grammar examples, have completed
on every path. The archived records contain 9,643 measured formatter calls out of
9,660 planned. Completed cases are checkpointed; resuming does not repeat them.
An interrupted in-flight call is not counted as a completed measurement.

[Current checkpoint](../../benchmarks/formatting/expanded-v2/progress.json),
[reproduction and metrics](../../benchmarks/formatting/expanded-v2/README.md),
[selection and provenance](../../benchmarks/formatting/expanded-v2/selection.json).

## Completed190-case comparison

The fixed prefix contains 43 authored, 37 controlled, 40 JFLEG, 40 Write & Improve,
and 30 synthetic-speech/Parakeet cases. Every example received three timed calls.
Exact matching uses the first output against acceptable references. These are
reference-agreement scores, not human judgments of semantic correctness.

| Path | Exact /190 | Exact95% interval | W&I edit F0.5 /40 cases | JFLEG GLEU /40 cases | Warm case median /p95 ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| Pass through | 61 | 25.7–39.0% | 0.0 | 43.4 | 0.00 /0.00 | 33 |
| Existing rules | 54 | 22.0–35.7% | 0.0 | 44.1 | 0.14 /0.22 | 165 |
| Raw punctuation/case ONNX | 40 | 14.7–28.4% | 14.8 | 45.1 | 22.12 /29.83 | 861 |
| Guarded punctuation | 68 | 28.9–43.1% | 6.1 | 44.0 | 0.00 /23.99 | 820 |
| GECToR | 78 | 33.0–49.0% | 59.0 | 56.3 | 73.34 /119.93 | 1649 |
| CoEdIT-small | 81 | 35.0–50.9% | 46.0 | 47.5 | 90.03 /213.74 | 751 |
| Bonsai/native | 84 | 36.0–52.7% | 52.6 | 53.8 | 115.74 /303.76 | 2731 |

RSS above is the largest completed/held session across both stages, not
190-only memory. Timing quantiles give equal weight to each case’s median;
the raw per-call times are also retained. A near-zero guarded-path median means
it bypassed most inputs, while still loading its model.

Bonsai leads exact reference agreement at 84/190, CoEdIT has 81, and GECToR 78.
Paired family-bootstrap intervals against Bonsai span zero for CoEdIT and
GECToR, so this checkpoint does not establish a clear overall winner. GECToR
has the best W&I edit F0.5 point estimate; its95% interval is44.4–72.3, while
Bonsai’s is36.1–68.8. These intervals also overlap.

ERRANT3.0.2 compares predicted edits against the selected W&I gold M2 edits.
The unchanged identity control has zero false edits. JFLEG uses four human
references and the pinned author’s GLEU implementation. These are selected
subsets, not full-dataset leaderboard results. GLEU reference-sampling intervals
must not be mistaken for uncertainty about a user population.

## Complete800-text checkpoint

This pre-existing source cohort is complete for every model; it is not a
post-hoc selection of successful examples. Full edit/fluency rescoring will run
after memory is available. Exact matches alone can penalize valid alternative
wording, so do not use this table alone to select the grammar model.

| Path | Exact /800 | JFLEG exact /200 | W&I exact /200 |
|---|---:|---:|---:|
| Pass through | 247 | 56 | 45 |
| Existing rules | 187 | 53 | 41 |
| Raw punctuation/case ONNX | 131 | 34 | 34 |
| Guarded punctuation | 274 | 57 | 42 |
| GECToR | 325 | 51 | 57 |
| CoEdIT-small | 339 | 52 | 55 |
| Bonsai/native | 348 | 66 | 49 |

## Are the models comparable?

They are comparable as candidate paths for the same app tasks, with important
differences in capability, representation and execution backend. The experiment
compares their available artifacts, not equally quantized/equally accelerated
versions.

| Candidate | Weight payload | Execution in this test | Scope |
|---|---:|---|---|
| Punctuation/case ONNX |199.8 MiB|CPU/four threads; Python wrapper imports Torch|Punctuation and capitalization|
| CoEdIT-small |293.6 MiB|CPU/four threads; autoregressive T5|Grammar/text editing; about77M stored parameters|
| GECToR |487.9 MiB|CPU/four threads; iterative edit tagging|Local grammar edits; about128M parameters|
| Bonsai |1,105.0 MiB|Metal; native llama-server; Q1_0|General contextual formatting/writing;8.19B parameters|

File size is separate from resident memory. Python/runtime overhead, tokenizers,
activations, iteration and KV caches differ. Bonsai’s16,384-token production-like
context allocates a1,224 MiB Metal KV buffer, in addition to its weights. Thus
reducing context or lazy loading deserves separate measurement before replacing
the generator. No native Core ML conversion, matched quantization experiment or
total-application memory/battery comparison is claimed. GECToR’s tested checkpoint
is noncommercial research-only. Exact artifact hashes are in
[environment metadata](../../benchmarks/formatting/expanded-v2/environment.json).

## Speed and latency

Every recorded formatter call has a wall-clock duration. The first190 cases
have three repeats; the remaining810 have one. Raw arrays, per-call median/p95,
equal-case median/p95, cold initialization, first-call time and session peak
process-tree RSS are retained. The ASR records separately include daemon inference
duration, which is not full audio-request wall time. Neither measure is full
hotkey-to-paste latency.

The other experiment has priority. After the user’s request, our processes ran
at low CPU priority with a one-second resource watcher. Timing therefore reflects
concurrent load; the initial calls also predate that priority adjustment. These
measurements must not be presented as isolated hardware speed rankings. CPU
priority is not a guarantee of exclusive or lower-priority GPU scheduling.

## Memory hold and resumption

The watcher observed normal macOS pressure and no new swap-outs, but physical
free memory dropped below the 16 GiB reserve, to 9.67 GiB at the trigger. It stopped
new scheduling, interrupted our worker, and released our owned llama-server.
Later samples fell below 1 GiB free while the priority test continued, so the
benchmark remains held. The other workload was not stopped or changed.

The remaining17 cases and final rescoring can resume only after sufficient
stable headroom returns. The checkpoints verify input/model/code signatures.
A five-minute low-overhead monitor found no stable recovery. Before restarting,
require 24 GiB free, normal pressure and no new swap-outs for 60 seconds. The
monitor loaded no model, and no automatic restart is scheduled.
The normal reserve guard must remain attached to any resumed run.

## Coverage and remaining limitations

The full selection includes400 human grammar examples, 357 controlled cases,
43 original authored cases, and 200 synthetic speech cases. Speech covers four
installed voices (50 each),152 clean clips and 48 seeded20dB-noise clips. There
are 956 distinct text/context inputs and 411 family clusters; related speech/text
and template variants are grouped, with a deduplicated input view reported.

No model/gate thresholds were tuned on the evaluation. External training overlap
with public corpora is unknown. W&I document grouping is available; JFLEG lacks
document IDs here. Synthetic accents/noise are not a substitute for human
recordings, capture failures or microphone tests. This mixture is deliberately
constructed and does not estimate production traffic. Human corpus text and
predictions remain local; tracked evidence contains IDs, counts and timings.

The context verifier remains a reference-free shadow feature prototype. These
measurements do not train or calibrate its confidence, and exact/gold-reference
mismatches are review signals rather than proof of semantic harm. Production
model defaults, installation and release remain unchanged.
