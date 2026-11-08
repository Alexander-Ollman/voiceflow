---
title: Expanded Formatting Evaluation
icon: 🔬
order: 33
tags: [benchmark, inference, performance]
---

# Expanded formatting evaluation

Completed October 15–16, 2026. **All seven paths finished 1,000 cases.** The quality
evaluation records 9,660 formatter calls. A separate timing rerun, after the user
paused the competing experiment, adds **6,650 measured calls**: 190 cases × five
shuffled rounds × seven paths. No recorded inference errors or empty quality
outputs occurred. Model defaults and the application runtime are unchanged.

[Reproduction and limits](../../benchmarks/formatting/expanded-v2/README.md),
[full quality results](../../benchmarks/formatting/expanded-v2/summary-1000.json),
[fresh timing results](../../benchmarks/formatting/expanded-v2/latency-paused-peer/summary.json),
[checkpoint status](../../benchmarks/formatting/expanded-v2/progress.json).

## Fresh speed and memory results

Same Apple M4 Max with 128 GiB RAM; batch one, normal CPU priority, one model loaded at a
time. Encoders use CPU/four threads; Bonsai uses native llama-server/Metal with
16,384-token context and q8_0 KV. Every path sees the same 190 inputs in the same
seeded shuffle within each round. The table includes every measured call; slow
calls are not discarded. The models have different scopes and execution formats.

| Path | Median ms | p95 ms | Weight MiB | Peak process-tree RSS MiB |
|---|---:|---:|---:|---:|
| Pass through | <0.01 | <0.01 | — | 28 |
| Existing rules | 0.1 | 0.2 | — | 43 |
| Punctuation/case ONNX | 21.3 | 70.9 | 199.8 | 862 |
| Guarded punctuation | <0.01 | 22.6 | 199.8 | 787 |
| GECToR | 75.9 | 135.2 | 487.9 | 1,660 |
| CoEdIT-small | 91.6 | 216.3 | 293.6 | 743 |
| Bonsai/native | 270.8 | 808.3 | 1,105.0 | 2,597 |

Bonsai’s measured warm median was 271 ms and p95 was 808 ms. GECToR was 76/135 ms,
CoEdIT 92/216 ms, and raw punctuation restoration 21/71 ms. Those are **configured
pipeline** timings, not a claim that one architecture is intrinsically faster.
Punctuation restoration solves a narrower problem, while CoEdIT/GECToR do not
consume the contextual instructions that the native Bonsai formatter receives.

Guarded punctuation actually invoked its model on 130/950 calls; the rest bypassed
inference. Its model-invocation-only median/p95 were 20.8/72.5ms. The near-zero
overall median must not be presented as punctuation inference speed. It still
loaded the model, so its memory cost remains.

Bonsai’s per-round medians ranged 243–349ms and p95 values ranged 520–1,252ms.
Its memory environment changed during the pass, and another large MLX model
remained resident for part of it. All rounds remain in the primary result.
Background work, thermal state, prompt caching and backend differences prevent
a claim of isolated hardware performance, even with the named experiment paused.

These are full formatter-call wall latencies, including tokenization, request
and text processing where those belong to the adapter. They are **not full
hotkey-to-paste latency**. ASR records separately contain daemon inference time;
that does not include every audio-request or UI stage.

### Startup is separate from warmed calls

Three priming calls precede measured rounds. The first call below is captured
separately; the following two prime additional inputs. Load includes imports,
weight initialization and server readiness. File caches were not flushed, so
these are fresh-process startup measurements, not cold-disk benchmarks.

| Model | Load/import seconds | First formatter call ms |
|---|---:|---:|
| Punctuation/case ONNX | 1.26 | 23.1 |
| GECToR | 6.57 | 33.3 |
| CoEdIT-small | 5.03 | 737.6 |
| Bonsai/native | 1.08 | 5446.3 |

Weight payload is not resident memory. Bonsai is an 8.19B-parameter Q1_0 model;
CoEdIT-small has about 77M stored parameters and GECToR about 128M, in different
representations. Bonsai’s production-like context allocates a 1,224 MiB Metal KV
buffer in addition to its weights. Encoder RSS includes Python/Torch, even for
the ONNX wrapper; peak RSS includes startup and is not steady-state or isolated
model memory. Native Core ML/ONNX conversion and matched quantization have not
been measured. The tested GECToR checkpoint is noncommercial research-only.

## Completed1,000-case quality comparison

The frozen suite contains 43 original authored text cases, 357 controlled cases,
200 JFLEG cases, 200 Cambridge W&I development cases and 200 synthetic Parakeet
transcripts. It contains 956 distinct text/context inputs and 411 family clusters.
The first 190 cases received three calls each; the remaining 810 received one.
Quality always uses the first output, never the best repeat.

Exact matches mean agreement with the available acceptable references. They are
not human semantic accuracy. W&I uses ERRANT3.0.2 against gold M2 edits; JFLEG uses
the pinned author’s GLEU implementation and four human references. Scores below
use0–100 scales for F0.5/GLEU. These are selected subsets, not leaderboard results.

| Path | Exact /1,000 | Exact family-bootstrap95% interval | W&I F0.5 /200 | JFLEG GLEU /200 | Sensitive failures /129 |
|---|---:|---:|---:|---:|---:|
| Pass through | 286 | 23.1–34.6% | 0.0 | 38.9 | 0 |
| Existing rules | 223 | 17.0–28.1% | 2.1 | 39.2 | 96 |
| Punctuation/case ONNX | 158 | 11.7–20.6% | 19.2 | 41.3 | 96 |
| Guarded punctuation | 313 | 25.5–37.2% | 5.2 | 39.7 | 0 |
| GECToR | 376 | 31.0–43.9% | 49.8 | 51.3 | 0 |
| CoEdIT-small | 396 | 33.0–46.0% | 36.7 | 45.4 | 0 |
| Bonsai/native | 404 | 33.6–47.0% | 43.0 | 53.4 | 83 |

Bonsai leads overall exact matches 404 to 396 for CoEdIT and 376 for GECToR.
Paired family-bootstrap intervals for their exact differences against Bonsai
include zero, so this does not establish a clear overall winner. The deduplicated
956-input view preserves the same ordering.

GECToR has the best W&I edit F0.5 point estimate 49.8 versus Bonsai 43.0. Its paired
document-bootstrap difference interval is−0.16 to+13.73 percentage points, still
including zero. Bonsai leads the JFLEG fluency metric 53.4 versus GECToR 51.3 and
CoEdIT 45.4. W&I edit precision/recall are 59.4/30.2 for GECToR, 51.0/26.4 for Bonsai
and 47.2/19.4 for CoEdIT. A model’s edit confidence threshold is not a calibrated
probability that its edits are correct.

### Preservation remains a prerequisite fix

Of 286 inputs already matching an accepted reference, Bonsai moved 140 off that
reference set, rules 118, GECToR 25 and CoEdIT 29. Such changes are review flags,
not automatic proof of semantic harm: reference sets omit valid alternatives.
However, known controlled regressions include spacing damage to times, amounts,
URLs, versions and mixed-case identifiers in the cached native preprocessing
path. Sensitive-span counts above require exact preservation and expose this
risk independently of aggregate grammar scores.

Prioritize protecting those spans and existing list content before freely
normalizing or regenerating. A small verifier can then score changes against
source/context and abstain. The current verifier remains a shadow feature
prototype; this evaluation does not train or calibrate it. Do not swap in the
punctuation model as a full formatter or add another permanently resident model
and assume the application becomes lighter.

## Coverage, uncertainty and reproducibility

The nested190 checkpoint is retained separately, without tuning gates on it.
It contained 43 original, 37 controlled, 40 JFLEG, 40 W&I and 30 speech cases. Bonsai
led exact matches 84/190, followed by CoEdIT 81 and GECToR 78. The full evaluation
confirms that the task-specific tradeoffs persist beyond that small pilot.

The 200 speech cases use four installed voices, 50 each: Samantha, Daniel, Karen
and Rishi. There are 152 clean clips and 48 with seeded20dB Gaussian noise, plus
250ms silence at each end. They test synthetic speech, not human accents, real
microphones, hotkey capture failures or the reported beachball. Derived audio
and template variants are related and remain grouped for uncertainty estimates.

Intervals use 2,000 seeded family/document bootstrap draws. W&I covers 132
documents; JFLEG lacks document IDs here, so its sentence clusters cannot certify
speaker/document independence. GLEU’s reference-sampling interval is not corpus
or user-population uncertainty. The deliberately constructed mixture does not
estimate production traffic. External checkpoint training overlap with these
public corpora remains unknown. No evaluation threshold was fitted to this set.

Human corpus text, references, context and predictions stay in the ignored local
cache under the separate review worktree. Tracked evidence includes selection
IDs, hashes, redacted per-case measurements, official-scorer counts and aggregates.
All 200 reconstructed W&I references were checked against gold annotation order,
including multiple insertions at a single boundary. No corpus data changed.

## Resource handling and completion

The earlier priority experiment caused a 16 GiB physical-free reserve hold after
Bonsai 983/1,000. Completed calls were retained. After the user paused that job,
the remaining 17 cases and both final scoring stages completed successfully.

The fresh speed pass initially retained that physical-free guard. It admitted
the CPU timing runs, then stopped Bonsai during startup before any measured
Bonsai call. Inspection showed normal macOS pressure, zero new swap-outs and
roughly 34 GiB estimated available memory despite few physically unused pages.
For latency-only runs the guard now uses psutil’s macOS available estimate
(inactive + free), with the same 16 GiB reserve, normal-pressure and zero-new-swap
checks. The historical quality guard retains its physical-free policy.

Completed timing files were verified and retained; Bonsai and the rules control
then completed normally. The initial admission stop contributed no timing
samples. The benchmark changed no other job and stopped its owned model processes
at completion. Production model defaults, installation, merging and release
remain unchanged.
