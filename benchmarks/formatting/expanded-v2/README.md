# Expanded formatter evaluation

Two nested stages: **190**, then **1,000** frozen cases. The complete corpus has
43 original authored text cases, 357 controlled cases, 200 JFLEG cases (100 dev,
100 test), 200 Cambridge Write & Improve development cases (70 A, 65 B, 65 C),
and 200 synthetic-speech/Parakeet cases. No training split or LOCNESS text is
sampled. The exact IDs, source revision and archive hash are in `selection.json`.
Selections, task definitions and model settings were frozen before scoring.
External checkpoint training/tuning overlap with these public corpora is unknown;
“frozen local evaluation” does not mean uncontaminated model pretraining.

The 190-case prefix contains 43 original, 37 controlled, 40 JFLEG, 40 W&I, and
30 speech cases. It is a checkpoint, not a development set for tuning the later
810. The first190 receive three measured calls each; the remaining810 receive
one. Exact scores use the first output, never a majority vote or best result.
All seven paths run at190 before any path resumes to1000.

Speech uses Samantha (US), Daniel (UK), Karen (Australia), and Rishi (India),
50 cases each. There are152 clean examples and 48 with seeded Gaussian noise at
20dB relative to speech-active RMS, plus250ms silence at each end. This tests
synthetic voices/noise, not human accents or microphone capture. Audio hashes,
ASR timing, clipping counts and original spoken protections stay in local raw
records. These audio cases derive from text cases; they are not independent new
utterance content. The full suite has956 distinct text+context inputs and 411
family clusters. Aggregate results include a deduplicated text/context view.

## Research data and reproduction

W&I is licensed for noncommercial research/education and limits redistribution;
LOCNESS is excluded. We do not publish human dataset text, reference spans,
context, predictions or excerpts here. The local raw corpus/results live under
ignored `VoiceFlowApp/build/formatting-bench/expanded-v2`. Tracked evidence is
aggregate scores, selection metadata and redacted per-case measurements.
JFLEG human text/predictions are also kept local. The models and experiment are
not application runtime dependencies. GECToR weights remain research-only.

Sources and required acknowledgement:

- [JFLEG, Napoles, Sakaguchi & Tetreault (EACL2017)](https://github.com/keisks/jfleg),
  pinned revision in the manifest; includes four human references per example.
- [Cambridge English Write & Improve / BEA2019 release](https://www.cl.cam.ac.uk/research/nl/bea2019st/):
  Yannakoudakis, Andersen, Geranpayeh, Briscoe & Nicholls (2018), *Developing an
  automated writing placement system for ESL learners*, Applied Measurement in
  Education31(3),251–267. See the archive's `licence.wi.txt` before separate use.
- [ERRANT](https://github.com/chrisjbryant/errant), version3.0.2, with spaCy3.8.16
  and `en_core_web_sm`3.8.0. Scores are on the selected development subset,
  not the official full BEA2019 test or version2 leaderboard configuration.

Use the existing isolated benchmark Python3.12 environment and pinned model
preparation. Install this folder's `requirements.txt` for `errant==3.0.2`, `sacremoses==0.1.1`, `scipy==1.17.0`,
`spacy==3.8.16` and the `en_core_web_sm`3.8.0 wheel. Do not install these into the
application's environment. Corpus construction refuses to overwrite an existing
frozen corpus. It downloads the named research datasets; it reads no private
history. The collector resumes200 speech cases using an isolated daemon.

```sh
# Use the benchmark venv's Python for all commands below.
python scripts/formatting_bench/expand.py
python scripts/formatting_bench/collect_expanded_asr.py --python /path/to/parakeet-venv/bin/python
python scripts/formatting_bench/expanded_suite.py --library /path/to/libvoiceflow_ffi.dylib
```

Workers fsync one completed case per line. Resume verifies corpus, adapter,
worker and native-library signatures, along with pinned model metadata, and
requires an exact prefix of completed IDs. An interrupted partial final line
is discarded; completed cases are not rerun. A model session's peak memory and
cold load are separate from warm per-case measurements.

## Metrics and limits

Report exact-reference agreement, family-cluster bootstrap95% intervals,
paired exact differences against Bonsai, document-cluster bootstrap intervals
for W&I F0.5 and paired F0.5 differences, sensitive-span failures, errors, output
stability and off-reference changes to already-acceptable inputs. Intervals use2,000 seeded
cluster resamples. Controlled variants share a template family; related speech
and text share their family; W&I uses matched document IDs. JFLEG lacks document
IDs here, so its sentence clusters cannot establish document/speaker
independence. The synthetic mixture is deliberately constructed, not an estimate
of user traffic, and no interval certifies semantic correctness.

On W&I, predictions are tokenized by spaCy and aligned to the original gold
source tokens; the installed ERRANT implementation extracts predicted edits and
compares against official source M2 gold edits. Report edit precision, recall
and F0.5, including false edits and misses. On JFLEG, use the pinned author's
GLEU script with four references and500 reference samples. Its returned interval
concerns reference sampling, not held-out corpus or user-population uncertainty.
Neither subset score should be compared directly with full-dataset leaderboards.

Source/category, speech voice/noise, acceptable-input off-reference changes and deduplicated
views remain separate. GECToR input-window and CoEdIT generation-limit warnings
are counted rather than dropping difficult examples. These cases are evaluation
only; no calibrated verifier or model is trained on them.

## Resource priority

Another test has priority. The suite now starts its memory watcher before any
worker and refuses unguarded execution. It runs one model at a time at low CPU
priority. `watch_resources.py --suite-pid PID --resume` samples macOS pressure,
physical free memory and swap-outs every second. It stops our scheduling,
interrupts only this benchmark's workers and releases owned descendants if
pressure is non-normal, physical free memory falls below 16GiB, or swap-outs
increase. Completed checkpoints remain available; there is no automatic restart
under pressure. It never stops the priority workload. Resource samples remain
local. Concurrent work means the recorded latency is operational evidence, not
an isolated hardware-performance comparison.

Off-reference changes and gold-edit false positives are review flags, not proof of
semantic harm: human reference sets do not enumerate every valid correction.
Identity must have zero false edits under the W&I tokenization/scoring adapter;
the scorer refuses to publish if that control fails. Per-call latency quantiles
weight the190 repeated cases three times. Separate quantiles over each case's
median give equal weight to all cases. Neither is time-to-paste latency.

## Current checkpoint resumption

The existing raw corpus/checkpoints are in the separate `voiceflow-pr-update`
worktree's ignored build directory. Resume there after stable headroom returns;
rebuilding a fresh corpus elsewhere does not recover those completed calls.
From the original `voiceflow` checkout:

```sh
cd ../voiceflow-pr-update
VoiceFlowApp/build/formatting-bench/venv/bin/python scripts/formatting_bench/expanded_suite.py --library ../voiceflow/target/release/libvoiceflow_ffi.dylib
```

The suite automatically attaches the resource watcher before loading anything.
Do not lower the reserve simply to finish the remaining17 cases. The190-case
files are complete, the800-text exact checkpoint is complete, and final1,000-case
scoring remains pending. Source/reference text and predictions stay local.
