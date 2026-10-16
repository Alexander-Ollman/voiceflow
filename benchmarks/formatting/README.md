# Local formatting experiment

Exploratory benchmark, not an application dependency or production model switch.
See [results and recommendation](../../docs/reviews/formatting-benchmark-2026-10-15.md).
All inputs are authored or synthetic speech. No dictation history is read.

## Reproduce

Use an isolated Python 3.12 environment. The measurements used the versions in
`requirements.txt`; `environment.json` records actual packages, artifact hashes,
hardware and backend settings. Model revisions and the noncommercial GECToR
restriction are in `models.json`. Downloaded weights stay in the Hugging Face
cache; helper source, audio and logs stay under ignored `VoiceFlowApp/build`.

```sh
python3.12 -m venv VoiceFlowApp/build/formatting-bench/venv
VoiceFlowApp/build/formatting-bench/venv/bin/pip install -r benchmarks/formatting/requirements.txt
VoiceFlowApp/build/formatting-bench/venv/bin/python scripts/formatting_bench/prepare.py
VoiceFlowApp/build/formatting-bench/venv/bin/python scripts/formatting_bench/run.py --backend pcs_guarded --output /tmp/pcs-results.json
```

Run each backend separately and sequentially: `identity`, `rules`, `pcs`,
`pcs_guarded`, `gector`, `coedit`, `bonsai`. The native controls need `--library`
pointing to a built `libvoiceflow_ffi.dylib`. Bonsai also needs an installed
VoiceFlow `llama-server` and the cached `Bonsai-8B-Q1_0.gguf` at the app's normal
model location. It starts its own loopback server on port 18080 and stops only
that child; it refuses an occupied port. Model downloads and reference source
are pinned. No remote Hugging Face model code is enabled.

Use `--fixtures benchmarks/formatting/asr-fixtures.jsonl` for the collected
Parakeet transcripts, and `--repeats 3` for the measured sampling count. The
checked-in results contain inputs, first output, every repeated output, wall
latencies, failures and evaluation flags. Summarize explicit result paths with
`scripts/formatting_bench/summarize.py`; summary/feature JSON files are not raw
benchmark results and should not be passed to it.

Optional synthetic ASR recollection needs macOS Samantha, `say`, `afconvert`,
and a separate Python with `parakeet-mlx` and the pinned model cached:

```sh
VoiceFlowApp/build/formatting-bench/venv/bin/python scripts/formatting_bench/collect_asr.py --python /path/to/parakeet-venv/bin/python
```

This overwrites `asr-fixtures.jsonl`; macOS voice/model changes can change the
corpus. It creates a unique socket and separate PID/log files, never using the
application's service. Generated audio is not committed; SHA256s are recorded.
The original collection resolved the same model revision recorded in
`environment.json`. The default application daemon log path remains unchanged.

## Scoring and adapters

Exact match allows NFKC, whitespace and smart-quote normalization, but preserves
case, punctuation and words. It is reference agreement, not human semantic
quality. Protected spans are evaluation-only, case-sensitive and bounded by
word boundaries. `preserve_changed` compares output to input, independently of
whether an ASR input already differed from the intended sentence. Errors/empty
outputs remain failures. Stability compares all three outputs; exact scoring
uses the first, never the best. Repeats are not independent corpus examples.

Raw PCS lowercases/strips punctuation as required by the checkpoint. Guarded
PCS bypasses already-capitalized terminal-punctuated text and sensitive patterns
(digits, URLs, email, code separators and mixed case), then rejects lexical
changes. It checks the original input before any normalizer. This is a deliberately
limited punctuation prototype, not a grammar detector or calibrated classifier.
Both variants load the same model even when bypassing; a future native/lazy
implementation would be a separate measurement. Raw PCS normalization losses
count as real pipeline failures. GECToR uses the pinned implementation, KEEP
bias .3, error threshold .6, five maximum passes and batch one. No thresholds
were fitted to these fixtures. Transformers 5 compatibility reconstructs the
RoBERTa-base architecture, verifies its legacy position-ID buffer and strictly
loads every learned safetensors weight. CoEdIT uses the author's grammar prefix,
greedy generation and a 128-new-token limit.

Native controls use compiled defaults via an explicit nonexistent config path;
the prompt directory was empty, with no personal replacements. Bonsai runs the
cached native formatter including its production pre/postprocessing. A zero
native LLM duration is treated as a possible silent fallback error. These
measurements identify the binary by hash and do not establish parity with a
fresh Rust build or every user's settings.

RSS samples process-tree memory every 20ms, with boundary samples for short
controls. This includes Python/Torch overhead and the Bonsai server; it is not
isolated model memory or total Metal allocation. Encoders use CPU/four threads;
Bonsai uses Metal and production-like context/KV settings. Warm p95 is the sample
percentile across three calls per example, not a confidence interval. Imports,
model initialization and first inference are separate fields; cold disk caches
were not reset. There was no simultaneous model inference.

## Context verifier prototype

`shadow_verifier.py` accepts source, candidate and context, with no reference or
expected protection annotations. It emits changed word spans, context-word
matches, numeric/negation changes, lost identifiers and list structure features.
It does not accept/reject edits, train a classifier, or estimate confidence.
Context matches are supporting features, never proof of correctness. Rules can
flag legitimate changes, such as converting spelled numbers or contractions.

```sh
python3 scripts/formatting_bench/shadow_verifier.py benchmarks/formatting/results/bonsai.json --output /tmp/shadow-features.json
python3 scripts/formatting_bench/test_metrics.py
python3 scripts/formatting_bench/test_verifier.py
```

Only authored context is exported here. A production shadow mode would need
explicit design for context minimization, labels and aggregate-only telemetry;
this script does not collect application context or audio.
