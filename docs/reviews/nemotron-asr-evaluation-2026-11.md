---
title: Nemotron ASR Evaluation on Apple Silicon
icon: 🎙️
order: 33
tags: [inference, benchmark, asr]
---

# Nemotron ASR evaluation on Apple Silicon (2026-11-07)

This was exploration only. No production Swift code, daemon code or app defaults changed.
The harness is in `scripts/asr_eval/`. Audio, venvs and raw results are under the ignored
`VoiceFlowApp/build/asr-eval/`.

## Decision (2026-11-07)

**Keep Parakeet TDT 0.6B v2. Drop the model swap, including the overlay-only
streaming-en prototype.** Neither candidate improves English dictation:
- Both are worse on real read speech (2.3% WER for Parakeet; 3.4% for Unified; 4.3% for
  streaming-en).
- Both give weaker dictation-ready text: spoken-form numbers, missing ending punctuation,
  garbled short commands.
- Streaming would save only 25–90 ms over Parakeet's existing 35–117 ms post-release
  latency.
- Running streaming-en for the overlay would add about 2.5 GB resident.
- streaming-en's NVIDIA Open Model License is materially more onerous than Parakeet's
  CC-BY-4.0.

No implementation work is scheduled. Revisit only if one of these holds:
- **Multilingual dictation becomes a requirement.** Start from Unified, and expect to
  need a released mlx-audio.
- **A real-voice dictation evaluation shows the gap closing.** This evaluation used only
  synthetic voices and 73 read clips.
- **A future checkpoint fixes number formatting and punctuation.**

Any revisit must begin with real-voice dictation data, not a prototype.

## Recommendation (as evaluated)

**Do not replace Parakeet TDT 0.6B v2.** Neither candidate is clearly more accurate for
English dictation. Both are worse on the one real-speech set we have (LibriSpeech). Both
also give weaker dictation-ready text.

- **`nemotron-speech-streaming-en-0.6b`: prototype further, but only for a narrow purpose.**
  It runs well on the Mac through MLX (mlx-audio 0.6.4 from PyPI) and does true cache-aware
  streaming. Each 160 ms chunk takes about 17 ms to process, and the final text is ready
  about 20 ms after the audio ends. That makes it a strong engine for the **live overlay
  preview**. It is not ready to produce the final text:
  - LibriSpeech normalized WER is 4.3% offline and 5.6% at 160 ms, against 2.3% for
    Parakeet v2.
  - It writes numbers in spoken form ("five thousand dollars", "three thirty PM"), so it
    has no inverse text normalization.
  - Ending punctuation is often missing at low latency: 57% of clips end in `.?!` at
    160 ms, against 100% for Parakeet.
  - Parakeet already returns the final text 35–117 ms after release, so streaming saves
    only about 25–90 ms.
- **`nemotron-labs-unified-asr-0.6b`: reject as an English replacement.** It works through
  unreleased mlx-audio git main (ecd43b9). Its English accuracy is within noise of
  Parakeet, but it is worse on LibriSpeech (3.4% vs 2.3%). It drops ending punctuation on
  26% of clips and mangles some short commands ("Delete that." → "Delitas"). Its offline
  pass is the fastest measured (29 ms p50 for 2–5 s clips) and it uses about 1.3 GB.
  Revisit only if multilingual dictation becomes a requirement. It is a "labs" release.

Any prototype must first pass a **real-voice dictation evaluation**. Everything below uses
synthetic `say` voices plus 73 read-speech clips.

## Hardware and software

- Mac17,13, Apple M5 Pro (18 cores), 48 GB, macOS 27.2 (26B8125). Another session was
  building UI on the machine at the same time. Memory pressure stayed at 89–93% free.
- Python 3.12.16 (uv-managed, under `VoiceFlowApp/build/asr-eval/`). No Homebrew changes.
- MLX 0.32.4 in all MLX venvs.
  - parakeet-mlx **0.5.1**, the version production pins in `build-parakeet-daemon.sh`.
  - mlx-audio **0.6.4** (PyPI).
  - mlx-audio **git ecd43b9** (2026-11-07), needed for Unified.
- Transformers 5.22.2 + torch 2.15.0 (MPS), used for the reference parity run.
- Model revisions:
  - `mlx-community/parakeet-tdt-0.6b-v2@8ae1553`, the same revision the benchmark
    scripts pin.
  - `nvidia/nemotron-speech-streaming-en-0.6b@ebe59e5` (the native F32 `model.safetensors`;
    the `.nemo` and `.gguf` files were not downloaded).
  - `mlx-community/nemotron-labs-unified-asr-0.6b-bf16@bd8075b`.

## 1. Mac runtime paths

| Path | streaming-en | Unified |
|---|---|---|
| parakeet-mlx 0.5.1 / 0.7.0 (latest) | ❌ The 0.7.0 source has no causal convolution, no causal downsampling and no prompt kernel. The animaslabs MLX card also marks parakeet-mlx as unsupported for cache-aware models. | ❌ Same: no language-prompt kernel. |
| **mlx-audio 0.6.4 (PyPI)** | ✅ **Loads NVIDIA's native safetensors directly.** Supports offline and cache-aware streaming at 80/160/560/1120 ms. This is what was measured. | ❌ `NotImplementedError: conv_norm_type=batch_norm`; streaming also needs `normalize=NA` |
| **mlx-audio git main ecd43b9** | ✅ (not re-measured) | ✅ **Offline (full context) and NeMo-style buffered streaming**, 160–2080 ms. This is what was measured. |
| Transformers 5.22.2 on MPS | ✅ Offline works on macOS even though the card lists Linux only. Needed `librosa`. Used as the parity reference. | n/a (`.nemo` only) |
| Community MLX conversions | animaslabs fp32/8-bit/4-bit; lightsofapollo 8-bit (not tested) | mlx-community bf16 (tested), 8-bit, 4-bit |
| GGUF (transcribe.cpp / NeMo-Speech.cpp) | Not built: `cmake` is not installed and adding it would need Homebrew. handy-computer's card reports Metal RTF 216× on an M4 Max and LibriSpeech test-clean 2.31% at q8_0. | none found |
| Core ML / ANE (FluidInference etc.) | Exists (int8 encoder, 4 chunk sizes); not tested | Exists (fp16/int8, ANE); not tested |
| NeMo on CPU/MPS | Not attempted, because MLX worked | Not attempted |

**Parity check.** I compared mlx-audio's streaming-en against the Transformers reference
on all 217 clips:
- LibriSpeech errors were identical: 50 errors in 1,169 words for both.
- 11 clips differ, almost all in punctuation.

So the accuracy gaps below belong to the model, not to the port. I did not check Unified's
bf16 MLX conversion against NeMo locally. Its card claims token parity with NeMo 3.1.

## 2. Accuracy vs Parakeet v2

**Corpus** (`scripts/asr_eval/corpus.py`, manifest sha256 `4cae21f6…`), 217 clips:
- **144 synthetic dictation clips.** 36 authored sentences, each with an exact written
  reference. Six `say` voices (US, GB, AU, IE, IN, ZA), three voices per sentence.
  - One voice per sentence also has a copy with 20 dB white noise.
  - Silence pad of 0.25 s at each end.
  - Categories: 48 short clips (<2 s of speech), 56 medium (2.4–5 s), 16 numeric and
    24 long (10–15 s).
- **73 real read-speech clips** from `hf-internal-testing/librispeech_asr_dummy`
  (validation-clean). Their references are unpunctuated, so only normalized WER applies
  to them.

The repo's `asr-fixtures.jsonl` stores hashes but no audio, so the clips were regenerated
the same way `collect_asr.py` makes them.

**Metrics** (corpus-level WER):
- **norm**: Whisper EnglishTextNormalizer, as used on the Open ASR leaderboard.
- **PnC**: raw strings, where case and punctuation count.
- The **numeric** category is dominated by formatting differences ("3.30 p.m." vs
  "3:30 PM") that the normalizer does not merge. Read it as a formatting signal, not word
  accuracy.

| Run | All norm | TTS norm / PnC | LibriSpeech norm | Short (<2 s) norm / PnC | Long norm / PnC | Noisy norm / PnC | TTS ends in `.?!` |
|---|---|---|---|---|---|---|---|
| **Parakeet v2 (batch)** | 2.29 | 2.28 / **3.67** | **2.31** | 6.45 / **11.29** | 1.00 / **2.90** | 2.40 / **4.82** | **100%** |
| streaming-en offline (1120 ctx) | 2.43 | 1.14 / 6.57 | 4.28 | 5.65 / 18.55 | 0.11 / 3.79 | 1.44 / 7.71 | 92% |
| streaming-en, Transformers MPS (reference) | 2.40 | 1.08 / 6.02 | 4.28 | 4.84 / 13.71 | 0.11 / 3.68 | 1.44 / 7.47 | 96% |
| streaming-en stream 1120 ms | **2.22** | **1.02** / 6.93 | 3.93 | **3.23** / 17.74 | 0.11 / 3.79 | 1.92 / 7.71 | 85% |
| streaming-en stream 560 ms | 3.10 | 1.44 / 7.77 | 5.47 | 5.65 / 20.16 | 0.22 / 3.91 | 1.92 / 8.43 | 78% |
| streaming-en stream 160 ms | 3.24 | 1.62 / 9.34 | 5.56 | 5.65 / 23.39 | 0.67 / 4.91 | 2.40 / 10.36 | 57% |
| Unified offline (bf16, en prompt) | 2.68 | 2.16 / 5.48 | 3.42 | 4.84 / 15.32 | 0.78 / 4.58 | 2.88 / 6.75 | 74% |
| Unified buffered 1120 ms | 2.71 | 2.34 / 5.72 | 3.25 | 4.84 / 16.13 | 1.00 / 4.80 | 3.12 / 7.95 | 65% |
| Unified buffered 560 ms | 3.17 | 2.76 / 6.20 | 3.76 | 6.45 / 17.74 | 1.12 / 5.36 | 3.12 / 7.95 | 65% |
| Unified buffered 160 ms | 5.11 | 5.22 / 9.28 | 4.96 | 8.06 / 21.77 | 2.57 / 6.81 | 7.67 / 13.01 | 60% |

Raw error counts (normalized) are small, so treat differences of a few points as noise.

| Run | TTS without numeric clips | LibriSpeech |
|---|---|---|
| Parakeet v2 | 19 / 1,552 | 27 / 1,169 |
| streaming-en offline | 11 / 1,552 | 50 / 1,169 |
| streaming-en 1120 ms | 7 / 1,552 | 46 / 1,169 |
| streaming-en 560 ms | 12 / 1,552 | 64 / 1,169 |
| streaming-en 160 ms | 15 / 1,552 | 65 / 1,169 |
| Unified offline | 14 / 1,552 | 40 / 1,169 |
| Unified 1120 ms | 16 / 1,552 | 38 / 1,169 |
| Unified 560 ms | 19 / 1,552 | 44 / 1,169 |
| Unified 160 ms | 48 / 1,552 | 58 / 1,169 |

What the errors look like:

- **Parakeet v2** mostly fails on short clips and accents:
  - "Yes." → "Yeah." (×3)
  - "Ship it." → "She did." / "Shook it."
  - "My flight" → "My flower it" (Rishi voice)
  - It writes numbers in written form, which suits dictation.
- **streaming-en** gets the words right more often on synthetic speech, and it gets
  "Yes." right. But:
  - It writes **every number in spoken form**: "$5,000" becomes "five thousand dollars",
    "250" becomes "two hundred and fifty", "14th" becomes "fourteenth". VoiceFlow would need
    inverse text normalization or would have to depend on the formatter.
  - Ending punctuation drops as latency falls.
  - It has short-clip failures too: "Ship it." → "It.", "Got it, thanks." → "Goddard thanks."
- **Unified** writes numbers in written form, but:
  - It often leaves out the final period and joins sentences with commas ("…the draft,
    I read through it…").
  - It mangles short commands: "Delete that." → "Delitas"; "Ship it." → "Shebid" /
    "Shivage".

## 3. Latency and memory on this Mac

Batch latency is post-release time from in-memory audio to text. Each TTS clip was timed 3×
and the per-clip median was used. The model is warm and the disk cache is warm. Streaming
latency is the time from the end of the audio (hotkey release) to the final text, with
80 ms mic packets processed as they arrive.

| Run | Load (warm disk) | Short <2 s p50/p95 | 2–5 s p50/p95 | 10–15 s p50/p95 | Peak RSS | MLX peak |
|---|---|---|---|---|---|---|
| **Parakeet v2 batch** (parakeet-mlx, bf16 in memory) | 0.64 s | 35 / 39 ms | 46 / 53 ms | 105 / 117 ms | 0.93 GB | 2.5 GB* |
| streaming-en offline batch (fp32) | 2.1 s | 42 / 44 ms | 90 / 112 ms | 292 / 338 ms | 2.5 GB | 2.5 GB |
| streaming-en Transformers MPS (fp32) | 3.3 s | 45 / 120 ms | 71 / 142 ms | 197 / 307 ms | 1.2 GB† | n/a |
| Unified offline batch (bf16) | 2.1 s | **18 / 20 ms** | **29 / 34 ms** | **80 / 86 ms** | 1.3 GB | 1.7 GB |
| **streaming-en stream 160 ms**, time to final text | 0.8 s | 18 / 29 ms | 18 / 28 ms | 23 / 29 ms | 2.5 GB | 2.4 GB |
| streaming-en stream 560 ms, time to final text | 1.0 s | 19 / 31 ms | 18 / 25 ms | 19 / 20 ms | 2.5 GB | 2.4 GB |
| streaming-en stream 1120 ms, time to final text | 0.6 s | 20 / 23 ms | 20 / 23 ms | 21 / 24 ms | 2.5 GB | 2.5 GB |
| Unified buffered 160 ms, time to final text | 0.5 s | 12 / 13 ms | 14 / 15 ms | 18 / 18 ms | 1.3 GB | 1.3 GB |
| Unified buffered 560 ms, time to final text | 0.8 s | 14 / 15 ms | 15 / 17 ms | 19 / 20 ms | 1.3 GB | 1.3 GB |
| Unified buffered 1120 ms, time to final text | 0.8 s | 16 / 17 ms | 17 / 19 ms | 21 / 22 ms | 1.3 GB | 1.3 GB |

\* Parakeet's MLX peak comes from the 29 s LibriSpeech clip under full attention. It holds
about 1.2 GB when idle.
† RSS does not include MPS allocations.

**Per-chunk compute during streaming** (p50 / p95; the model keeps up when compute is
below the chunk length):

| Model and setting | p50 / p95 per chunk | Real-time factor at p95 |
|---|---|---|
| streaming-en 160 ms | 17.2 / 18.5 ms | 0.12 |
| streaming-en 560 ms | 20.0 / 21.8 ms | 0.04 |
| streaming-en 1120 ms | 23.8 / 25.7 ms | 0.02 |
| Unified 160 ms (80 ms chunks, re-encodes the 5.6 s window each step) | 14.6 / 26.7 ms | 0.33 |

The Unified 160 ms setting had one 345 ms spike while the window was still filling.

**Cold-start notes:**
- The very first Parakeet process after download took 5.1 s to load. Its first call took
  2.2 s, which includes Metal kernel compilation.
- A second process loaded in 0.64 s, and its first call took 53 ms.
- First calls for the Nemotron models took about 0.9–1.0 s.
- I did not purge the page cache, so true cold-boot load times were not measured.

**bf16 cast after load:** casting streaming-en to bf16 at runtime did not help. Peak RSS
stayed at 2.46 GB because the fp32 load spike still happens, and per-chunk compute got
*slower* (32 ms). Reaching Parakeet-like memory needs a checkpoint pre-converted to bf16
with bf16 activations. That was not measured.

## 4. Integration fit

**Can streaming-en's live text be the final text?** Technically yes:
- At 160 ms the overlay updates every 160 ms.
- The final text is ready about 20 ms after release, so the batch pass could be removed.

The blockers are quality, not plumbing:
- Real-speech WER is about 2× Parakeet's.
- Numbers come out in spoken form.
- Ending punctuation is missing on many clips.

The 1120 ms setting is the best compromise: 3.9% on LibriSpeech, 85% of clips ending in
`.?!`. It still updates the overlay about once a second, which is the same cadence as
today's 1 s chunks.

**Daemon protocol sketch.** The existing `stream_*` commands nearly fit:
- `stream_open {sample_rate, delay_ms}` maps to `create_streaming_session(transcription_delay_ms=…)`.
- `stream_feed`: queue the PCM, then step until the queued audio is used up. Return
  `{finalized: <all text so far>, draft: ""}`. RNN-T output is append-only, so the draft
  field becomes unnecessary.
- `stream_close` → `session.close()`, then step until `done`. Return a **new `text`
  field** holding the final transcript. Swift would use it in place of the batch
  `transcribe` call, perhaps behind a setting with the batch path as fallback.
- Swift change: `ParakeetStreamingEngine` keeps 1 s chunks and silently drops a trailing
  partial chunk in `endSession`. It would need to send 80–160 ms packets and flush the
  remainder before `stream_close`.
- The mlx-audio session is single-consumer, and its stepping relies on private attributes
  (`_queued`, `_encoder`). A daemon should hold one session lock and wrap this in a small
  adapter.

**Packaging:**
- Weights:
  - streaming-en's native checkpoint is 2.47 GB fp32, the same size as Parakeet v2's MLX
    repo.
  - Unified bf16 is 1.25 GB.
  - Both are downloaded at first launch, like Parakeet today, so the app bundle does not
    grow.
- Running streaming-en for the overlay *next to* Parakeet for the final text would add
  about 2.5 GB resident (or about 1.2 GB with a pre-converted bf16 checkpoint). This is
  the main cost of the "overlay only" idea.
- `mlx-audio` as a dependency brings in transformers (57 MB), scipy (71 MB), sounddevice,
  miniaudio and other packages that parakeet-mlx does not need.
- Vendoring only `mlx_audio/stt/models/nemotron_asr` would avoid that. It is about 2.9k
  lines, MIT-licensed, and depends on mlx/numpy plus a few mlx_audio audio/alignment
  helpers.
- Unified also needs an unreleased git commit, so pin it or vendor it.
- PyInstaller `--collect-all mlx_audio` would pull in all 30+ STT model packages, so
  vendoring is cleaner.

**Licensing** (summary of the license texts, not legal advice):
- **Parakeet v2 (today): CC-BY-4.0.**
- **streaming-en: NVIDIA Open Model License** (version of 2025-10-24).
  - Commercial use and derivatives are allowed.
  - Anyone distributing the model must include a copy of the agreement and a Notice file
    saying "Licensed by NVIDIA Corporation under the NVIDIA Open Model License".
  - The license is revocable. Rights terminate if you circumvent its guardrails or start
    patent or copyright litigation over the model. NVIDIA can update the terms.
  - It requires indemnifying NVIDIA and refers to NVIDIA's Trustworthy AI terms.
  - This is more onerous than CC-BY and should go to legal review before shipping. Because
    VoiceFlow downloads weights from Hugging Face at runtime rather than bundling them, the
    redistribution duties are lighter, but the app should still show the notice.
- **Unified: OpenMDW-1.1.**
  - Permissive: no limits on use or outputs.
  - Keep the license and the notices.
  - Rights terminate if you start patent or copyright litigation.
- mlx-audio code: MIT.

## What was not measured

- **Real human dictation.** The accuracy evidence is synthetic `say` voices plus 73 read
  LibriSpeech clips. Nothing covers spontaneous speech, disfluencies, real microphones,
  rooms or real noise; the only noise is synthetic white noise at 20 dB.
- End-to-end latency through the Unix socket, base64 and Swift scheduling. The numbers
  above are in-process compute only.
- True cold-boot load after a page-cache purge, energy and thermals, long-form audio
  (>30 s), and concurrency with the LLM formatter or the other GPU workload.
- GGUF/transcribe.cpp, NeMo-Speech.cpp, Core ML/ANE ports and NeMo itself.
- Quantized MLX variants (8-bit/4-bit), and a pre-converted bf16 streaming-en checkpoint.
- Local parity of Unified's MLX conversion against NeMo, Unified's non-English languages,
  and its `auto` language mode.
- Whether VoiceFlow's formatter would fix the spoken-form numbers and missing punctuation.

## Reproduce

```sh
cd VoiceFlowApp/build/asr-eval   # venvs: venv-mlx (parakeet-mlx==0.5.1), venv-mlxaudio (mlx-audio==0.6.4),
                                 # venv-mlxaudio-main (mlx-audio@ecd43b9), venv-hf (transformers==5.22.2, torch, librosa)
venv-mlx/bin/python ../../../scripts/asr_eval/corpus.py           # needs pyarrow, soundfile
venv-mlx/bin/python ../../../scripts/asr_eval/bench_batch.py --engine parakeet --model mlx-community/parakeet-tdt-0.6b-v2 --label parakeet-v2
venv-mlxaudio/bin/python ../../../scripts/asr_eval/bench_batch.py --engine mlxaudio --model nvidia/nemotron-speech-streaming-en-0.6b --label streaming-en-offline-1120
venv-mlxaudio/bin/python ../../../scripts/asr_eval/bench_stream.py --delay-ms 160      # also 560, 1120
venv-mlxaudio-main/bin/python ../../../scripts/asr_eval/bench_batch.py --engine mlxaudio --model mlx-community/nemotron-labs-unified-asr-0.6b-bf16 --language en --label unified-bf16-offline
venv-mlxaudio-main/bin/python ../../../scripts/asr_eval/bench_stream.py --model mlx-community/nemotron-labs-unified-asr-0.6b-bf16 --language en --delay-ms 560 --label stream-unified-560ms
venv-hf/bin/python ../../../scripts/asr_eval/bench_hf.py --device mps
venv-mlxaudio/bin/python ../../../scripts/asr_eval/score.py results/*.json
venv-mlx/bin/python ../../../scripts/asr_eval/latency.py results/*.json
python3 ../../../scripts/asr_eval/diff.py results/batch-parakeet-v2.json results/batch-unified-bf16-offline.json short,numeric
```

Run one model process at a time. Each script loads a single model and exits.
