# Release Notes

## Unreleased — Parakeet + Bonsai pivot

This release retools VoiceFlow around two on-device models:
**Parakeet TDT 0.6B v2** for speech-to-text (NVIDIA, MLX-accelerated) and
**Bonsai-8B Q1_0** for LLM formatting (PrismML, 1.125-bit GGUF).
Together they take ~2.3 GB on disk and run cleanly on a recent Apple
Silicon Mac. Older models (Whisper, Moonshine streaming, Qwen3.5,
Gemma 4) still compile but are no longer surfaced in the UI.

### Highlights

- **New STT engine: Parakeet TDT 0.6B v2** via `parakeet-mlx`.
  ~1.7% WER on LibriSpeech test-clean, ~10× more accurate than
  Moonshine medium on conversational dictation. Runs as a Python
  daemon over a Unix socket using the same length-prefixed JSON
  protocol as the existing Qwen3-ASR daemon.
- **New LLM serving path: OpenAI-compatible HTTP backend.** A new
  `OpenAIServerBackend` in `voiceflow-core` talks SSE to any local
  `/v1/completions` endpoint (`llama-server`, `mlx_lm.server`, vLLM,
  Ollama with the OAI shim, etc.). Lets us hot-swap models without
  rebuilding the Rust pipeline.
- **Default LLM: Bonsai-8B Q1_0** served by PrismML's `llama.cpp`
  fork (Q1_0 is not yet in upstream). Roughly 14× smaller than an
  fp16 8B model with usable accuracy for short dictation cleanup.
- **Personas + per-app/per-site context.** A new persona system
  injects domain context into the LLM prompt:
  - **Personas**: reusable system-prompt fragments. Ships with four
    seeds — Software Engineer, Professional Email, Casual Chat,
    Technical Writing — all editable from the new **Personas**
    settings tab.
  - **App Profiles**: tie any installed app to a persona. Settings
    has an **Auto-Detect Apps** button that scans
    `/Applications` (and Utilities + System Applications) and
    pre-assigns personas using a curated bundle-id table plus
    display-name patterns.
  - **Browser Sites**: when the foreground app is a browser
    (Safari/Chrome/Brave/Vivaldi/Edge/Arc), VoiceFlow reads the
    active tab's URL via AppleScript and matches against per-host
    rules. Ships with 45 seeded rules across the four built-in
    personas.
  - **LLM-driven classification**: a new **Classify Unmapped (LLM)**
    button hands every persona-less app to Bonsai for a
    JSON-formatted `(persona, confidence, reason)` decision and
    applies any classification ≥ 90% confidence.
- **Intent-aware formatting level.** A new `intent` formatting level
  in the menu bar (alongside Minimal / Moderate / Aggressive) tells
  the LLM to fix obvious STT mishears using sentence-level coherence
  and phonetic similarity. Conservative by design — keeps the
  original word when uncertain.
- **Models settings dashboard.** The Models tab is now a live
  dashboard showing both model files (size, install state, on-disk
  path) and both running services (Parakeet daemon and llama-server)
  with PID, uptime, RSS, and rolling request count. Polls every 3 s
  while visible.
- **Onboarding wizard simplified.** The model-selection step is
  replaced with a single **Setup** button that downloads Bonsai-8B
  and triggers Parakeet's first-load fetch in one go (~2.3 GB total,
  one-time).
- **Sidebar gains three sections**: **Personas**, **App Profiles**,
  **Browser Sites**.
- **Streaming-release LLM regression fixed.** A working-tree change
  had the streaming-release path calling
  `pipeline.process_text_deterministic` (no LLM) regardless of
  setting. Restored to `formatTextStreaming` so the LLM actually
  runs on release. While there, dropped the per-token formatting
  bar — modern hardware finishes generation faster than the
  animation could be read; we now show the simpler `processing`
  indicator until the result is ready.

### Architecture Notes

- Backend enums (`SttEngine`, `LlmModel`, `LlmBackend`,
  `MoonshineModel`, `WhisperModel`, `ConsolidatedModel`) are
  intentionally **left intact**. Only the UI/onboarding has been
  narrowed to Parakeet + Bonsai — older models still compile and
  can be selected by editing config or env vars.
- New env vars: `VOICEFLOW_USE_PARAKEET=1`,
  `VOICEFLOW_LLM_BACKEND=openai_server`,
  `VOICEFLOW_LLM_SERVER_ENDPOINT`, `VOICEFLOW_LLM_SERVER_MODEL`.
- New Rust deps: `ureq` (sync HTTP + SSE), `base64` (image
  encoding for OAI vision content blocks).
- AppleScript automation requires a one-time macOS approval per
  browser. `NSAppleEventsUsageDescription` is set in `Info.plist`
  to surface a clear permission prompt.

### Known Issues

- **LLM output occasionally truncated.** `llama-server` is started
  with `-c 4096`. Once the system prompt + persona + transcript
  cross ~4070 tokens (which happens on long dictations or when a
  large app/site persona is active), the output gets cut off mid-
  sentence (`truncated = 1` in the server log).
  **Fix**: bump `-c` to 8192 or 16384 when launching `llama-server`.
  Untouched in this release.
- **Sidebar occasionally renders empty after navigation.** Cause
  not fully isolated; clicking another sidebar item or reopening
  the window via the menu bar restores it. Suspect a SwiftUI
  `NavigationSplitView` column-visibility quirk.
- **Persona auto-classification can be over-confident on short
  prompts.** Despite a 0.90 confidence threshold and a prompt that
  explicitly tells the model to bail on browsers and generic
  utilities, Bonsai-8B occasionally rationalizes a confident
  Software Engineer or Professional Email pick for ambiguous apps.
  Mitigation: review picks in App Profiles and override manually.
- **PrismML `llama-server` must be installed manually.** The Setup
  button does not yet build or download the PrismML fork. Until
  that's automated, install from
  `github.com/PrismML-Eng/llama.cpp` and start it pointed at the
  Bonsai GGUF before launching VoiceFlow.
- **Parakeet first-run model download (~1.2 GB) blocks the UI**
  with a spinner. No granular progress for the parakeet-mlx side
  yet (Bonsai download has byte-level progress).

### Files Added

- `VoiceFlowApp/Sources/VoiceFlowApp/ParakeetASREngine.swift`
- `VoiceFlowApp/Sources/VoiceFlowApp/BrowserContext.swift`
- `VoiceFlowApp/Sources/VoiceFlowApp/PersonaClassifier.swift`
- `VoiceFlowApp/Sources/VoiceFlowApp/ServiceMonitor.swift`
- `VoiceFlowApp/Sources/VoiceFlowApp/SetupHelper.swift`
- `crates/voiceflow-core/src/llm/openai_server_backend.rs`
- `scripts/parakeet_asr_daemon.py`
- `RELEASE_NOTES.md` (this file)
