# Release Notes

## v2.0.0 — Parakeet + Bonsai by default, vocabulary-aware spelling, new Settings UI

VoiceFlow v2.0.0 is a major release. The Parakeet → Bonsai stack is now
the **shipped default** — no env vars, no toggles, it just launches.
The polish prompt is fully rewritten around an explicit pipeline + 6
end-to-end examples. Spelling for proper nouns and technical terms is
now driven by a tiered vocabulary system seeded from your active
persona, the focused window, and your recent dictations. The settings
window is a top-to-bottom redesign with a new activity dashboard.

### Default runtime stack

- **Parakeet TDT 0.6B (MLX) is now the default STT.** Previously gated
  behind `VOICEFLOW_USE_PARAKEET=1`. The Rust pipeline's built-in STT is
  unloaded at launch; Parakeet runs out-of-process over a UNIX socket at
  `/tmp/voiceflow_parakeet_daemon.sock`. Set `VOICEFLOW_USE_PARAKEET=0`
  to opt out and use the legacy stack.
- **Bonsai-8B (Q1_0) is now the default LLM.** Served by a bundled
  PrismML `llama-server` fork that the app spawns automatically on
  launch with `-c 16384`. The Rust pipeline routes through
  `LlmBackend::OpenAIServer` to `http://127.0.0.1:8080`. Set
  `VOICEFLOW_LLAMA_SERVER_AUTOSTART=0` if you'd rather run your own
  server.

### Vocabulary-aware spelling

- **Personas now carry vocabulary lists.** Built-in personas come seeded
  (Software Engineer: `kubectl`, `Docker`, `Kubernetes`, `Postgres`,
  `Terraform`, `gRPC`, …; Technical Writer: `API`, `JSON`, `WebSocket`,
  `endpoint`, …). A tag-style chip editor in the Personas settings pane
  lets you add or remove terms with one click.
- **New `[VOCABULARY HINT]` context block.** At dictation time, a new
  `VocabularyContext` module assembles up to ~48 candidate terms from
  four sources, in confidence tiers:
    1. **Hard bias** — proper nouns and identifiers visible in the
       focused field (read from the AX text-before-cursor buffer).
    2. **Hard bias** — host + path tokens for the active browser tab.
    3. **Soft bias** — your active persona's vocabulary list.
    4. **Soft bias** — terms recurring in your recent dictation history.
  The block sits in the LLM prompt's cached prefix, so it costs ~150
  tokens and effectively zero added latency after the first request.
- **Migration is transparent.** Older stored personas without a
  `vocabulary` field decode cleanly; built-in personas get the seed
  terms backfilled on first launch.

### Polish prompt overhaul

`prompts/default.txt` is fully rewritten around an explicit eight-step
pipeline and a single, deduplicated set of safety rules.

- **Spelling Resolution is now the first step,** with phonetic-match
  rules across `[VOCABULARY HINT]`, `[INPUT CONTEXT]`,
  `[PERSONAL_DICTIONARY]`, and on-screen text. Examples added for the
  cases we'd actually been getting wrong: "argo c d" → "ArgoCD", "oh
  auth" → "OAuth", "post gress" → "Postgres", "kuda" → "CUDA", "era
  core" → "Era Code".
- **Prosody hints are now imperative tags.** The previous
  natural-language hint ("The speaker's pitch rose at the end,
  suggesting a question") is replaced by `[PROSODY: rising pitch]` /
  `[PROSODY: falling pitch]` / `[PROSODY: emphatic]` / `[PROSODY: long
  pauses N]`, each explicitly mapped to a punctuation action in the
  prompt. Small models stopped ignoring the hint.
- **Six end-to-end examples** added at the bottom of the prompt
  covering prose-with-fillers, voice command + self-correction, URL /
  IP / version notation, list, vocabulary-driven spelling, and a
  question detected by prosody.
- **Anti-conversational guarantee strengthened** with seven diverse
  examples (questions to AI, soliciting feedback, hedged opinions, a
  "tell me a joke" provocation) so the model stops answering questions
  it should be transcribing.
- **Abstain rule** — empty, single-syllable, or garbled input is now
  passed through verbatim instead of hallucinated into a sentence.
- **Length guard** — output word count must approximate input word
  count minus fillers. Cuts mid-sentence drift on smaller models.
- **Voice command disambiguation** — phrases only execute as commands
  when at sentence boundaries or grammatically nonsensical as nouns.
  Explicit negative examples ("the comma is overused" → keeps "comma"
  as a word).
- **Numbers / URL / IP rules consolidated.** Self-contradicting bullets
  ("Small counts spell out" vs "ALWAYS use digits") removed. One rule
  per context.
- **`email.txt`, `code.txt`, `audio_direct.txt` brought to parity.**
  Each now has the same anti-conversational + preserve-content + Spelling
  Resolution guarantees as the default prompt.

### Personal dictionary as a labeled block

`personal_dictionary` is now injected as a `[PERSONAL_DICTIONARY]`
metadata block rather than a free-floating "Personal vocabulary: …"
line. Combined with `[VOCABULARY HINT]`, the Spelling Resolution
section now has a uniform way to address vocabulary across all sources.

### Settings UI

The whole settings window is rebuilt on a new "Liquid Glass" design
system (`VF.*` tokens):

- **New Home with Overview + Activity tabs.** A speed gauge shows your
  current WPM and tier (`Steady` 80–130, `Fast` 130–200, `Top` 200+).
  Per-app usage bars surface where you dictate most. A
  GitHub-style streak heatmap tracks daily activity, with a "longest
  streak" stat. An "estimated time saved vs typing at 40 wpm" headline
  number sits at the top.
- **Persona vocabulary editor.** Inline chip-style tag editor on the
  Personas pane — type a term, press Enter, it's added; click the × to
  remove.
- **Sidebar regrouped** into named sections via the new
  `SidebarGroup` / `SidebarLayout` model.
- **All settings panes restyled** with consistent `VFCard`,
  `VFPageHeader`, `VFSectionHeader`, and pill-button affordances.
  Style, AI Features, General, Snippets, App Profiles, Browser Site
  Rules, and Personas all share the same visual language now.

### Devex / observability

- **LLM input/output pairs are logged on every dictation path.** Three
  new `NSLog` lines per request:
    ```
    [VoiceFlow] Parakeet → LLM: formatting N-char transcript
    [VoiceFlow] LLM input:  <raw Parakeet output>
    [VoiceFlow] LLM output: <Bonsai-formatted text>
    ```
  Pipe through `/usr/bin/log stream --predicate 'eventMessage CONTAINS
  "[VoiceFlow]"'` for a live trace.
- **Runtime prompt override.** Prompts under
  `~/Library/Application Support/com.era-laboratories.voiceflow/prompts/<name>.txt`
  are loaded if present, otherwise the compiled-in copy is used. Useful
  for iterating on prompts without a rebuild.
- **Leaked-block stripper expanded.** Any of `[VOCABULARY HINT]`,
  `[APPLICATION CONTEXT: …]`, `[PERSONAL_DICTIONARY]`, or `[PROSODY: …]`
  appearing in model output is now truncated by the post-processor.

### Migration notes

- No config changes required. If you previously launched VoiceFlow with
  `VOICEFLOW_USE_PARAKEET=1` etc. in your shell rc, you can remove
  those — the defaults now match.
- Existing persona JSONs are forward-compatible. The new `vocabulary`
  field decodes to `[]` on old records, and built-in personas get
  seed terms backfilled on first launch.

## v0.2.1 — Stability & lifecycle

Follow-up to v0.2.0 focused on the known issues called out in that
release. Headline: VoiceFlow now self-hosts the local LLM server,
which incidentally also fixes the output-truncation bug and removes
the manual `llama-server` setup step.

### Fixes

- **LLM output no longer truncated mid-sentence.** A new
  `LlamaServerManager` (Swift) spawns the PrismML `llama-server`
  fork on app launch with **`-c 16384`** (was 4096). Long prompts
  with persona context + transcript no longer hit
  `truncated = 1` in the server log. The manager also adopts an
  already-running server on `:8080` instead of double-spawning,
  searches multiple paths for the binary (`Bundle.main` Resources →
  `VOICEFLOW_LLAMA_SERVER` env → `~/PrismML-llama.cpp/build/bin/...`
  → `~/dev/PrismML-llama.cpp/...`), and surfaces a clear error
  with build instructions when the binary isn't found.
- **Persona auto-classifier is much more conservative.** The
  **Classify Unmapped (LLM)** path now uses self-consistency: 3
  trials per app at `temperature=0.5` (was 1 trial at 0.1). A
  classification is only applied when a single persona wins **≥2
  votes** AND every vote for it clears the **0.90** confidence
  threshold. Ties or split votes are dropped. Per-app vote tallies
  are logged so you can audit decisions in
  `/tmp/voiceflow_app.log`.
- **Parakeet first-load shows real progress.** Setup now polls the
  Hugging Face cache directory every 500 ms while parakeet-mlx
  fetches its weights and surfaces byte-level progress, e.g.
  `Downloading Parakeet TDT 0.6B — 612 / 1200 MB`. Replaces the
  previous indeterminate spinner.
- **Sidebar emptiness mitigation.** Couldn't reliably reproduce,
  but cached the app logo `NSImage` as a static (was reloaded from
  disk on every body re-eval — a known cause of SwiftUI flicker on
  macOS) and pinned `minWidth: 200 / minHeight: 400` on the sidebar
  VStack so the inner `Spacer()` can't collapse it during a layout
  pass.

### New

- **`LlamaServerManager.swift`** — lifecycle manager for the local
  LLM server. Spawns on `applicationDidFinishLaunching`, terminates
  on `applicationWillTerminate`. Health-checked via a `/v1/models`
  probe before reporting `.ready`.
- **`VOICEFLOW_LLAMA_SERVER_AUTOSTART=0`** env var lets developers
  who run their own server bypass autostart.

### Known Issues (carried forward)

- **PrismML `llama-server` install is still manual.** The new
  manager finds the binary if you've built it locally, but Setup
  doesn't yet build/download it. Documented in
  `LlamaServerManager.start()`'s error path.

## v0.2.0 — Parakeet + Bonsai pivot

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
