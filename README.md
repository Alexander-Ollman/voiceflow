<p align="center">
  <img src="img/app.png" alt="VoiceFlow Logo" width="200" height="200">
</p>

<h1 align="center">VoiceFlow</h1>

<p align="center">
  <strong>Lightning-fast voice-to-text for macOS</strong><br>
  Hold Option+Space, speak, release to paste. It's that simple.
</p>

<p align="center">
  <a href="#features">Features</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#voice-commands">Voice Commands</a> &bull;
  <a href="#building-from-source">Building</a>
</p>

---

## Features

- **Push-to-talk** &mdash; Hold `⌥ Space` to record, release to transcribe and paste
- **100% local, Apple Silicon native** &mdash; Two on-device models, no cloud, no telemetry
- **Parakeet TDT 0.6B v2** &mdash; NVIDIA's MLX-accelerated STT, ~1.7% WER on LibriSpeech test-clean (≈10× more accurate than the previous Moonshine default on conversational dictation)
- **Bonsai-8B Q1_0** &mdash; PrismML's 1.125-bit quantized 8B language model for context-aware formatting, served by a local `llama-server` VoiceFlow auto-launches
- **Personas** &mdash; Reusable LLM context fragments (Software Engineer, Professional Email, Casual Chat, Technical Writing) that get injected into the formatting prompt based on what app you're in
- **Per-app + per-website context** &mdash; VoiceFlow auto-maps your installed apps to personas, and detects the active browser tab's URL to apply site-specific rules (`github.com` → Software Engineer, `mail.google.com` → Professional Email, etc.)
- **Intent-aware formatting** &mdash; Optional formatting level that lets the LLM fix obvious STT mishears using sentence-level coherence (e.g. "use cube cuttle to deploy" → "use kubectl to deploy" when context indicates engineering work)
- **LLM-driven app classification** &mdash; One-click "Classify Unmapped (LLM)" hands every persona-less app to Bonsai for a 3-vote consensus pick (≥0.90 confidence required)
- **Live dashboard** &mdash; Models tab shows model state on disk + service state (PID, uptime, RSS, request count) for both the Parakeet daemon and llama-server, polled every 3 s
- **Smart formatting** &mdash; Automatic punctuation, capitalization, em-dashes, bullet lists
- **Number normalization** &mdash; "fifty thousand dollars" → "$50,000", "three thirty pm" → "3:30 PM", phone numbers, percentages, dates, and keyword numbers (port 8000) all converted automatically
- **Voice commands** &mdash; Say "new paragraph", "bullet point", "question mark" and more
- **Context-aware** &mdash; Reads cursor context via Accessibility API for seamless spacing
- **Visual context** &mdash; Capture and analyze your screen via the multimodal LLM to extract names, terms, and writing context
- **Correction learning** &mdash; Learns from your edits to fix recurring spelling mistakes automatically
- **Voice snippets** &mdash; Custom trigger phrases that expand into any text
- **AI voice commands** &mdash; Say "reply to this", "rewrite this", "proofread this", "continue writing", "summarize this" for context-aware AI actions on the current text field
- **Repeat to replace** &mdash; Re-say a sentence (or just fix a word) and VoiceFlow replaces your last dictation instead of appending it. A deterministic pre-check &mdash; correction hotwords ("scratch that", "I meant…") plus word-overlap with your last output &mdash; gates the call, then Bonsai confirms redo-vs-new. Hold `⌥⇧ Space` to force a replace, or to speak an edit ("make it more formal")
- **Menu bar app** &mdash; Minimal footprint, no dock icon, launch at login

## How It Works

VoiceFlow runs two locally-hosted services as children of the macOS app:

```
                ┌──────────────────────────────────────────────────────┐
                │           macOS SwiftUI App (menu bar)               │
                │   Hotkeys, audio capture, persona resolution, UI     │
                └────────────────────┬─────────────────────────────────┘
                                     │
                ┌────────────────────┴─────────────────────┐
                │                                          │
        Hold ⌥ Space, speak, release ⌥ Space               │
                │                                          │
                ▼                                          ▼
   ┌────────────────────────┐                  ┌────────────────────────────┐
   │ Parakeet daemon (Py)   │                  │  llama-server (PrismML)    │
   │ parakeet-mlx           │                  │  Bonsai-8B Q1_0, ctx 16k   │
   │ Unix socket, length-   │                  │  127.0.0.1:8080            │
   │ prefixed JSON          │                  │  /v1/completions (SSE)     │
   └───────────┬────────────┘                  └────────────┬───────────────┘
               │                                            │
               │  raw transcript                            │  formatted text
               └────────────────┬───────────────────────────┘
                                ▼
                    paste at cursor (with persona-aware
                    spacing, capitalization, snippet
                    expansion, optional retroactive correction)
```

VoiceFlow spawns and supervises both services on launch. They shut down with the app.

Before appending a paste, VoiceFlow checks whether you're **redoing** the last one: a deterministic gate (correction hotwords + word-overlap with the previous output) decides whether to consult Bonsai, which confirms redo-vs-new and produces the replacement. On a redo it selects the prior insertion via the Accessibility API and replaces it in place. Holding `⌥⇧ Space` forces this — re-say to replace, or speak an instruction to edit.

The LLM prompt is composed at dictation time:

```
[base system prompt]
  + [persona prompt]            ← derived from frontmost app or browser-tab URL
  + [input field context]       ← text already at the cursor (if any)
  + [correction history]        ← past user edits
  + [transcript]
```

## Installation

### Requirements

- **macOS 13.0** (Ventura) or later
- **Apple Silicon** (M1 or later) — Parakeet runs via MLX which is Apple-Silicon-only
- **~2.5 GB** disk space for the two models

### Download (recommended)

1. Download the latest `VoiceFlow.zip` from [Releases](https://github.com/Alexander-Ollman/voiceflow/releases)
2. Extract and launch — the app will offer to move itself to Applications
3. Grant **Microphone**, **Accessibility**, and **Automation** permissions when prompted
4. Click **Setup** in the onboarding wizard. Two models download (~2.3 GB total, one-time)

### Build from source

See [Building from Source](#building-from-source) below — note you'll also need to build PrismML's `llama.cpp` fork (Bonsai uses Q1_0 quantization which upstream `llama.cpp` doesn't yet support).

## Usage

1. **Launch VoiceFlow** &mdash; Look for the icon in your menu bar
2. **Hold `⌥ Space`** &mdash; Start speaking
3. **Release `⌥ Space`** &mdash; Text is transcribed by Parakeet, formatted by Bonsai with the right persona for your current app, and pasted at your cursor

### Editing & repeating

VoiceFlow can replace your **last** dictation in place instead of always appending:

- **Just say it again** &mdash; Re-dictate the sentence, or only the fix (e.g. say "store" after "I went to the stoor"). When your utterance looks like a redo &mdash; a correction hotword (`scratch that`, `I meant…`) or high word-overlap with the last output &mdash; Bonsai decides whether to replace or append. Clearly-new speech is always appended, and the replace only happens while your last paste is still present and recent in the same app.
- **Hold `⌥⇧ Space`** (the edit hotkey) &mdash; Hold it and either re-say the text to replace it, or speak an instruction (`make it more formal`, `change 3pm to 4pm`). The spoken edit is applied to your last paste in place. Works while that paste is recent and you haven't switched apps.

### Status Indicator

The menu bar icon changes color to show status:

| Color | State |
|-------|-------|
| Default | Ready |
| Red | Recording |
| Yellow | Processing |

The floating overlay pill shows real-time status during dictation and AI features:

| Pill State | Appearance |
|------------|------------|
| Recording | Animated gradient + waveform bars |
| Processing | Dark pill + pulsing dots |
| AI Processing | Dark pill + sparkle icon + status text ("Summarizing...") |

### Settings

Access from the menu bar icon → **Settings** or click the app icon. The sidebar has:

| Tab | What it does |
|-----|--------------|
| **Home** | Stats and recent dictation history |
| **Models** | Live dashboard: model files, llama-server + Parakeet daemon status (PID, uptime, RSS, request count). One-click Setup / Re-verify |
| **Snippets** | Custom trigger phrases that expand into any text |
| **Style** | Formatting level (Minimal / Moderate / Intent-Aware / Aggressive), spacing, punctuation toggles |
| **Personas** | Reusable LLM context fragments. Edit the four built-ins or add your own |
| **App Profiles** | Per-app persona assignments. Auto-Detect Apps scans `/Applications` and maps based on a curated bundle-id table. Classify Unmapped (LLM) uses Bonsai itself with a 3-vote consensus to fill in the rest |
| **Browser Sites** | Hostname rules for when a browser is foreground (`github.com` → Software Engineer, `mail.google.com` → Professional Email, etc.). Wildcards: `*.atlassian.net` |
| **AI Features** | Voice command toggles (reply, rewrite, proofread, continue, summarize) |
| **Settings** | Permissions, hotkey, launch-at-login |

## Models

VoiceFlow ships with two models. Both are downloaded by **Setup** in the onboarding wizard or the Models tab.

### Speech-to-Text

**Parakeet TDT 0.6B v2** &mdash; NVIDIA's FastConformer-TDT English ASR, ported to MLX by mlx-community.

| Spec | Value |
|------|-------|
| Parameters | 600 M |
| File size | ~1.2 GB (bf16) |
| LibriSpeech test-clean WER | 1.69% |
| Runtime | [parakeet-mlx](https://github.com/senstella/parakeet-mlx) (MLX, Apple Silicon) via a Python daemon |
| HF repo | [`mlx-community/parakeet-tdt-0.6b-v2`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) |
| License | CC-BY-4.0 |

### LLM (formatting)

**Bonsai-8B Q1_0** &mdash; PrismML's 1.125-bit quantized 8B model, derived from the Qwen3-8B base.

| Spec | Value |
|------|-------|
| Parameters | 8.19 B (~6.95 B non-embedding) |
| Quantization | Q1_0 g128 (1.125 bits per weight) |
| File size | ~1.1 GB (vs. ~16 GB at fp16 — 14× smaller) |
| Runtime | [PrismML's `llama.cpp` fork](https://github.com/PrismML-Eng/llama.cpp) (Q1_0 not yet in upstream `llama.cpp`) |
| Context | 16 384 tokens (configurable in `LlamaServerManager`) |
| HF repo | [`prism-ml/Bonsai-8B-gguf`](https://huggingface.co/prism-ml/Bonsai-8B-gguf) |

VoiceFlow auto-spawns `llama-server` on app launch with `-c 16384 -ngl 99` pointed at the GGUF file, and shuts it down on app exit. To bypass autostart (e.g. you want to run your own), set `VOICEFLOW_LLAMA_SERVER_AUTOSTART=0`.

> **Note:** the underlying Rust crate (`voiceflow-core`) still has backend code paths for Whisper, Moonshine, Qwen3-ASR, Qwen3.5, Gemma 4 etc. They're disabled in the shipping UI but remain wired for development and benchmarking via the CLI.

## Personas & Per-App Context

VoiceFlow injects domain context into the LLM prompt based on what you're typing into. There are three layers of resolution:

1. **Browser site rule** (highest priority) — if the foreground app is Safari/Chrome/Brave/Vivaldi/Edge/Arc, VoiceFlow reads the active tab's URL via AppleScript and matches against ~45 seeded host rules. First match wins, exact host beats wildcard.
2. **App profile persona** — every other app maps to a persona (`Software Engineer` for terminals/IDEs, `Casual Chat` for Slack/Messages, etc.).
3. **Category default** — fallback when no persona is set.

Built-in personas:

| Persona | Use case |
|---------|----------|
| Software Engineer | Terminal, IDEs, dev tools, technical chat. Knows `kubectl`, `Postgres`, `OAuth`, `JWT`, etc. |
| Professional Email | Mail clients, formal correspondence. Complete sentences, formal sign-offs. |
| Casual Chat | Slack, Messages, Discord, WhatsApp, FaceTime. Contractions, informal tone, lowercase acronyms. |
| Technical Writing | Notion, docs, blog editors. Long-form prose, precise terminology. |

You can edit the built-ins, add your own, and reassign apps freely under **Personas** and **App Profiles**.

### Auto-detection on first launch

Two zero-config systems pre-populate sensible defaults:

- **Auto-Detect Apps** scans `/Applications` (and `Utilities`, `System Applications`) once and maps known bundle-ids to personas using a curated table. Re-runnable from the App Profiles tab.
- **Classify Unmapped (LLM)** lets Bonsai decide for the apps the curated table didn't cover. Each app gets 3 independent classifications at temperature 0.5; only applied when a single persona wins ≥2 votes AND every vote ≥ 0.90 confidence. Per-app vote tallies logged for auditing.

## Voice Commands

### Punctuation

| Say | Output |
|-----|--------|
| "period" / "full stop" | `.` |
| "comma" | `,` |
| "question mark" | `?` |
| "exclamation mark" / "bang" | `!` |
| "colon" | `:` |
| "semicolon" | `;` |
| "ellipsis" / "dot dot dot" | `...` |

### Formatting

| Say | Output |
|-----|--------|
| "new line" / "line break" | Line break |
| "new paragraph" | Paragraph break |
| "open quote" / "close quote" | `"` |
| "apostrophe" | `'` |
| "dash" / "em dash" | `—` |
| "hyphen" | `-` |

### Brackets & Grouping

| Say | Output |
|-----|--------|
| "open paren" / "close paren" | `(` `)` |
| "open bracket" / "close bracket" | `[` `]` |
| "open brace" / "close brace" | `{` `}` |

### Special Characters

| Say | Output |
|-----|--------|
| "ampersand" | `&` |
| "at sign" | `@` |
| "hashtag" / "hash" | `#` |
| "dollar sign" | `$` |
| "percent" | `%` |
| "asterisk" / "star" | `*` |
| "underscore" | `_` |
| "slash" / "forward slash" | `/` |
| "backslash" | `\` |

### Programming

| Say | Output |
|-----|--------|
| "equals" | `=` |
| "plus" | `+` |
| "minus" | `-` |
| "greater than" | `>` |
| "less than" | `<` |
| "pipe" | `|` |
| "tilde" | `~` |
| "caret" | `^` |

### Automatic Features

- **Punctuation** &mdash; Added automatically based on speech patterns
- **Capitalization** &mdash; Sentences capitalized after punctuation
- **Number normalization** &mdash; Currency ($50,000), percentages (25%), times (3:30 PM), phone numbers (555-123-4567), dates (January 15), and keyword numbers (port 8000)
- **Abbreviations** &mdash; "doctor Smith" → "Dr. Smith", "follow up appointment" → "follow-up appointment"
- **Lists** &mdash; Enumerated items converted to bullet points
- **Em-dashes** &mdash; Mid-sentence pauses become `—`
- **Filler removal** &mdash; "um", "uh", "ah", "hmm", "er" removed
- **Spelled-out words** &mdash; "S M O L L M" → "SMOLLM"
- **Technical terms** &mdash; Configurable replacement dictionary

## Building from Source

### Prerequisites

- **Rust 1.70+** &mdash; Install via [rustup](https://rustup.rs/)
- **Xcode Command Line Tools** &mdash; `xcode-select --install`
- **Swift 5.9+** &mdash; Included with Xcode
- **Python 3.10+** with `parakeet-mlx` &mdash; Used by the Parakeet daemon
- **PrismML's `llama.cpp` fork** &mdash; Required for Bonsai's Q1_0 quantization

### Build PrismML's llama-server

Bonsai's Q1_0 quantization isn't in upstream `llama.cpp` yet, so you need to build PrismML's fork:

```bash
git clone https://github.com/PrismML-Eng/llama.cpp ~/PrismML-llama.cpp
cd ~/PrismML-llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON \
              -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build --target llama-server -j
```

The resulting binary at `~/PrismML-llama.cpp/build/bin/llama-server` is auto-discovered by `LlamaServerManager`. You can override with `VOICEFLOW_LLAMA_SERVER=/path/to/llama-server`.

### Build VoiceFlow

```bash
# Clone the repository
git clone https://github.com/Alexander-Ollman/voiceflow.git
cd voiceflow

# Install the Parakeet daemon's Python dependencies
pip install parakeet-mlx     # one-time, into your VoiceFlow Python env

# Build the Rust library + macOS app
cd VoiceFlowApp
./build.sh

# Run
open build/VoiceFlow.app
```

The first run will offer to download both models (~2.3 GB) via the Setup wizard.

## Configuration

VoiceFlow stores its configuration at:

```
~/Library/Application Support/com.era-laboratories.voiceflow/config.toml
```

Day-to-day, you shouldn't need to edit this — the UI covers the common settings. The TOML lets you override anything the UI exposes.

### Environment variable overrides

| Variable | Purpose |
|----------|---------|
| `VOICEFLOW_USE_PARAKEET=1` | Use Parakeet for STT (default in shipping builds) |
| `VOICEFLOW_LLM_BACKEND=openai_server` | Use the local llama-server for LLM |
| `VOICEFLOW_LLM_SERVER_ENDPOINT=http://127.0.0.1:8080` | LLM server URL |
| `VOICEFLOW_LLM_SERVER_MODEL=Bonsai-8B-Q1_0.gguf` | Model name to send in requests |
| `VOICEFLOW_LLAMA_SERVER=/path/to/llama-server` | Override llama-server binary path |
| `VOICEFLOW_LLAMA_SERVER_AUTOSTART=0` | Disable VoiceFlow's auto-spawn (run your own) |
| `VOICEFLOW_PYTHON=/path/to/python3` | Override Python interpreter for daemons |

### File paths

| Path | Contents |
|------|----------|
| `~/Library/Application Support/com.era-laboratories.voiceflow/config.toml` | Configuration |
| `~/Library/Application Support/com.era-laboratories.voiceflow/models/Bonsai-8B-Q1_0.gguf` | Bonsai weights |
| `~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/` | Parakeet weights (HF cache) |
| `/tmp/voiceflow_parakeet_daemon.{log,pid,sock}` | Parakeet daemon lifecycle |
| `/tmp/llama_server_bonsai.log` | llama-server log |

## Project Structure

```
voiceflow/
├── crates/
│   ├── voiceflow-core/                    # Core Rust library
│   │   ├── src/
│   │   │   ├── config.rs                  # Configuration management
│   │   │   ├── pipeline.rs                # Main processing pipeline
│   │   │   ├── llm/
│   │   │   │   ├── llamacpp_backend.rs    # In-process llama.cpp (legacy path)
│   │   │   │   ├── openai_server_backend.rs  # SSE-streaming HTTP client (default)
│   │   │   │   ├── mistralrs_backend.rs   # Optional, Gemma 4 audio (legacy)
│   │   │   │   └── prompts.rs             # Prompt assembly
│   │   │   ├── transcribe/                # In-process STT engines (legacy paths)
│   │   │   ├── prosody/                   # Voice commands, filler removal, pauses
│   │   │   └── audio/                     # Audio capture and resampling
│   │   └── Cargo.toml
│   ├── voiceflow-cli/                     # Command-line interface
│   └── voiceflow-ffi/                     # C FFI for Swift bindings
├── VoiceFlowApp/                          # macOS SwiftUI application
│   ├── Sources/VoiceFlowApp/
│   │   ├── VoiceFlowApp.swift             # Main app, menu bar, settings UI
│   │   ├── ParakeetASREngine.swift        # Daemon spawn + IPC
│   │   ├── LlamaServerManager.swift       # llama-server lifecycle
│   │   ├── BrowserContext.swift           # AppleScript URL extraction + site rules
│   │   ├── PersonaClassifier.swift        # LLM-driven app classification
│   │   ├── ServiceMonitor.swift           # Live PID/uptime/RSS polling
│   │   ├── SetupHelper.swift              # Setup wizard install logic
│   │   └── ...
│   └── build.sh                           # App bundle build script
├── prompts/                               # LLM system prompts
│   ├── default.txt                        # Base formatting prompt
│   └── replacements.toml                  # Word replacements
├── scripts/
│   ├── parakeet_asr_daemon.py             # Parakeet-MLX daemon
│   └── qwen3_asr_daemon.py                # Legacy Qwen3-ASR daemon
└── Cargo.toml                             # Workspace configuration
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, architecture details, and guidelines.

## License

MIT License &mdash; [Alex Ollman](https://github.com/Alexander-Ollman) 2025

See [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with care for fast typists who'd rather talk
</p>
