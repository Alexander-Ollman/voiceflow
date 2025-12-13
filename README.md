<p align="center">
  <img src="logo.png" alt="VoiceFlow Logo" width="200" height="200">
</p>

<h1 align="center">VoiceFlow</h1>

<p align="center">
  <strong>Lightning-fast voice-to-text for macOS</strong><br>
  Hold Option+Space, speak, release to paste. It's that simple.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#voice-commands">Voice Commands</a> •
  <a href="#building-from-source">Building</a>
</p>

---

## Features

- **Push-to-talk**: Hold `⌥ Space` to record, release to transcribe and paste
- **Blazing fast**: ~1.5 second total latency (Whisper + LLM)
- **100% local**: All processing happens on-device using Metal GPU acceleration
- **Smart formatting**: Automatic punctuation, em-dashes, bullet lists
- **Voice commands**: Say "new paragraph", "bullet point", "question mark" and more
- **Menu bar app**: Minimal footprint, no dock icon

## Installation

### Download

1. Download the latest `VoiceFlow-Installer.dmg` from [Releases](../../releases)
2. Open the DMG and drag VoiceFlow to Applications
3. Launch VoiceFlow from Applications
4. Grant Microphone and Accessibility permissions when prompted

### Requirements

- macOS 13.0 (Ventura) or later
- Apple Silicon (M1/M2/M3) recommended for best performance

## Usage

1. **Start VoiceFlow** - Look for the icon in your menu bar
2. **Hold `⌥ Space`** - Start speaking
3. **Release `⌥ Space`** - Text is transcribed, formatted, and pasted automatically

The icon changes color to indicate status:
- 🔴 **Red**: Recording
- 🟡 **Yellow**: Processing
- ⚪ **Normal**: Ready

## Voice Commands

VoiceFlow understands natural voice commands for formatting:

| Say this | Get this |
|----------|----------|
| "new line" | Line break |
| "new paragraph" | Paragraph break |
| "bullet point" | • Bullet item |
| "question mark" | ? |
| "exclamation point" | ! |
| "comma" | , |
| "period" | . |
| "colon" | : |
| "open quote ... close quote" | "quoted text" |

### Automatic Features

- **Punctuation**: Added automatically based on speech patterns
- **Lists**: Enumerated items converted to bullet points
- **Em-dashes**: Mid-sentence pauses become — dashes
- **Filler removal**: "um", "uh", "like", "you know" removed automatically

## Building from Source

### Prerequisites

- Rust 1.70+
- Xcode Command Line Tools
- Swift 5.9+

### Build

```bash
# Clone the repository
git clone https://github.com/yourusername/voiceflow.git
cd voiceflow

# Build the app
cd VoiceFlowApp
./build.sh

# Run
open build/VoiceFlow.app

# Create DMG installer
./create-dmg.sh
```

## Architecture

```
voiceflow/
├── crates/
│   ├── voiceflow-core/    # Whisper + LLM pipeline
│   ├── voiceflow-cli/     # Command-line interface
│   └── voiceflow-ffi/     # C FFI for Swift bindings
├── VoiceFlowApp/          # macOS SwiftUI app
├── models/                # ML models (Whisper, Qwen)
└── prompts/               # LLM system prompts
```

## Tech Stack

- **Transcription**: [Whisper](https://github.com/openai/whisper) via whisper-rs
- **LLM**: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen2.5-0.5B) via mistral.rs
- **GPU**: Metal acceleration on Apple Silicon
- **App**: SwiftUI + Carbon (hotkeys)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ for fast typists who'd rather talk
</p>
