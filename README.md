# VoiceFlow

Voice-to-text with AI formatting — runs entirely locally using Whisper.cpp and Qwen3/SmolLM3 on Apple Silicon and Linux.

## Features

- **Local-first**: No cloud APIs, complete privacy
- **Fast inference**: Optimized for Apple Silicon (Metal) and Linux (CUDA/CPU)
- **Smart formatting**: LLM cleans up transcripts, removes filler words, adds punctuation
- **Context-aware**: Different prompts for email, Slack, code comments, etc.
- **Cross-platform**: macOS CLI now, native app coming soon

## Supported Models

### Speech-to-Text
- Whisper (tiny, base, small, medium) via whisper.cpp

### LLM (Non-Meta, Apache/permissive licensed)
- **Qwen3-1.7B** (default) - Fast, multilingual, Apache 2.0
- **SmolLM3-3B** - Hugging Face's efficient model, Apache 2.0
- **Gemma 2 2B** - Google's compact model, Gemma license

## Installation

### Prerequisites

```bash
# macOS
brew install llvm cmake

# Ubuntu/Debian
sudo apt install build-essential cmake libclang-dev
```

### Build from source

```bash
git clone https://github.com/Era-Laboratories/voiceflow.git
cd voiceflow

# Download models
./scripts/download-models.sh

# Build
cargo build --release

# Install CLI
cargo install --path crates/voiceflow-cli
```

## Usage

### Basic recording

```bash
# Record and format (press Ctrl+C to stop)
voiceflow record

# Copy to clipboard instead of stdout
voiceflow record --clipboard

# Use specific context
voiceflow record --context email

# Raw transcript without LLM formatting
voiceflow record --raw
```

### Configuration

```bash
# Show current config
voiceflow config show

# Change LLM model
voiceflow config set-model smollm3-3b

# Add word to personal dictionary
voiceflow config add-word "VoiceFlow"
```

### Transcribe file

```bash
voiceflow file recording.wav --context slack
```

## Configuration

Config file location: `~/.config/voiceflow/config.toml`

```toml
[models]
whisper_model = "base"
llm_model = "qwen3-1.7b"

[llm]
temperature = 0.3
max_tokens = 512

[audio]
silence_duration_ms = 800
```

## Architecture

```
┌─────────────────────────────────────────┐
│              CLI / App                   │
├─────────────────────────────────────────┤
│           voiceflow-core                 │
├──────────────────┬──────────────────────┤
│   whisper-rs     │     llama-cpp-2      │
│   (whisper.cpp)  │   (Qwen3/SmolLM3)    │
└──────────────────┴──────────────────────┘
        Metal / CUDA / CPU
```

## License

MIT License - see [LICENSE](LICENSE)
