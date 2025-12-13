# Contributing to VoiceFlow

Thank you for your interest in contributing to VoiceFlow! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- **Rust 1.70+** - Install via [rustup](https://rustup.rs/)
- **Xcode Command Line Tools** - `xcode-select --install`
- **Swift 5.9+** - Included with Xcode

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/Era-Laboratories/voiceflow.git
cd voiceflow

# Build the Rust library
cargo build --release

# Build the macOS app
cd VoiceFlowApp
./build.sh

# Run tests
cargo test --workspace
```

### Download Models

Before running, you need to download the ML models:

```bash
# Using the CLI
cargo run --package voiceflow-cli -- setup

# Or manually download to ~/Library/Application Support/com.era-laboratories.voiceflow/models/
```

## Project Structure

```
voiceflow/
├── crates/
│   ├── voiceflow-core/     # Core Rust library
│   │   ├── src/
│   │   │   ├── audio/      # Audio capture and resampling
│   │   │   ├── config.rs   # Configuration management
│   │   │   ├── llm/        # LLM engine (mistral.rs)
│   │   │   ├── pipeline.rs # Main processing pipeline
│   │   │   ├── prosody/    # Voice commands, pause/pitch analysis
│   │   │   └── transcribe/ # Whisper and Moonshine engines
│   │   └── Cargo.toml
│   ├── voiceflow-cli/      # Command-line interface
│   └── voiceflow-ffi/      # C FFI for Swift bindings
├── VoiceFlowApp/           # macOS SwiftUI application
├── prompts/                # LLM system prompts
├── docs/                   # Technical documentation
└── Cargo.toml              # Workspace configuration
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              macOS SwiftUI App                      │
│  - Menu bar, hotkeys, audio recording               │
└─────────────────┬───────────────────────────────────┘
                  │ FFI (C bindings)
┌─────────────────▼───────────────────────────────────┐
│           voiceflow-ffi                             │
│  - C-compatible API                                 │
│  - Panic safety across FFI boundary                 │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│          voiceflow-core                             │
│                                                      │
│  Pipeline: Audio → STT → Prosody → LLM → Output    │
│                                                      │
│  STT Engines:     LLM Engine:      Prosody:         │
│  - Whisper        - mistral.rs     - Voice commands │
│  - Moonshine      - Qwen3/SmolLM   - Pause analysis │
│                                     - Pitch analysis │
└──────────────────────────────────────────────────────┘
```

## Development Guidelines

### Code Style

- Follow Rust idioms and conventions
- Use `cargo fmt` before committing
- Run `cargo clippy` and address warnings
- Add doc comments for public APIs

### Testing

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test --package voiceflow-core

# Run tests with output
cargo test -- --nocapture
```

### Adding New Features

1. **Discuss first** - Open an issue to discuss significant changes
2. **Branch** - Create a feature branch from `main`
3. **Test** - Add tests for new functionality
4. **Document** - Update relevant documentation
5. **PR** - Submit a pull request with clear description

### Commit Messages

Follow conventional commits:

```
feat: add energy-based pause detection
fix: correct punctuation spacing after periods
docs: update architecture diagram
test: add prosody module tests
refactor: move UI quoting to prompts module
```

## Key Areas for Contribution

### Good First Issues

- Add more voice command aliases
- Improve replacement dictionary with tech terms
- Add tests for edge cases

### Intermediate

- Implement energy-based pause detection for Moonshine
- Add prompt caching (see `docs/PROMPT_CACHING_SCOPE.md`)
- Improve error messages and recovery

### Advanced

- CUDA support for Linux
- Additional STT engine integrations
- Streaming transcription

## Testing Your Changes

### Unit Tests

```bash
cargo test --package voiceflow-core --lib
```

### Integration Tests

```bash
# Test CLI commands
cargo run --package voiceflow-cli -- file test_audio.wav

# Test with sample audio
cargo run --package voiceflow-cli -- bench
```

### macOS App Testing

```bash
cd VoiceFlowApp
./build.sh
open build/VoiceFlow.app
```

## Environment Variables

For testing and CI, you can override configuration via environment variables:

| Variable | Description |
|----------|-------------|
| `VOICEFLOW_STT_ENGINE` | `whisper` or `moonshine` |
| `VOICEFLOW_WHISPER_MODEL` | `tiny`, `base`, `small`, `medium` |
| `VOICEFLOW_LLM_TEMPERATURE` | Float between 0.0 and 2.0 |
| `VOICEFLOW_LLM_MAX_TOKENS` | Integer between 1 and 8192 |
| `VOICEFLOW_DEFAULT_CONTEXT` | `default`, `email`, `slack`, `code` |

See `crates/voiceflow-core/src/config.rs` for full list.

## Getting Help

- **Issues** - Search existing issues or create a new one
- **Discussions** - For questions and general discussion

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
