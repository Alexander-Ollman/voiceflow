#!/bin/bash
# Download models for VoiceFlow

set -e

MODELS_DIR="${HOME}/.local/share/voiceflow/models"
mkdir -p "$MODELS_DIR"

echo "VoiceFlow Model Downloader"
echo "=========================="
echo ""
echo "Models directory: $MODELS_DIR"
echo ""

# Whisper models
WHISPER_BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

download_whisper() {
    local model=$1
    local filename="ggml-${model}.bin"
    local path="${MODELS_DIR}/${filename}"

    if [ -f "$path" ]; then
        echo "✓ Whisper ${model} already downloaded"
    else
        echo "⬇ Downloading Whisper ${model}..."
        curl -L "${WHISPER_BASE_URL}/${filename}" -o "$path" --progress-bar
        echo "✓ Whisper ${model} downloaded"
    fi
}

# LLM models
download_qwen3() {
    local filename="qwen3-1.7b-q4_k_m.gguf"
    local path="${MODELS_DIR}/${filename}"

    if [ -f "$path" ]; then
        echo "✓ Qwen3 1.7B already downloaded"
    else
        echo "⬇ Downloading Qwen3 1.7B..."
        # Note: Actual URL may need updating based on HF repo structure
        curl -L "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/${filename}" -o "$path" --progress-bar
        echo "✓ Qwen3 1.7B downloaded"
    fi
}

download_smollm3() {
    local filename="smollm3-3b-q4_k_m.gguf"
    local path="${MODELS_DIR}/${filename}"

    if [ -f "$path" ]; then
        echo "✓ SmolLM3 3B already downloaded"
    else
        echo "⬇ Downloading SmolLM3 3B..."
        curl -L "https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/resolve/main/SmolLM3-3B-Q4_K_M.gguf" -o "$path" --progress-bar
        echo "✓ SmolLM3 3B downloaded"
    fi
}

# Parse arguments
WHISPER_MODEL="${1:-base}"
LLM_MODEL="${2:-qwen3}"

echo "Selected Whisper model: ${WHISPER_MODEL}"
echo "Selected LLM model: ${LLM_MODEL}"
echo ""

# Download Whisper
download_whisper "$WHISPER_MODEL"

# Download LLM
case "$LLM_MODEL" in
    qwen3|qwen)
        download_qwen3
        ;;
    smollm3|smollm)
        download_smollm3
        ;;
    *)
        echo "Unknown LLM model: $LLM_MODEL"
        echo "Available: qwen3, smollm3"
        exit 1
        ;;
esac

echo ""
echo "✓ All models downloaded!"
echo ""
echo "Run 'voiceflow record' to start using VoiceFlow"
