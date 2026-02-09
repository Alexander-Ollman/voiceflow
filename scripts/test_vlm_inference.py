#!/usr/bin/env python3
"""
Test VLM model loading and inference for VoiceFlow.

Verifies that both jinaai/jina-vlm and Qwen/Qwen3-VL-2B-Instruct can:
1. Load from local model directory
2. Run text-based inference (formatting dictation output)

Uses the existing ASR pipeline (Moonshine) for initial transcription,
then passes the raw transcript through each VLM for formatting.

Usage:
    python3 scripts/test_vlm_inference.py [--audio path/to/audio.wav]
"""

import argparse
import os
import sys
import time
import subprocess
import json

# Paths
MODELS_DIR = os.path.expanduser(
    "~/Library/Application Support/com.era-laboratories.voiceflow/models"
)
JINA_VLM_DIR = os.path.join(MODELS_DIR, "jina-vlm")
QWEN3_VL_DIR = os.path.join(MODELS_DIR, "qwen3-vl-2b-instruct")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def detect_device():
    """Detect best available device."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_with_moonshine(audio_path: str) -> str:
    """
    Transcribe audio using the VoiceFlow Rust pipeline (Moonshine STT).
    Falls back to a simple placeholder if the binary isn't available.
    """
    # Try the compiled CLI
    cli_path = os.path.join(
        os.path.dirname(__file__), "..", "target", "release", "voiceflow"
    )
    if not os.path.exists(cli_path):
        # Try debug build
        cli_path = os.path.join(
            os.path.dirname(__file__), "..", "target", "debug", "voiceflow"
        )

    if os.path.exists(cli_path):
        try:
            result = subprocess.run(
                [cli_path, "file", audio_path, "--raw"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            print(f"  Warning: CLI transcription failed: {e}")

    # Fallback: use a representative dictation sample
    print("  Using sample transcription (CLI not available)")
    return "hello this is a test of the voice flow dictation system it should format this text properly with punctuation and capitalization"


def check_model_files(model_dir: str, required_files: list[str]) -> bool:
    """Check that all required model files are present."""
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        print(f"  Missing files: {missing}")
        return False
    return True


def test_qwen3_vl(transcript: str, device: str) -> dict:
    """Test Qwen3-VL-2B-Instruct loading and inference."""
    print("\n" + "=" * 60)
    print("Testing: Qwen3-VL-2B-Instruct")
    print("=" * 60)

    required = [
        "config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "preprocessor_config.json",
    ]
    if not check_model_files(QWEN3_VL_DIR, required):
        return {"status": "error", "error": "Model files missing"}

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    results = {}

    # --- Load ---
    print(f"  Loading model from: {QWEN3_VL_DIR}")
    print(f"  Device: {device}")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(QWEN3_VL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        QWEN3_VL_DIR,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device if device != "mps" else None,
    )
    if device == "mps":
        model = model.to(device)

    load_time = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded in {load_time:.1f}s ({param_count:.2f}B params)")
    results["load_time_s"] = round(load_time, 1)
    results["params_b"] = round(param_count, 2)

    # --- Inference (text-only, dictation formatting) ---
    prompt = (
        "You are a dictation formatting assistant. "
        "Take the following raw speech transcript and output ONLY the properly "
        "formatted text with correct punctuation, capitalization, and spacing. "
        "Do not add or remove words.\n\n"
        f"Raw transcript: {transcript}\n\n"
        "Formatted text:"
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], return_tensors="pt", padding=True).to(model.device)

    print("  Running inference...")
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
        )
    infer_time = time.time() - t0

    # Decode only the new tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    formatted = processor.decode(generated, skip_special_tokens=True).strip()

    token_count = len(generated)
    tokens_per_sec = token_count / infer_time if infer_time > 0 else 0

    print(f"  Inference: {infer_time:.1f}s ({token_count} tokens, {tokens_per_sec:.1f} tok/s)")
    print(f"  Output: {formatted}")

    results["status"] = "ok"
    results["infer_time_s"] = round(infer_time, 1)
    results["tokens"] = token_count
    results["tokens_per_sec"] = round(tokens_per_sec, 1)
    results["output"] = formatted

    # Cleanup
    del model, processor
    import gc; gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return results


def _ensure_jina_dynamic_modules():
    """
    Copy Jina VLM custom Python modules into the transformers dynamic module
    cache so trust_remote_code=True can resolve relative imports when loading
    from a local directory.
    """
    import shutil
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/jina_hyphen_vlm"
    )
    os.makedirs(cache_dir, exist_ok=True)

    py_files = [
        "modeling_jvlm.py",
        "blocks_jvlm.py",
        "configuration_jvlm.py",
        "image_processing_jvlm.py",
        "processing_jvlm.py",
    ]
    for f in py_files:
        src = os.path.join(JINA_VLM_DIR, f)
        dst = os.path.join(cache_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def test_jina_vlm(transcript: str, device: str) -> dict:
    """Test Jina VLM loading and inference."""
    print("\n" + "=" * 60)
    print("Testing: Jina VLM")
    print("=" * 60)

    required = [
        "config.json", "model.safetensors.index.json",
        "model-00001-of-00003.safetensors", "tokenizer.json",
        "modeling_jvlm.py",
    ]
    if not check_model_files(JINA_VLM_DIR, required):
        return {"status": "error", "error": "Model files missing"}

    _ensure_jina_dynamic_modules()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}

    # --- Load ---
    print(f"  Loading model from: {JINA_VLM_DIR}")
    print(f"  Device: {device}")
    t0 = time.time()

    # Use AutoTokenizer instead of AutoProcessor to avoid custom processor
    # compatibility issues with transformers 5.x (CommonKwargs removed).
    # For text-only inference, the tokenizer is sufficient.
    tokenizer = AutoTokenizer.from_pretrained(JINA_VLM_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        JINA_VLM_DIR,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device if device != "mps" else None,
        trust_remote_code=True,
    )
    if device == "mps":
        model = model.to(device)

    load_time = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded in {load_time:.1f}s ({param_count:.2f}B params)")
    results["load_time_s"] = round(load_time, 1)
    results["params_b"] = round(param_count, 2)

    # --- Inference (text-only, dictation formatting) ---
    prompt = (
        "You are a dictation formatting assistant. "
        "Take the following raw speech transcript and output ONLY the properly "
        "formatted text with correct punctuation, capitalization, and spacing. "
        "Do not add or remove words.\n\n"
        f"Raw transcript: {transcript}\n\n"
        "Formatted text:"
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    print("  Running inference...")
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
        )
    infer_time = time.time() - t0

    # Decode only the new tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    formatted = tokenizer.decode(generated, skip_special_tokens=True).strip()

    token_count = len(generated)
    tokens_per_sec = token_count / infer_time if infer_time > 0 else 0

    print(f"  Inference: {infer_time:.1f}s ({token_count} tokens, {tokens_per_sec:.1f} tok/s)")
    print(f"  Output: {formatted}")

    results["status"] = "ok"
    results["infer_time_s"] = round(infer_time, 1)
    results["tokens"] = token_count
    results["tokens_per_sec"] = round(tokens_per_sec, 1)
    results["output"] = formatted

    # Cleanup
    del model, tokenizer
    import gc; gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Test VLM inference for VoiceFlow")
    parser.add_argument("--audio", default="test_audio.wav", help="Path to test audio file")
    args = parser.parse_args()

    print("VoiceFlow VLM Inference Test")
    print("=" * 60)

    device = detect_device()
    print(f"Device: {device}")
    print(f"Models dir: {MODELS_DIR}")
    print()

    # Step 1: Transcribe with ASR
    print("Step 1: Transcribing audio with Moonshine ASR")
    audio_path = os.path.join(os.path.dirname(__file__), "..", args.audio)
    if not os.path.isabs(args.audio) and not os.path.exists(args.audio):
        audio_path = os.path.normpath(audio_path)
    else:
        audio_path = args.audio

    transcript = transcribe_with_moonshine(audio_path)
    print(f"  Raw transcript: {transcript}")

    # Step 2: Test each VLM
    all_results = {"transcript": transcript, "device": device, "models": {}}

    # Test Qwen3-VL first (smaller, faster)
    qwen_results = test_qwen3_vl(transcript, device)
    all_results["models"]["qwen3-vl-2b"] = qwen_results

    # Test Jina VLM
    jina_results = test_jina_vlm(transcript, device)
    all_results["models"]["jina-vlm"] = jina_results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Raw transcript: {transcript}")
    print()

    for name, res in all_results["models"].items():
        status = res.get("status", "unknown")
        if status == "ok":
            print(f"  {name}: PASS")
            print(f"    Load: {res['load_time_s']}s | Infer: {res['infer_time_s']}s | {res['tokens_per_sec']} tok/s")
            print(f"    Output: {res['output'][:100]}...")
        else:
            print(f"  {name}: FAIL - {res.get('error', 'unknown error')}")
    print()

    # Overall pass/fail
    passed = all(r.get("status") == "ok" for r in all_results["models"].values())
    if passed:
        print("Result: ALL TESTS PASSED")
    else:
        print("Result: SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
