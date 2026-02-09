#!/usr/bin/env python3
"""
Qwen3-ASR daemon for VoiceFlow.

Persistent Python process that loads a Qwen3-ASR model and serves
transcription requests over a Unix domain socket using length-prefixed JSON.
Also supports VLM (Vision-Language Model) for visual context analysis.

Protocol:
  - Each message is: 4-byte big-endian length + JSON payload
  - Commands: check, status, preload, unload, transcribe,
              preload_vlm, unload_vlm, analyze_image, shutdown

Usage:
  python3 scripts/qwen3_asr_daemon.py

Dependencies:
  pip install qwen-asr torch soundfile transformers pillow
"""

import json
import os
import signal
import socket
import struct
import sys
import tempfile
import threading
import time
import traceback
import base64
import logging
from io import BytesIO

# Configure logging
LOG_PATH = "/tmp/voiceflow_daemon.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("qwen3_asr_daemon")

SOCKET_PATH = "/tmp/voiceflow_qwen3_asr_daemon.sock"
PID_FILE = "/tmp/voiceflow_qwen3_asr_daemon.pid"

# Environment setup for MPS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class Qwen3ASRDaemon:
    def __init__(self):
        # ASR model state
        self.model = None
        self.model_path = None
        self.device = None
        self.asr_lock = threading.Lock()

        # VLM model state
        self.vlm_model = None
        self.vlm_processor = None
        self.vlm_model_path = None
        self.vlm_model_type = None
        self.vlm_lock = threading.Lock()

        self.running = True

    def detect_device(self):
        """Detect best available device (MPS > CPU)."""
        try:
            import torch
            if torch.backends.mps.is_available():
                log.info("Using MPS (Apple Silicon) device")
                return "mps"
        except Exception:
            pass
        log.info("Using CPU device")
        return "cpu"

    def load_model(self, model_path: str, dtype: str = "bfloat16") -> dict:
        """Load the Qwen3-ASR model from the given path."""
        with self.asr_lock:
            if self.model is not None and self.model_path == model_path:
                return {"status": "ok", "message": "Model already loaded"}

            try:
                log.info(f"Loading model from: {model_path} (dtype={dtype})")
                self.device = self.detect_device()

                import torch
                from qwen_asr import Qwen3ASRModel

                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                torch_dtype = dtype_map.get(dtype, torch.bfloat16)

                self.model = Qwen3ASRModel.from_pretrained(
                    model_path,
                    dtype=torch_dtype,
                    device_map=self.device,
                )
                self.model_path = model_path
                self.dtype = dtype
                log.info(f"Model loaded successfully (dtype={dtype}, device={self.device})")
                return {"status": "ok", "message": f"Model loaded ({dtype})"}
            except Exception as e:
                log.error(f"Failed to load model: {e}")
                self.model = None
                self.model_path = None
                return {"status": "error", "message": str(e)}

    def unload_model(self) -> dict:
        """Unload the current model from memory."""
        with self.asr_lock:
            if self.model is None:
                return {"status": "ok", "message": "No model loaded"}
            try:
                del self.model
                self.model = None
                self.model_path = None

                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                import gc
                gc.collect()

                log.info("Model unloaded")
                return {"status": "ok", "message": "Model unloaded"}
            except Exception as e:
                log.error(f"Failed to unload model: {e}")
                return {"status": "error", "message": str(e)}

    def transcribe(self, audio_b64: str) -> dict:
        """Transcribe base64-encoded WAV audio."""
        with self.asr_lock:
            if self.model is None:
                return {"status": "error", "message": "No model loaded"}

            tmp_path = None
            try:
                audio_bytes = base64.b64decode(audio_b64)

                # Write to temp WAV file
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                start = time.monotonic()
                result = self.model.transcribe(audio=tmp_path)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                # Extract text from ASRTranscription result
                if isinstance(result, str):
                    text = result
                elif isinstance(result, list):
                    # qwen-asr returns list of ASRTranscription objects
                    parts = []
                    for item in result:
                        if hasattr(item, "text"):
                            parts.append(item.text)
                        else:
                            parts.append(str(item))
                    text = " ".join(parts).strip()
                elif hasattr(result, "text"):
                    text = result.text
                else:
                    text = str(result)
                log.info(
                    f"Transcribed in {elapsed_ms}ms: "
                    f"{text[:80]}{'...' if len(text) > 80 else ''}"
                )

                return {
                    "status": "ok",
                    "text": text,
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as e:
                log.error(f"Transcription failed: {e}\n{traceback.format_exc()}")
                return {"status": "error", "message": str(e)}
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # =========================================================================
    # VLM (Vision-Language Model) Support
    # =========================================================================

    def load_vlm_model(self, model_path: str, model_type: str = "qwen3-vl-2b") -> dict:
        """Load a VLM model for visual context analysis."""
        with self.vlm_lock:
            if self.vlm_model is not None and self.vlm_model_path == model_path:
                return {"status": "ok", "message": "VLM already loaded"}

            try:
                log.info(f"Loading VLM from: {model_path} (type={model_type})")
                device = self.detect_device()

                import torch
                from transformers import AutoProcessor

                if model_type == "jina-vlm":
                    from transformers import AutoModelForCausalLM
                    self.vlm_processor = AutoProcessor.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                    self.vlm_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                    )
                else:
                    # Default: Qwen3-VL style
                    from transformers import AutoModelForVision2Seq
                    self.vlm_processor = AutoProcessor.from_pretrained(model_path)
                    self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                    )

                if device == "mps":
                    self.vlm_model = self.vlm_model.to("mps")

                self.vlm_model_path = model_path
                self.vlm_model_type = model_type
                log.info(f"VLM loaded successfully (type={model_type}, device={device})")
                return {"status": "ok", "message": f"VLM loaded ({model_type})"}
            except Exception as e:
                log.error(f"Failed to load VLM: {e}\n{traceback.format_exc()}")
                self.vlm_model = None
                self.vlm_processor = None
                self.vlm_model_path = None
                return {"status": "error", "message": str(e)}

    def unload_vlm_model(self) -> dict:
        """Unload the VLM model from memory."""
        with self.vlm_lock:
            if self.vlm_model is None:
                return {"status": "ok", "message": "No VLM loaded"}
            try:
                del self.vlm_model
                del self.vlm_processor
                self.vlm_model = None
                self.vlm_processor = None
                self.vlm_model_path = None
                self.vlm_model_type = None

                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                import gc
                gc.collect()

                log.info("VLM unloaded")
                return {"status": "ok", "message": "VLM unloaded"}
            except Exception as e:
                log.error(f"Failed to unload VLM: {e}")
                return {"status": "error", "message": str(e)}

    def analyze_image(self, image_b64: str, prompt: str | None = None) -> dict:
        """Analyze a base64-encoded JPEG image using the VLM."""
        with self.vlm_lock:
            if self.vlm_model is None or self.vlm_processor is None:
                return {"status": "error", "message": "No VLM loaded"}

            try:
                from PIL import Image

                image_bytes = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")

                if prompt is None:
                    prompt = (
                        "Extract the following from this screenshot as structured data. "
                        "Be concise and only include what is clearly visible.\n\n"
                        "APP: [application name]\n"
                        "CONTEXT: [what the user is doing, e.g. 'replying to a Slack thread about deployment']\n"
                        "NAMES: [proper nouns, people's names, usernames visible, comma-separated]\n"
                        "TERMS: [technical terms, product names, brand names visible, comma-separated]\n"
                        "NEARBY_TEXT: [text near where the user appears to be typing, up to 50 words]\n\n"
                        "If a field has no visible data, write 'none'. Do not add explanation."
                    )

                start = time.monotonic()

                # Build chat messages in the VLM format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # Process with the model
                text_input = self.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.vlm_processor(
                    text=[text_input],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )

                # Move to device
                device = next(self.vlm_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                import torch
                with torch.no_grad():
                    output_ids = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                    )

                # Decode only the new tokens
                input_len = inputs["input_ids"].shape[1]
                description = self.vlm_processor.decode(
                    output_ids[0][input_len:], skip_special_tokens=True
                ).strip()

                elapsed_ms = int((time.monotonic() - start) * 1000)
                log.info(
                    f"VLM analyzed image in {elapsed_ms}ms: "
                    f"{description[:80]}{'...' if len(description) > 80 else ''}"
                )

                return {
                    "status": "ok",
                    "description": description,
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as e:
                log.error(f"VLM analysis failed: {e}\n{traceback.format_exc()}")
                return {"status": "error", "message": str(e)}

    def handle_command(self, data: dict) -> dict:
        """Dispatch a command and return a response dict."""
        cmd = data.get("command", "")

        if cmd == "check":
            return {"status": "ok"}

        elif cmd == "status":
            return {
                "status": "ok",
                "model_loaded": self.model is not None,
                "model_path": self.model_path,
                "device": self.device,
                "dtype": getattr(self, "dtype", None),
                "vlm_loaded": self.vlm_model is not None,
                "vlm_model_path": self.vlm_model_path,
                "vlm_model_type": self.vlm_model_type,
                "pid": os.getpid(),
            }

        elif cmd == "preload":
            model_path = data.get("model_path")
            if not model_path:
                return {"status": "error", "message": "model_path required"}
            dtype = data.get("dtype", "bfloat16")
            return self.load_model(model_path, dtype=dtype)

        elif cmd == "unload":
            return self.unload_model()

        elif cmd == "transcribe":
            audio_b64 = data.get("audio")
            if not audio_b64:
                return {"status": "error", "message": "audio field required"}
            return self.transcribe(audio_b64)

        elif cmd == "preload_vlm":
            model_path = data.get("model_path")
            if not model_path:
                return {"status": "error", "message": "model_path required"}
            model_type = data.get("model_type", "qwen3-vl-2b")
            return self.load_vlm_model(model_path, model_type=model_type)

        elif cmd == "unload_vlm":
            return self.unload_vlm_model()

        elif cmd == "analyze_image":
            image_b64 = data.get("image")
            if not image_b64:
                return {"status": "error", "message": "image field required"}
            prompt = data.get("prompt")
            return self.analyze_image(image_b64, prompt=prompt)

        elif cmd == "shutdown":
            log.info("Shutdown requested")
            self.running = False
            return {"status": "ok", "message": "Shutting down"}

        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}


def recv_message(conn: socket.socket) -> dict | None:
    """Read a length-prefixed JSON message from a socket."""
    # Read 4-byte length prefix
    length_data = b""
    while len(length_data) < 4:
        chunk = conn.recv(4 - len(length_data))
        if not chunk:
            return None
        length_data += chunk

    msg_len = struct.unpack(">I", length_data)[0]
    if msg_len > 100 * 1024 * 1024:  # 100 MB sanity limit
        return None

    # Read the JSON payload
    payload = b""
    while len(payload) < msg_len:
        chunk = conn.recv(min(65536, msg_len - len(payload)))
        if not chunk:
            return None
        payload += chunk

    return json.loads(payload.decode("utf-8"))


def send_message(conn: socket.socket, data: dict):
    """Send a length-prefixed JSON message over a socket."""
    payload = json.dumps(data).encode("utf-8")
    length_prefix = struct.pack(">I", len(payload))
    conn.sendall(length_prefix + payload)


def handle_client(daemon: Qwen3ASRDaemon, conn: socket.socket, addr):
    """Handle a single client connection."""
    try:
        while daemon.running:
            msg = recv_message(conn)
            if msg is None:
                break

            response = daemon.handle_command(msg)
            send_message(conn, response)

            # If shutdown was requested, break after sending response
            if msg.get("command") == "shutdown":
                break
    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception as e:
        log.error(f"Client handler error: {e}")
    finally:
        conn.close()


def write_pid_file():
    """Write the current PID to the PID file."""
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def cleanup():
    """Remove socket and PID files, but only if we still own them.

    A new daemon may have already replaced these files, so check the PID file
    to avoid deleting another daemon's socket.
    """
    my_pid = os.getpid()
    try:
        with open(PID_FILE, "r") as f:
            file_pid = int(f.read().strip())
    except (OSError, ValueError):
        file_pid = my_pid  # If we can't read it, assume we own it

    if file_pid != my_pid:
        log.info(f"PID file belongs to {file_pid}, skipping cleanup (we are {my_pid})")
        return

    for path in (SOCKET_PATH, PID_FILE):
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


def main():
    log.info(f"Starting Qwen3-ASR daemon (PID {os.getpid()})")

    # Write PID file early so the old daemon's cleanup knows we took over
    write_pid_file()

    # Clean up stale socket
    if os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass

    daemon = Qwen3ASRDaemon()

    # Set up signal handlers
    def signal_handler(signum, frame):
        log.info(f"Received signal {signum}, shutting down")
        daemon.running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Create Unix domain socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(SOCKET_PATH)
    server.listen(5)
    server.settimeout(1.0)  # Allow periodic check of daemon.running

    log.info(f"Listening on {SOCKET_PATH}")

    try:
        while daemon.running:
            try:
                conn, addr = server.accept()
                thread = threading.Thread(
                    target=handle_client,
                    args=(daemon, conn, addr),
                    daemon=True,
                )
                thread.start()
            except socket.timeout:
                continue
            except OSError:
                if daemon.running:
                    raise
                break
    finally:
        log.info("Daemon shutting down")
        server.close()
        daemon.unload_model()
        daemon.unload_vlm_model()
        cleanup()
        log.info("Daemon stopped")


if __name__ == "__main__":
    main()
