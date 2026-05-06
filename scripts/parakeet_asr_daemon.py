#!/usr/bin/env python3
"""
Parakeet-MLX ASR daemon for VoiceFlow.

Persistent Python process that loads NVIDIA Parakeet TDT 0.6B v2 via
parakeet-mlx (MLX-accelerated on Apple Silicon) and serves transcription
requests over a Unix domain socket using length-prefixed JSON.

Mirrors the protocol of qwen3_asr_daemon.py so the Swift bridge can stay
near-identical. Batch transcription only (no streaming) — VoiceFlow records
the full clip, sends it on release, and gets back the final transcript.

Protocol:
  - Each message is: 4-byte big-endian length + JSON payload
  - Commands: check, status, preload, unload, transcribe, shutdown

Usage:
  python3 scripts/parakeet_asr_daemon.py

Dependencies:
  pip install parakeet-mlx
"""

import base64
import json
import logging
import os
import signal
import socket
import struct
import sys
import tempfile
import threading
import time
import traceback

LOG_PATH = "/tmp/voiceflow_parakeet_daemon.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("parakeet_asr_daemon")

SOCKET_PATH = "/tmp/voiceflow_parakeet_daemon.sock"
PID_FILE = "/tmp/voiceflow_parakeet_daemon.pid"

DEFAULT_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v2"


class ParakeetDaemon:
    def __init__(self):
        self.model = None
        self.model_id = None
        self.lock = threading.Lock()
        self.running = True

    def load_model(self, model_id: str) -> dict:
        with self.lock:
            if self.model is not None and self.model_id == model_id:
                return {"status": "ok", "message": "Model already loaded"}

            try:
                log.info(f"Loading parakeet-mlx model: {model_id}")
                from parakeet_mlx import from_pretrained

                start = time.monotonic()
                self.model = from_pretrained(model_id)
                self.model_id = model_id
                elapsed_ms = int((time.monotonic() - start) * 1000)
                log.info(f"Model loaded in {elapsed_ms}ms: {model_id}")
                return {
                    "status": "ok",
                    "message": f"Model loaded ({model_id})",
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as e:
                log.error(f"Failed to load model: {e}\n{traceback.format_exc()}")
                self.model = None
                self.model_id = None
                return {"status": "error", "message": str(e)}

    def unload_model(self) -> dict:
        with self.lock:
            if self.model is None:
                return {"status": "ok", "message": "No model loaded"}
            try:
                del self.model
                self.model = None
                self.model_id = None
                import gc
                gc.collect()
                try:
                    import mlx.core as mx
                    mx.metal.clear_cache()
                except Exception:
                    pass
                log.info("Model unloaded")
                return {"status": "ok", "message": "Model unloaded"}
            except Exception as e:
                log.error(f"Failed to unload model: {e}")
                return {"status": "error", "message": str(e)}

    def transcribe(self, audio_b64: str) -> dict:
        # Don't hold the lock across transcribe — `model.transcribe` may take
        # several hundred ms; we want a snapshot of the model and release.
        with self.lock:
            model = self.model
        if model is None:
            return {"status": "error", "message": "No model loaded"}

        tmp_path = None
        try:
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            start = time.monotonic()
            result = model.transcribe(tmp_path)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            # AlignedResult.text is the joined transcript
            text = getattr(result, "text", None)
            if text is None:
                text = str(result)
            text = text.strip()

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
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def handle_command(self, data: dict) -> dict:
        cmd = data.get("command", "")

        if cmd == "check":
            return {"status": "ok"}

        if cmd == "status":
            return {
                "status": "ok",
                "model_loaded": self.model is not None,
                "model_id": self.model_id,
                "pid": os.getpid(),
            }

        if cmd == "preload":
            model_id = data.get("model_id") or DEFAULT_MODEL_ID
            return self.load_model(model_id)

        if cmd == "unload":
            return self.unload_model()

        if cmd == "transcribe":
            audio_b64 = data.get("audio")
            if not audio_b64:
                return {"status": "error", "message": "audio field required"}
            return self.transcribe(audio_b64)

        if cmd == "shutdown":
            log.info("Shutdown requested")
            self.running = False
            return {"status": "ok", "message": "Shutting down"}

        return {"status": "error", "message": f"Unknown command: {cmd}"}


# ---------------------------------------------------------------------------
# Length-prefixed JSON socket protocol (matches qwen3_asr_daemon)
# ---------------------------------------------------------------------------

def recv_message(conn: socket.socket):
    length_data = b""
    while len(length_data) < 4:
        chunk = conn.recv(4 - len(length_data))
        if not chunk:
            return None
        length_data += chunk
    msg_len = struct.unpack(">I", length_data)[0]
    if msg_len > 100 * 1024 * 1024:
        return None
    payload = b""
    while len(payload) < msg_len:
        chunk = conn.recv(min(65536, msg_len - len(payload)))
        if not chunk:
            return None
        payload += chunk
    return json.loads(payload.decode("utf-8"))


def send_message(conn: socket.socket, data: dict):
    payload = json.dumps(data).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def handle_client(daemon: ParakeetDaemon, conn: socket.socket):
    try:
        while daemon.running:
            msg = recv_message(conn)
            if msg is None:
                break
            response = daemon.handle_command(msg)
            send_message(conn, response)
            if msg.get("command") == "shutdown":
                break
    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception as e:
        log.error(f"Client handler error: {e}")
    finally:
        conn.close()


def write_pid_file():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def cleanup():
    """Remove socket+PID files only if we still own them."""
    my_pid = os.getpid()
    try:
        with open(PID_FILE, "r") as f:
            file_pid = int(f.read().strip())
    except (OSError, ValueError):
        file_pid = my_pid
    if file_pid != my_pid:
        log.info(f"PID file owned by {file_pid}, leaving alone (we are {my_pid})")
        return
    for path in (SOCKET_PATH, PID_FILE):
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


def main():
    log.info(f"Starting Parakeet-MLX daemon (PID {os.getpid()})")
    write_pid_file()

    if os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass

    daemon = ParakeetDaemon()

    def signal_handler(signum, _frame):
        log.info(f"Received signal {signum}, shutting down")
        daemon.running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(SOCKET_PATH)
    server.listen(5)
    server.settimeout(1.0)
    log.info(f"Listening on {SOCKET_PATH}")

    try:
        while daemon.running:
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if daemon.running:
                    raise
                break
            # Handle the connection synchronously on the main thread.
            # MLX has GPU stream affinity to the thread that created tensors,
            # so all inference must stay on the load thread (this one).
            # Single-user dictation does not need per-connection parallelism.
            handle_client(daemon, conn)
    finally:
        log.info("Daemon shutting down")
        server.close()
        daemon.unload_model()
        cleanup()
        log.info("Daemon stopped")


if __name__ == "__main__":
    main()
