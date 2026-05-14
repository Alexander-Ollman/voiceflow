#!/usr/bin/env python3
"""
Parakeet-MLX ASR daemon for VoiceFlow.

Persistent Python process that loads NVIDIA Parakeet TDT 0.6B v2 via
parakeet-mlx (MLX-accelerated on Apple Silicon) and serves transcription
requests over a Unix domain socket using length-prefixed JSON.

Mirrors the protocol of qwen3_asr_daemon.py so the Swift bridge can stay
near-identical.

Supports two modes:
  1. Batch transcribe — full clip in, full transcript out. Used for the
     final transcript that feeds the LLM (offline WER ~6.04%).
  2. Streaming — chunked feed, live finalized+draft tokens out. Used for
     live overlay display only. parakeet-tdt-0.6b-v2 is offline-trained,
     so streaming WER degrades at short chunks. Clients should pin chunk
     size at >=1s and not rely on streaming output for the final transcript.

Protocol:
  - Each message is: 4-byte big-endian length + JSON payload
  - Batch commands: check, status, preload, unload, transcribe, shutdown
  - Streaming commands (model-agnostic, any future ASR daemon can implement):
      stream_open  {sample_rate: int}
        -> {status:"ok", session_id, sample_rate}
      stream_feed  {session_id, audio: <b64 int16 LE PCM at sample_rate>}
        -> {status:"ok", finalized: str, draft: str}
      stream_close {session_id}
        -> {status:"ok", duration_ms: int}
    Only one stream session exists at a time. Opening a new one
    auto-closes any prior session (orphan recovery on Swift crash).

Usage:
  python3 scripts/parakeet_asr_daemon.py

Dependencies:
  pip install parakeet-mlx
"""

import base64
import io
import json
import logging
import os
import signal
import socket
import struct
import sys
import threading
import time
import traceback
import uuid
import wave

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

SOCKET_PATH = os.environ.get("VOICEFLOW_PARAKEET_SOCKET", "/tmp/voiceflow_parakeet_daemon.sock")
PID_FILE = os.environ.get("VOICEFLOW_PARAKEET_PID", "/tmp/voiceflow_parakeet_daemon.pid")

DEFAULT_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v2"


def _decode_wav_to_float32(audio_bytes: bytes, expected_sr: int):
    """Parse a 16-bit PCM WAV (what Swift's ParakeetASREngine.encodeAsWAV emits)
    into a float32 numpy array at expected_sr. Raises if sample rate doesn't
    match or sample width isn't 16-bit."""
    import numpy as np

    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sampwidth != 2:
        raise RuntimeError(f"unsupported sample width {sampwidth*8}-bit; expected 16-bit PCM")
    if sr != expected_sr:
        raise RuntimeError(f"sample rate {sr}Hz != expected {expected_sr}Hz")

    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    return pcm.astype(np.float32) / 32768.0


class ParakeetDaemon:
    def __init__(self):
        self.model = None
        self.model_id = None
        self.lock = threading.Lock()
        self.running = True

        self.stream_ctx = None
        self.stream_transcriber = None
        self.stream_session_id = None
        self.stream_opened_at = 0.0
        self._token_api_logged = False

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
        if self.stream_transcriber is not None:
            log.info("Closing active stream before model unload")
            self._stream_force_close()
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

        try:
            audio_bytes = base64.b64decode(audio_b64)

            # Decode WAV in-process. parakeet-mlx's model.transcribe(path) goes
            # through load_audio() which shells out to ffmpeg; the daemon's PATH
            # doesn't include /opt/homebrew/bin, so that path 500s. Decoding the
            # WAV ourselves and calling generate(mel) directly matches the
            # streaming path (which also feeds raw float PCM) and removes the
            # ffmpeg dependency entirely.
            import numpy as np
            import mlx.core as mx
            from parakeet_mlx.audio import get_logmel

            expected_sr = int(model.preprocessor_config.sample_rate)
            audio = _decode_wav_to_float32(audio_bytes, expected_sr)
            audio_mx = mx.array(audio)

            start = time.monotonic()
            mel = get_logmel(audio_mx, model.preprocessor_config)
            results = model.generate(mel)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            result = results[0] if results else None
            text = getattr(result, "text", None) if result is not None else None
            if text is None:
                text = "" if result is None else str(result)
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

    def stream_open(self, sample_rate: int) -> dict:
        with self.lock:
            model = self.model
        if model is None:
            return {"status": "error", "message": "No model loaded"}

        if self.stream_transcriber is not None:
            log.info(f"Auto-closing orphan stream {self.stream_session_id}")
            self._stream_force_close()

        try:
            expected_sr = int(model.preprocessor_config.sample_rate)
            if int(sample_rate) != expected_sr:
                return {
                    "status": "error",
                    "message": f"sample_rate must be {expected_sr}, got {sample_rate}",
                }
            # Right-context of 16 frames (~1.3s) keeps finalized_tokens flowing
            # quickly enough to drive a live UI. Default (256) lags ~20s — fine
            # for offline accuracy, useless for streaming UX. Streaming output
            # is display-only here; the final transcript comes from a batch
            # transcribe so WER tradeoff is acceptable.
            self.stream_ctx = model.transcribe_stream(context_size=(256, 16))
            self.stream_transcriber = self.stream_ctx.__enter__()
            self.stream_session_id = uuid.uuid4().hex
            self.stream_opened_at = time.monotonic()
            log.info(f"Stream opened: {self.stream_session_id} @ {expected_sr}Hz")
            return {
                "status": "ok",
                "session_id": self.stream_session_id,
                "sample_rate": expected_sr,
            }
        except Exception as e:
            log.error(f"stream_open failed: {e}\n{traceback.format_exc()}")
            self._stream_force_close()
            return {"status": "error", "message": str(e)}

    def stream_feed(self, session_id: str, audio_b64: str) -> dict:
        if self.stream_transcriber is None:
            return {"status": "error", "message": "No active stream"}
        if session_id and session_id != self.stream_session_id:
            return {"status": "error", "message": "Stale session_id"}

        try:
            import numpy as np
            import mlx.core as mx

            pcm_bytes = base64.b64decode(audio_b64)
            pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio = mx.array(pcm)
            self.stream_transcriber.add_audio(audio)

            finalized = self._tokens_to_text(
                getattr(self.stream_transcriber, "finalized_tokens", None)
            )
            draft = self._tokens_to_text(
                getattr(self.stream_transcriber, "draft_tokens", None)
            )
            return {
                "status": "ok",
                "finalized": finalized,
                "draft": draft,
            }
        except Exception as e:
            log.error(f"stream_feed failed: {e}\n{traceback.format_exc()}")
            return {"status": "error", "message": str(e)}

    def stream_close(self, session_id: str) -> dict:
        if self.stream_transcriber is None:
            return {"status": "ok", "message": "No active stream", "duration_ms": 0}
        if session_id and session_id != self.stream_session_id:
            return {"status": "error", "message": "Stale session_id"}

        duration_ms = int((time.monotonic() - self.stream_opened_at) * 1000)
        sid = self.stream_session_id
        self._stream_force_close()
        log.info(f"Stream closed: {sid} ({duration_ms}ms)")
        return {"status": "ok", "duration_ms": duration_ms}

    def _stream_force_close(self):
        if self.stream_ctx is not None:
            try:
                self.stream_ctx.__exit__(None, None, None)
            except Exception as e:
                log.warning(f"stream cleanup error: {e}")
        self.stream_ctx = None
        self.stream_transcriber = None
        self.stream_session_id = None
        self.stream_opened_at = 0.0

    def _tokens_to_text(self, tokens) -> str:
        """Extract text from a parakeet-mlx token list. Defensive — logs the
        token type once so we can adjust if the API differs from expectations."""
        if not tokens:
            return ""
        if not self._token_api_logged:
            sample = tokens[0]
            log.info(
                f"token type: {type(sample).__name__}, "
                f"attrs: {[a for a in dir(sample) if not a.startswith('_')][:10]}"
            )
            self._token_api_logged = True
        parts = []
        for t in tokens:
            text = getattr(t, "text", None)
            if text is None:
                text = str(t)
            parts.append(text)
        return "".join(parts)

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

        if cmd == "stream_open":
            sample_rate = data.get("sample_rate")
            if not sample_rate:
                return {"status": "error", "message": "sample_rate field required"}
            return self.stream_open(sample_rate)

        if cmd == "stream_feed":
            session_id = data.get("session_id", "")
            audio_b64 = data.get("audio")
            if not audio_b64:
                return {"status": "error", "message": "audio field required"}
            return self.stream_feed(session_id, audio_b64)

        if cmd == "stream_close":
            session_id = data.get("session_id", "")
            return self.stream_close(session_id)

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
