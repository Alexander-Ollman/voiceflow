"""Exercise the short-clip guard without loading MLX or downloading a model."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("voiceflow_parakeet", Path(__file__).with_name("parakeet_asr_daemon.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ShortAudioTests(unittest.TestCase):
    def test_short_audio_never_reaches_metal(self):
        daemon = module.ParakeetDaemon()
        daemon.model = types.SimpleNamespace(preprocessor_config=types.SimpleNamespace(sample_rate=16000))
        fake_mx = types.ModuleType("mlx.core")
        fake_mx.array = lambda _: self.fail("A short clip reached Metal")
        fake_audio = types.ModuleType("parakeet_mlx.audio")
        fake_audio.get_logmel = lambda *_: self.fail("A short clip reached STFT")
        fake_modules = {"numpy": types.ModuleType("numpy"), "mlx": types.ModuleType("mlx"), "mlx.core": fake_mx,
                        "parakeet_mlx": types.ModuleType("parakeet_mlx"), "parakeet_mlx.audio": fake_audio}
        for length in (0, 1, 256, 3199):
            with self.subTest(length=length), patch.dict(sys.modules, fake_modules), patch.object(module, "_decode_wav_to_float32", return_value=[0.0] * length):
                result = daemon.transcribe("AA==")
                self.assertEqual(result, {"status": "ok", "text": "", "elapsed_ms": 0})

    def test_missing_model_still_reports_error(self):
        self.assertEqual(module.ParakeetDaemon().transcribe("AA==")["status"], "error")


if __name__ == "__main__":
    unittest.main()
