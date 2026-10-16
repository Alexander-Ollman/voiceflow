#!/usr/bin/env python3
"""Isolated, local-only formatter comparison. See benchmarks/formatting/README.md."""
import argparse, contextlib, ctypes, hashlib, io, json, os, re, statistics, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'VoiceFlowApp/build/formatting-bench'
MODELS = json.loads((ROOT / 'benchmarks/formatting/models.json').read_text())

def canonical(text):
    import unicodedata
    return ' '.join(unicodedata.normalize('NFKC', text).replace('’', "'").replace('“', '"').replace('”', '"').split())

def contains(text, span):
    prefix = r'(?<!\w)' + (r'(?<!\d[.,:/-])' if span[:1].isdigit() else '')
    suffix = r'(?!\w)' + (r'(?![.,:/-]\d)' if span[-1:].isdigit() else '')
    return re.search(prefix + re.escape(span) + suffix, text) is not None

def score(case, output):
    return dict(exact=canonical(output) in [canonical(x) for x in case['references']],
                changed=canonical(output) != canonical(case['input']),
                protected_failures=[x for x in case['protected'] if not contains(output, x)],
                empty=not output.strip())

class Native:
    def __init__(self, library):
        os.environ.update(VOICEFLOW_STT_ENGINE='qwen3_asr', VOICEFLOW_LLM_BACKEND='openai_server',
                          VOICEFLOW_LLM_SERVER_ENDPOINT='http://127.0.0.1:18080', VOICEFLOW_USE_PARAKEET='1')
        class Result(ctypes.Structure):
            _fields_ = [('success', ctypes.c_bool), ('formatted_text', ctypes.c_void_p),
                        ('raw_transcript', ctypes.c_void_p), ('error_message', ctypes.c_void_p),
                        ('transcription_ms', ctypes.c_uint64), ('llm_ms', ctypes.c_uint64), ('total_ms', ctypes.c_uint64)]
        self.lib = ctypes.CDLL(library)
        self.lib.voiceflow_init.argtypes = [ctypes.c_char_p]
        self.lib.voiceflow_init.restype = ctypes.c_void_p
        self.lib.voiceflow_destroy.argtypes = [ctypes.c_void_p]
        self.lib.voiceflow_free_result.argtypes = [Result]
        for name in ('voiceflow_format_text', 'voiceflow_format_text_deterministic'):
            fn = getattr(self.lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p] + ([ctypes.c_char_p] if name == 'voiceflow_format_text' else [])
            fn.restype = Result
        # Nonexistent explicit config gives compiled defaults rather than personal settings.
        self.handle = self.lib.voiceflow_init(str(CACHE / 'absent-default-config.toml').encode())
        if not self.handle:
            raise RuntimeError('Native initialization failed')
    def format(self, text, context='', rules=False):
        fn = self.lib.voiceflow_format_text_deterministic if rules else self.lib.voiceflow_format_text
        args = [self.handle, text.encode()] + ([] if rules else [context.encode() or None])
        result = fn(*args)
        try:
            if not result.success:
                raise RuntimeError(ctypes.string_at(result.error_message).decode() if result.error_message else 'native error')
            if not rules and result.llm_ms == 0:
                raise RuntimeError('Native LLM returned zero formatting time; possible silent fallback')
            return ctypes.string_at(result.formatted_text).decode()
        finally:
            self.lib.voiceflow_free_result(result)
    def close(self):
        self.lib.voiceflow_destroy(self.handle)

# General guard: no fixture/reference/protected-span information is consulted.
def pcs_guard(text, candidate):
    words = lambda s: re.findall(r"\w+(?:['’]\w+)?", s.casefold())
    if words(text) != words(candidate):
        return text
    return candidate

def adapter(name, library):
    if name == 'identity': return lambda c: c['input'], lambda: None
    if name in ('rules', 'bonsai'):
        server = None
        log = None
        if name == 'bonsai':
            import socket, urllib.request
            # Fail closed if this benchmark port is in use; never reuse an unknown service.
            with socket.socket() as check:
                check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                check.bind(('127.0.0.1', 18080))
            log = (CACHE / 'bonsai-server.log').open('w')
            server = subprocess.Popen(['/Applications/VoiceFlow.app/Contents/MacOS/llama-server',
                '-m', str(Path.home() / 'Library/Application Support/com.era-laboratories.voiceflow/models/Bonsai-8B-Q1_0.gguf'),
                '--host', '127.0.0.1', '--port', '18080', '-c', '16384', '-ngl', '99',
                '-ctk', 'q8_0', '-ctv', 'q8_0', '--seed', '42'], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if server.poll() is not None: raise RuntimeError('Bonsai server exited; see cached log')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=1) as response:
                            if response.status == 200: break
                    except OSError: time.sleep(.2)
                else: raise TimeoutError('Bonsai startup timeout')
            except BaseException:
                server.terminate(); server.wait(timeout=10); log.close(); raise
        try:
            native = Native(library)
        except BaseException:
            if server: server.terminate(); server.wait(timeout=10); log.close()
            raise
        def close():
            try: native.close()
            finally:
                if server:
                    server.terminate()
                    try: server.wait(timeout=10)
                    except subprocess.TimeoutExpired: server.kill(); server.wait()
                    log.close()
        return lambda c: native.format(c['input'], c.get('context', ''), name == 'rules'), close
    import torch
    torch.set_num_threads(4)
    key = 'pcs' if name.startswith('pcs') else name
    paths = {key: CACHE_PATH(MODELS[key])}
    if name.startswith('pcs'):
        import onnxruntime as ort
        from unittest.mock import patch
        from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX, PunctCapSegModelONNX
        original = ort.InferenceSession
        def session(path):
            opts = ort.SessionOptions(); opts.intra_op_num_threads = 4; opts.inter_op_num_threads = 1
            return original(path, sess_options=opts, providers=['CPUExecutionProvider'])
        with patch('onnxruntime.InferenceSession', session):
            model = PunctCapSegModelONNX(PunctCapSegConfigONNX(directory=str(paths['pcs']), spe_filename='spe_32k_lc_en.model', model_filename='punct_cap_seg_en.onnx'))
        guarded = name == 'pcs_guarded'
        def infer(case):
            text = case['input']
            if guarded and (re.search(r'https?://|@|[`_/]|\d|\b\w*[a-z][A-Z]\w*\b', text) or (text[:1].isupper() and text.rstrip().endswith(('.', '?', '!')))):
                return text
            # The checkpoint is explicitly trained on lowercase, unpunctuated text.
            normalized = ' '.join(re.findall(r"\w+(?:['’]\w+)?", text.lower()))
            candidate = model.infer([normalized], apply_sbd=False, num_workers=0)[0]
            return pcs_guard(text, candidate) if guarded else candidate
        return infer, lambda: None
    if name == 'gector':
        from unittest.mock import patch
        from transformers import RobertaConfig, RobertaModel, AutoTokenizer
        from safetensors.torch import load_file
        revision = subprocess.check_output(['git', '-C', str(CACHE / 'gector-source'), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != MODELS['gector']['source_revision']: raise RuntimeError('Unexpected GECToR source revision')
        sys.path.insert(0, str(CACHE / 'gector-source/src'))
        from gector.modeling import GECToR
        from gector.configuration import GECToRConfig
        from gector.predict import predict, load_verb_dict
        cfg = GECToRConfig.from_pretrained(paths['gector']); cfg.model_id = str(paths['gector'])
        cfg.max_length = json.loads((paths['gector'] / 'config.json').read_text())['max_length']
        with patch('gector.modeling.AutoModel.from_pretrained', lambda *a, **k: RobertaModel(RobertaConfig(max_position_embeddings=514, type_vocab_size=1), add_pooling_layer=False)):
            model = GECToR(cfg)
        state = load_file(str(paths['gector'] / 'model.safetensors'))
        # Older transformers serialized this deterministic, nonlearned buffer.
        position_ids = state.pop('bert.embeddings.position_ids')
        assert torch.equal(position_ids, model.bert.embeddings.position_ids)
        model.load_state_dict(state, strict=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(paths['gector'])
        encode, decode = load_verb_dict(str(CACHE / 'verb-form-vocab.txt'))
        def infer(case):
            infer.last_metadata = {'input_subwords_exceed_80': len(tokenizer(['$START'] + case['input'].split(' '), is_split_into_words=True)['input_ids']) > cfg.max_length}
            with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
                return predict(model, tokenizer, [case['input']], encode, decode, keep_confidence=.3, min_error_prob=.6, n_iteration=5, batch_size=1)[0]
        return infer, lambda: None
    if name == 'coedit':
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(paths['coedit'])
        model = AutoModelForSeq2SeqLM.from_pretrained(paths['coedit'], local_files_only=True).eval()
        def infer(case):
            batch = tokenizer('Fix grammatical errors in this sentence: ' + case['input'], return_tensors='pt')
            with torch.inference_mode():
                result = model.generate(**batch, max_new_tokens=128, do_sample=False, num_beams=1)
            infer.last_metadata = {'generation_limit_reached': int(result[0][-1]) != tokenizer.eos_token_id}
            return tokenizer.decode(result[0], skip_special_tokens=True)
        return infer, lambda: None
    raise ValueError(name)

def CACHE_PATH(model):
    from huggingface_hub import snapshot_download
    patterns = ['*.json', '*.txt', '*.model', '*.yaml', '*.onnx', '*.safetensors']
    return Path(snapshot_download(model['repo'], revision=model['revision'], allow_patterns=patterns, local_files_only=True))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', required=True, choices=['identity', 'rules', 'pcs', 'pcs_guarded', 'gector', 'coedit', 'bonsai'])
    p.add_argument('--fixtures', type=Path, default=ROOT / 'benchmarks/formatting/fixtures.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--library', default=str(ROOT / 'target/release/libvoiceflow_ffi.dylib'))
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    cases = [json.loads(x) for x in a.fixtures.read_text().splitlines() if x.strip()]
    for c in cases:
        assert all(contains(c['input'], x) for x in c['protected']), c['id']
    import psutil
    done = threading.Event(); peaks = [psutil.Process().memory_info().rss]
    def sample():
        process = psutil.Process()
        while not done.wait(.02):
            try: peaks.append(sum(x.memory_info().rss for x in [process] + process.children(recursive=True) if x.is_running()))
            except psutil.Error: pass
    threading.Thread(target=sample, daemon=True).start()
    start = time.perf_counter()
    infer, close = adapter(a.backend, a.library)
    load = time.perf_counter() - start
    rows = []
    try:
        first = time.perf_counter(); infer(cases[0]); first = time.perf_counter() - first
        for case in cases:
            outputs, durations, errors = [], [], []
            for _ in range(a.repeats):
                start = time.perf_counter()
                try: output = infer(case)
                except Exception as e: output = ''; errors.append(str(e))
                durations.append((time.perf_counter() - start) * 1000); outputs.append(output)
            rows.append(dict(case, output=outputs[0], outputs=outputs, latency_ms=durations, errors=errors, **score(case, outputs[0])))
            print(a.backend, case['id'], round(statistics.median(durations), 1), flush=True)
    finally:
        peaks.append(sum(x.memory_info().rss for x in [psutil.Process()] + psutil.Process().children(recursive=True) if x.is_running()))
        close(); done.set()
    result = dict(backend=a.backend, load_seconds=load, first_inference_seconds=first,
                  peak_process_tree_rss_bytes=max(peaks, default=0), repeats=a.repeats,
                  fixture_sha256=hashlib.sha256(a.fixtures.read_bytes()).hexdigest(),
                  native_library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest() if a.backend in ('rules', 'bonsai') else None,
                  python=sys.version, platform=__import__('platform').platform(), rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')

if __name__ == '__main__': main()
