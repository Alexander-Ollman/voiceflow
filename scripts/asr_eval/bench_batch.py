#!/usr/bin/env python3
"""Batch (offline, full-clip) ASR benchmark: one engine/model per process.

Measures, in a fresh process:
  * load_s: import + model construction + weight load (disk cache warm after the
    first download; true first-launch download time is not included)
  * per-clip transcription latency (WAV bytes already in memory -> text),
    matching what the VoiceFlow daemon does after hotkey release
  * peak RSS (ru_maxrss) and MLX peak Metal/unified allocation (mx.get_peak_memory)

Engines:
  parakeet  -- parakeet-mlx, same code path as scripts/parakeet_asr_daemon.py
               (get_logmel + model.generate)
  mlxaudio  -- mlx-audio nemotron_asr Model.generate(..., chunk_duration=None)

Usage:
  python bench_batch.py --engine parakeet --model mlx-community/parakeet-tdt-0.6b-v2 --label parakeet-v2
  python bench_batch.py --engine mlxaudio --model nvidia/nemotron-speech-streaming-en-0.6b --label streaming-en-offline
  python bench_batch.py --engine mlxaudio --model mlx-community/nemotron-labs-unified-asr-0.6b-bf16 --language en --label unified-bf16
"""
import argparse, json, resource, sys, time, wave
from pathlib import Path

t_proc = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VoiceFlowApp/build/asr-eval'


def read_pcm(path):
    import numpy as np
    with wave.open(path, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype('float32') / 32768.0


def load_engine(engine, model_id, language, att):
    import mlx.core as mx
    if engine == 'parakeet':
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel
        model = from_pretrained(model_id)

        def run(audio):
            mel = get_logmel(mx.array(audio), model.preprocessor_config)
            res = model.generate(mel)
            return res[0].text.strip() if res else ''
        return model, run
    if engine == 'mlxaudio':
        from mlx_audio.stt import load
        model = load(model_id)
        kw = {}
        if language: kw['language'] = language
        if att: kw['att_context_size'] = att

        def run(audio):
            return model.generate(mx.array(audio), chunk_duration=None, **kw).text.strip()
        return model, run
    raise SystemExit(f'unknown engine {engine}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--language')
    p.add_argument('--att', help='att_context_size as "left,right" (mlxaudio only)')
    p.add_argument('--repeats', type=int, default=3, help='timed repeats per TTS clip for latency')
    a = p.parse_args()
    att = [int(x) for x in a.att.split(',')] if a.att else None

    import mlx.core as mx
    t0 = time.monotonic()
    model, run = load_engine(a.engine, a.model, a.language, att)
    mx.eval(model.parameters())
    load_s = time.monotonic() - t0
    proc_to_loaded = time.monotonic() - t_proc
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    rows = [json.loads(l) for l in (OUT / 'manifest.jsonl').read_text().splitlines()]
    # Warm-up (first call compiles kernels): one short + one long clip, not timed.
    t_first = time.monotonic(); run(read_pcm(rows[0]['path'])); first_call_s = time.monotonic() - t_first
    run(read_pcm(next(r for r in rows if r['category'] == 'long')['path']))
    mx.reset_peak_memory()

    out = []
    for r in rows:
        audio = read_pcm(r['path'])
        reps = a.repeats if r['part'] == 'tts' else 1
        times, text = [], None
        for _ in range(reps):
            t = time.monotonic(); text = run(audio); times.append(time.monotonic() - t)
        out.append(dict(id=r['id'], hyp=text, latency_ms=[round(x * 1000, 1) for x in times]))
        print(f"{r['id']:40s} {min(times)*1000:7.1f}ms  {text[:70]!r}", flush=True)

    meta = dict(label=a.label, engine=a.engine, model=a.model, language=a.language, att_context_size=att,
                load_s=round(load_s, 2), process_start_to_loaded_s=round(proc_to_loaded, 2),
                first_call_s=round(first_call_s, 3),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                rss_after_load_mb=round(rss_after_load / 2**20, 1),
                mlx_peak_during_transcribe_mb=round(mx.get_peak_memory() / 2**20, 1),
                mlx_active_mb=round(mx.get_active_memory() / 2**20, 1))
    for pkg in ['mlx', 'parakeet-mlx', 'mlx-audio']:
        try:
            import importlib.metadata as md; meta[pkg] = md.version(pkg)
        except Exception: pass
    dest = OUT / 'results' / f'batch-{a.label}.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
