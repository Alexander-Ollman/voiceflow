#!/usr/bin/env python3
"""Reference run of nemotron-speech-streaming-en-0.6b through Hugging Face Transformers
(PyTorch, offline generate, model-card recipe). Used as (a) a parity check on the
mlx-audio port and (b) the Transformers-on-Mac runtime data point (CPU or MPS).

Usage: python bench_hf.py --device mps|cpu [--label hf-mps]
"""
import argparse, json, resource, time, wave
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'VoiceFlowApp/build/asr-eval'


def read_pcm(path):
    with wave.open(path, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype('float32') / 32768.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--label')
    p.add_argument('--model', default='nvidia/nemotron-speech-streaming-en-0.6b')
    a = p.parse_args()
    label = a.label or f'hf-streaming-en-{a.device}'
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor
    t0 = time.monotonic()
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForRNNT.from_pretrained(a.model).to(a.device).eval()
    load_s = time.monotonic() - t0
    sr = proc.feature_extractor.sampling_rate

    def run(audio):
        inputs = proc(audio, sampling_rate=sr)
        inputs.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, return_dict_in_generate=True)
        text = proc.decode(out.sequences, skip_special_tokens=True)
        return (text[0] if isinstance(text, list) else text).strip()

    rows = [json.loads(l) for l in (OUT / 'manifest.jsonl').read_text().splitlines()]
    run(read_pcm(rows[0]['path']))
    out = []
    for r in rows:
        audio = read_pcm(r['path'])
        t = time.monotonic(); text = run(audio)
        if a.device == 'mps': torch.mps.synchronize()
        dt = (time.monotonic() - t) * 1000
        out.append(dict(id=r['id'], hyp=text, latency_ms=[round(dt, 1)]))
        print(f"{r['id']:40s} {dt:7.1f}ms {text[:70]!r}", flush=True)
    import transformers
    meta = dict(label=label, engine='transformers', model=a.model, device=a.device, load_s=round(load_s, 2), first_call_s=0,
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                mlx_peak_during_transcribe_mb=float('nan'), torch=torch.__version__, transformers=transformers.__version__)
    (OUT / 'results' / f'batch-{label}.json').write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
