#!/usr/bin/env python3
"""Latency/memory table from results/*.json.

Batch: per clip take the median of its timed repeats (TTS clips have 3), then
report p50/p95 across clips per duration bucket. This is post-release latency
(full clip already captured -> text).
Stream: time_to_final_ms (end of audio -> final text) and per-chunk compute.

Usage: python latency.py results/batch-*.json results/stream-*.json
"""
import json, sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'VoiceFlowApp/build/asr-eval'
m = {json.loads(l)['id']: json.loads(l) for l in (OUT / 'manifest.jsonl').read_text().splitlines()}
BUCKETS = [('short <2s speech', lambda r: r['category'] == 'short'),
           ('2-5s', lambda r: r['part'] == 'tts' and r['category'] in ('medium', 'numeric')),
           ('10-15s', lambda r: r['category'] == 'long')]
pct = lambda xs, q: float(np.percentile(xs, q)) if xs else float('nan')

for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text()); meta = d['meta']
    cells = []
    for name, f in BUCKETS:
        rows = [r for r in d['rows'] if f(m[r['id']])]
        xs = [float(np.median(r['latency_ms'])) for r in rows] if 'latency_ms' in d['rows'][0] else [r['time_to_final_ms'] for r in rows]
        cells.append(f"{name}: p50 {pct(xs, 50):6.1f} / p95 {pct(xs, 95):6.1f} ms")
    extra = f"chunk p50/p95 {meta['chunk_ms_p50']}/{meta['chunk_ms_p95']} ms @ {meta['encoder_chunk_s']*1000:.0f} ms chunks" if 'chunk_ms_p50' in meta else f"first call {meta['first_call_s']*1000:.0f} ms"
    print(f"{meta['label']:28s} load {meta['load_s']:5.2f}s  RSS {meta['peak_rss_mb']:6.0f}MB  MLX peak {meta['mlx_peak_during_transcribe_mb']:6.0f}MB  {extra}")
    print('    ' + ' | '.join(cells))
