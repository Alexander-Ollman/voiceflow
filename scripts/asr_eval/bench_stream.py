#!/usr/bin/env python3
"""Cache-aware streaming benchmark for nemotron-speech-streaming-en-0.6b via mlx-audio.

Simulates live capture: audio is fed in mic-sized packets (default 80 ms, the
cadence a 16 kHz AVAudioEngine tap would deliver), and after each packet the
session is drained of every complete encoder chunk -- i.e. the model "keeps up"
as long as per-chunk compute < chunk duration (reported as RTF).

Per clip it records:
  * per-chunk compute (ms) for every encoder chunk processed while audio flows
  * time_to_final_ms: wall time from end-of-audio (hotkey release) to final
    text = close() + drain of the residual partial chunk and flush. This is
    the number to compare with the batch model's post-release latency.
  * the streamed final text (scored by score.py like batch results)

Usage: python bench_stream.py --delay-ms 160 [--packet-ms 80] [--model nvidia/nemotron-speech-streaming-en-0.6b]
"""
import argparse, json, resource, time, wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VoiceFlowApp/build/asr-eval'


def read_pcm(path):
    with wave.open(path, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype('float32') / 32768.0


def drain(session, chunk_samples, chunk_times):
    """Process every complete encoder chunk currently queued; time each one."""
    text = []
    if getattr(session, '_context', None) is not None:
        # Buffered (Unified): one step ingests all queued audio, re-encodes each
        # ready [left|chunk|right] window and decodes it. Time only steps that encoded.
        before = session._buffered._next
        t = time.monotonic()
        text += session.step()
        if session._buffered._next != before:
            n = (session._buffered._next - before) // session._buffered.chunk_samples
            chunk_times.extend([(time.monotonic() - t) * 1000 / n] * n)
        return text
    while True:
        enc = session._encoder
        need = (enc.first_chunk_mel if enc.consumed == 0 else enc.chunk_mel) * session.model.preprocessor_config.hop_length
        if not session._encoded and session._queued < need:
            break
        t = time.monotonic()
        while True:  # one ingest + full decode of the resulting encoder frames
            d = session.step(max_decode_tokens=64)
            text += d
            if not session._encoded:
                break
        chunk_times.append((time.monotonic() - t) * 1000)
    return text


def run_clip(model, audio, delay_ms, packet, language=None):
    import mlx.core as mx
    kw = dict(language=language) if language else {}
    s = model.create_streaming_session(transcription_delay_ms=delay_ms, **kw)
    if getattr(s, '_context', None) is not None:
        chunk_samples = s._buffered.chunk_samples
    else:
        chunk_samples = s._encoder.chunk_mel * model.preprocessor_config.hop_length
    chunk_times, text = [], []
    first_text_audio_s = None
    for i in range(0, len(audio), packet):
        s.feed(audio[i:i + packet])
        new = drain(s, chunk_samples, chunk_times)
        if new and first_text_audio_s is None:
            first_text_audio_s = min(i + packet, len(audio)) / 16000
        text += new
    t = time.monotonic()
    s.close()
    while not s.done:
        text += s.step(max_decode_tokens=64)
    ttf = (time.monotonic() - t) * 1000
    mx.clear_cache()
    return ''.join(text).strip(), chunk_times, ttf, chunk_samples / 16000, first_text_audio_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='nvidia/nemotron-speech-streaming-en-0.6b')
    p.add_argument('--delay-ms', type=int, required=True, help='80/160/560/1120 for streaming-en; 160..2080 for Unified (buffered)')
    p.add_argument('--language', help='prompt language for Unified, e.g. en')
    p.add_argument('--packet-ms', type=int, default=80)
    p.add_argument('--label')
    p.add_argument('--bf16', action='store_true', help='cast weights to bfloat16 after load')
    a = p.parse_args()
    label = a.label or f'stream-en-{a.delay_ms}ms'
    lang = a.language
    import mlx.core as mx
    from mlx_audio.stt import load
    t0 = time.monotonic(); model = load(a.model)
    if a.bf16:  # cast fp32 checkpoint weights in memory (halves resident weights)
        model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters()); load_s = time.monotonic() - t0
    packet = 16 * a.packet_ms
    rows = [json.loads(l) for l in (OUT / 'manifest.jsonl').read_text().splitlines()]
    run_clip(model, read_pcm(rows[0]['path']), a.delay_ms, packet, lang)  # warm-up
    run_clip(model, read_pcm(next(r for r in rows if r['category'] == 'long')['path']), a.delay_ms, packet, lang)
    mx.reset_peak_memory()
    out, all_chunks, ttfs = [], [], []
    for r in rows:
        audio = read_pcm(r['path'])
        text, chunks, ttf, chunk_s, first_s = run_clip(model, audio, a.delay_ms, packet, lang)
        out.append(dict(id=r['id'], hyp=text, chunk_ms=[round(c, 2) for c in chunks], time_to_final_ms=round(ttf, 1),
                        first_text_at_audio_s=first_s, audio_s=r['duration_s']))
        all_chunks += chunks[1:]; ttfs.append(ttf)
        print(f"{r['id']:40s} ttf={ttf:6.1f}ms chunks={len(chunks)} p50={np.median(chunks) if chunks else 0:5.1f}ms {text[:60]!r}", flush=True)
    pct = lambda xs, q: round(float(np.percentile(xs, q)), 1)
    meta = dict(label=label, engine='mlxaudio-stream', model=a.model, delay_ms=a.delay_ms, language=lang, att_context_size=None, packet_ms=a.packet_ms,
                encoder_chunk_s=chunk_s, load_s=round(load_s, 2),
                chunk_ms_p50=pct(all_chunks, 50), chunk_ms_p95=pct(all_chunks, 95), chunk_ms_max=pct(all_chunks, 100),
                rtf_p95=round(pct(all_chunks, 95) / (chunk_s * 1000), 3),
                time_to_final_ms_p50=pct(ttfs, 50), time_to_final_ms_p95=pct(ttfs, 95), time_to_final_ms_max=pct(ttfs, 100),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                mlx_peak_during_transcribe_mb=round(mx.get_peak_memory() / 2**20, 1))
    (OUT / 'results').mkdir(exist_ok=True)
    (OUT / 'results' / f'{label}.json').write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
