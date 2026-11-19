#!/usr/bin/env python3
"""Score result files from bench_batch.py / bench_stream.py against manifest references.

Two WER flavours (corpus-level: total edits / total reference words):
  * pnc  -- raw strings, whitespace tokenised, case and punctuation significant.
            Only for TTS clips (LibriSpeech references are unpunctuated upper-case).
  * norm -- Whisper EnglishTextNormalizer on both sides (lower-case, strip
            punctuation, standardise numbers/spellings), as on the Open ASR leaderboard.

Usage: python score.py results/batch-*.json results/stream-*.json   (needs jiwer, whisper-normalizer)
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VoiceFlowApp/build/asr-eval'
NORM = EnglishTextNormalizer()


def edits(ref, hyp):
    if not ref.split():
        return 0, 0
    o = jiwer.process_words(ref, hyp if hyp.strip() else '<empty>')
    return o.substitutions + o.deletions + o.insertions, len(ref.split())


def score(path, manifest):
    data = json.loads(Path(path).read_text())
    groups = defaultdict(lambda: dict(pnc=[0, 0], norm=[0, 0], n=0, exact=0, term=0, cased=0))
    for row in data['rows']:
        m = manifest[row['id']]
        ref, hyp = m['reference'], row['hyp'] or ''
        keys = ['ALL', m['part'], m['category']] + (['tts-noisy'] if m.get('snr_db') else [])
        e_n = edits(NORM(ref), NORM(hyp))
        e_p = edits(ref, hyp) if m['cased_reference'] else None
        for k in keys:
            g = groups[k]; g['n'] += 1
            g['norm'][0] += e_n[0]; g['norm'][1] += e_n[1]
            if e_p: g['pnc'][0] += e_p[0]; g['pnc'][1] += e_p[1]
            if m['cased_reference']:
                g['cased'] += 1
                g['exact'] += ref.strip() == hyp.strip()
                g['term'] += hyp.strip()[-1:] in ('.', '?', '!')
    res = {}
    for k, g in groups.items():
        res[k] = dict(n=g['n'], norm_wer=round(100 * g['norm'][0] / max(g['norm'][1], 1), 2),
                      pnc_wer=round(100 * g['pnc'][0] / g['pnc'][1], 2) if g['pnc'][1] else None,
                      exact=g['exact'], norm_edits=g['norm'][0], norm_words=g['norm'][1],
                      terminal_punct_pct=round(100 * g['term'] / g['cased'], 1) if g['cased'] else None)
    return data['meta'], res


def main():
    manifest = {json.loads(l)['id']: json.loads(l) for l in (OUT / 'manifest.jsonl').read_text().splitlines()}
    order = ['ALL', 'tts', 'librispeech', 'short', 'medium', 'numeric', 'long', 'tts-noisy']
    summary = {}
    print(f"{'label':34s} " + ' '.join(f'{k:>16s}' for k in order))
    for path in sys.argv[1:]:
        meta, res = score(path, manifest)
        summary[meta['label']] = res
        cells = []
        for k in order:
            r = res.get(k)
            cells.append(f"{r['norm_wer']:6.2f}/{r['pnc_wer'] if r['pnc_wer'] is not None else '-':>6}" if r else ' ' * 13)
        print(f"{meta['label']:34s} " + ' '.join(f'{c:>16s}' for c in cells))
    print('(cells: normalized WER / PnC WER, %)')
    print(f"\n{'label':34s} {'tts norm edits/words':>22s} {'libri norm edits/words':>24s} {'tts exact':>10s} {'tts ends .?!':>13s}")
    for label, res in summary.items():
        t, l = res['tts'], res.get('librispeech', {})
        print(f"{label:34s} {t['norm_edits']:>12d}/{t['norm_words']:<9d} {l.get('norm_edits', 0):>12d}/{l.get('norm_words', 0):<11d}"
              f" {t['exact']:>4d}/{t['n']:<5d} {t['terminal_punct_pct']:>12}%")
    (OUT / 'results' / 'wer-summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
