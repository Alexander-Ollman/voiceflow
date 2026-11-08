#!/usr/bin/env python3
"""Print clips where two result files disagree. Usage: diff.py A.json B.json short,numeric[,...]"""
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'VoiceFlowApp/build/asr-eval'
m = {json.loads(l)['id']: json.loads(l) for l in (OUT / 'manifest.jsonl').read_text().splitlines()}
A = json.loads(Path(sys.argv[1]).read_text())['rows']
B = {r['id']: r for r in json.loads(Path(sys.argv[2]).read_text())['rows']}
cats = sys.argv[3].split(',')
for r in A:
    mm = m[r['id']]
    if mm['category'] in cats and r['hyp'] != B[r['id']]['hyp']:
        print(r['id']); print('  REF', mm['reference']); print('  A  ', r['hyp']); print('  B  ', B[r['id']]['hyp'])
