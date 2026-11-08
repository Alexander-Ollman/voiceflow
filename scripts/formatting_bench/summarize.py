#!/usr/bin/env python3
"""Summarize checked-in, per-case evidence without hiding errors."""
import argparse, json, math, statistics
from pathlib import Path

def summarize(path):
    data = json.loads(path.read_text()); rows=data['rows']
    latency = sorted(t for r in rows for t in r['latency_ms'])
    groups={}
    for category in sorted({r['category'] for r in rows}):
        selected=[r for r in rows if r['category']==category]
        groups[category]=f"{sum(r['exact'] for r in selected)}/{len(selected)}"
    return dict(backend=data['backend'], cases=len(rows), exact=sum(r['exact'] for r in rows), category_exact=groups,
        preserve_changed=sum(r['changed'] for r in rows if r['category']=='preserve'),
        protected_failed_cases=sum(bool(r['protected_failures']) for r in rows),
        protected_cases=sum(bool(r['protected']) for r in rows),
        errors=sum(bool(r['errors']) or r['empty'] for r in rows),
        unstable=sum(len(set(r['outputs']))>1 for r in rows),
        punctuation_warm_median_ms=round(statistics.median(t for r in rows if r['category']=='punctuation' for t in r['latency_ms']),2),
        warm_median_ms=round(statistics.median(latency),2), warm_p95_ms=round(latency[math.ceil(.95*len(latency))-1],2),
        load_seconds=round(data['load_seconds'],2),first_inference_seconds=round(data['first_inference_seconds'],2),
        peak_process_tree_rss_mib=round(data['peak_process_tree_rss_bytes']/2**20))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+',type=Path);a=p.parse_args()
    print(json.dumps([summarize(x) for x in a.paths],indent=2))
