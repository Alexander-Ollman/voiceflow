#!/usr/bin/env python3
"""Run190 then1000 with mandatory memory guard and durable worker checkpoints."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from run import ROOT
from expand import V2
from expanded_worker import BACKENDS,atomic,sha
from watch_resources import snapshot,guard_reason

def checked_child(command,watcher,log=None):
    child=subprocess.Popen(command,stdout=log,stderr=log)
    try:
        while child.poll() is None:
            if watcher.poll() is not None:raise RuntimeError('Resource watcher exited; refusing unguarded inference')
            time.sleep(.25)
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,command)
    except BaseException:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--latency-only',action='store_true');a=p.parse_args()
    reading=snapshot();memory_metric='available' if a.latency_only else 'free';reason=guard_reason(reading,memory_metric=memory_metric)
    if reason:
        atomic(V2/'suite-status.json',dict(state='held_for_priority_workload',reason=reason,reading=reading))
        raise SystemExit('Benchmark held before loading: '+reason)
    if not a.latency_only:os.nice(19)
    watcher=subprocess.Popen([sys.executable,str(Path(__file__).with_name('watch_resources.py')),'--suite-pid',str(os.getpid()),'--nice','0' if a.latency_only else '19','--memory-metric',memory_metric])
    try:
        time.sleep(.25)
        if watcher.poll() is not None:raise RuntimeError('Resource watcher failed to start')
        if a.latency_only:
            import random
            order=BACKENDS.copy();random.Random(20261015).shuffle(order)
            for name in order:
                existing=ROOT/'benchmarks/formatting/expanded-v2/latency-paused-peer'/(name+'.json')
                if existing.exists():
                    meta=json.loads(existing.read_text())
                    assert meta['measured_calls']==950 and meta['cpu_nice']==0, 'Invalid completed timing artifact'
                    assert meta['corpus_sha256']==sha(V2/'corpus.jsonl')
                    assert meta['adapter_sha256']==sha(Path(__file__).with_name('run.py'))
                    assert meta['worker_sha256']==sha(Path(__file__).with_name('expanded_latency.py'))
                    if name in ['rules','bonsai']:assert meta['native_library_sha256']==sha(a.library)
                    print('Completed timing retained',name,flush=True)
                    continue
                atomic(V2/'latency-status.json',dict(state='running',backend=name,pid=os.getpid()))
                with (V2/(name+'-latency-paused-peer.log')).open('a') as log:
                    checked_child([sys.executable,str(Path(__file__).with_name('expanded_latency.py')),'--backend',name,'--library',a.library],watcher,log)
            atomic(V2/'latency-status.json',dict(state='complete',backends=order,pid=os.getpid()))
            atomic(V2/'suite-status.json',dict(state='complete',quality_cases=1000,latency_cases=190,latency_rounds=5,pid=os.getpid()))
            return
        for limit in [190,1000]:
            for name in BACKENDS:
                atomic(V2/'suite-status.json',dict(stage=limit,backend=name,state='running',pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
                with (V2/(name+f'-stage{limit}.log')).open('a') as log:
                    checked_child([sys.executable,str(Path(__file__).with_name('expanded_worker.py')),'--backend',name,'--limit',str(limit),'--library',a.library],watcher,log)
            atomic(V2/'suite-status.json',dict(stage=limit,state='scoring',pid=os.getpid()))
            checked_child([sys.executable,str(Path(__file__).with_name('expanded_score.py')),'--limit',str(limit)],watcher)
            print('STAGE COMPLETE',limit,flush=True)
        atomic(V2/'suite-status.json',dict(stage=1000,state='complete',pid=os.getpid()))
    finally:
        if watcher.poll() is None:watcher.terminate();watcher.wait(timeout=5)

if __name__=='__main__':main()
