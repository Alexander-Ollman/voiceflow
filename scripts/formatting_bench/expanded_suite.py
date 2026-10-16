#!/usr/bin/env python3
"""Run190 then1000 with mandatory memory guard and durable worker checkpoints."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from run import ROOT
from expand import V2
from expanded_worker import BACKENDS,atomic
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
    p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args()
    reading=snapshot();reason=guard_reason(reading)
    if reason:
        atomic(V2/'suite-status.json',dict(state='held_for_priority_workload',reason=reason,reading=reading))
        raise SystemExit('Benchmark held before loading: '+reason)
    os.nice(19)
    watcher=subprocess.Popen([sys.executable,str(Path(__file__).with_name('watch_resources.py')),'--suite-pid',str(os.getpid())])
    try:
        time.sleep(.25)
        if watcher.poll() is not None:raise RuntimeError('Resource watcher failed to start')
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
