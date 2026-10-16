#!/usr/bin/env python3
"""Protect a priority workload by stopping only our checkpointed benchmark tree."""
import argparse,json,os,re,signal,subprocess,time
from pathlib import Path
import psutil
from expand import V2
from expanded_worker import atomic

def snapshot():
    vm=subprocess.check_output(['vm_stat'],text=True)
    page_size=int(re.search(r'page size of (\d+)',vm).group(1))
    values={line.split(':')[0]:int(line.split(':')[1].strip().rstrip('.')) for line in vm.splitlines()[1:] if ':' in line}
    pressure=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
    return dict(time=time.time(),pressure=pressure,free_bytes=values['Pages free']*page_size,
                swapout_bytes=values['Swapouts']*page_size)

def guard_reason(reading, reserve_gib=16, swapout_delta=0):
    if reading['pressure'] != 1:return 'Non-normal memory pressure'
    if reading['free_bytes'] < reserve_gib*2**30:return 'Physical free memory below reserve'
    if swapout_delta > 0:return 'New swap-outs observed'
    return None

def hold(suite,reason,reading):
    atomic(V2/'resource-hold.json',dict(reason=reason,reading=reading,checkpoints_preserved=True))
    atomic(V2/'suite-status.json',dict(state='held_for_priority_workload',reason=reason,reading=reading))
    children=[]
    try:
        suite.send_signal(signal.SIGSTOP)
        children=[p for p in suite.children(recursive=True) if p.pid!=os.getpid()]
        workers=[]
        for child in children:
            try:
                if any(name in ' '.join(child.cmdline()) for name in ['expanded_worker.py','expanded_score.py']):workers.append(child)
            except psutil.Error:pass
        for child in workers:
            try:child.send_signal(signal.SIGINT);child.send_signal(signal.SIGCONT)
            except psutil.Error:pass
        _,alive=psutil.wait_procs(workers,timeout=10)
        for child in alive:
            try:child.terminate()
            except psutil.Error:pass
        _,alive=psutil.wait_procs(alive,timeout=3)
        for child in alive:
            try:child.kill()
            except psutil.Error:pass
    finally:
        for child in children:
            try:
                if child.is_running() and child.status()!=psutil.STATUS_ZOMBIE:child.terminate()
            except psutil.Error:pass
        try:suite.terminate();suite.send_signal(signal.SIGCONT)
        except psutil.Error:pass
    print('Benchmark held:',reason,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite-pid',type=int,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reserve-gib',type=float,default=16);a=parser.parse_args()
    suite=psutil.Process(a.suite_pid)
    if 'expanded_suite.py' not in ' '.join(suite.cmdline()):raise RuntimeError('Refusing to control an unrelated process')
    first=snapshot();previous=first
    if guard_reason(first,a.reserve_gib):
        hold(suite,guard_reason(first,a.reserve_gib),first);return
    for p in [suite]+suite.children(recursive=True):
        try:
            p.nice(19)
            if a.resume:p.send_signal(signal.SIGCONT)
        except psutil.Error:pass
    with (V2/'resource-watch.jsonl').open('a') as log:
        while suite.is_running() and suite.status()!=psutil.STATUS_ZOMBIE:
            try:reading=snapshot()
            except Exception as exc:
                hold(suite,'Resource sampling failed: '+type(exc).__name__,previous);raise
            reading['swapout_delta_bytes']=reading['swapout_bytes']-previous['swapout_bytes'];previous=reading
            log.write(json.dumps(reading)+'\n');log.flush()
            reason=guard_reason(reading,a.reserve_gib,reading['swapout_delta_bytes'])
            if reason:
                hold(suite,reason,reading);return
            for p in suite.children(recursive=True):
                try:p.nice(19)
                except psutil.Error:pass
            time.sleep(1)
    print('Resource watcher finished; benchmark tree exited.',flush=True)

if __name__=='__main__':main()
