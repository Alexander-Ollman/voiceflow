#!/usr/bin/env python3
"""Durable sequential worker. Stage190 is a strict prefix of frozen1000."""
import argparse, hashlib, json, os, platform, statistics, threading, time
from pathlib import Path
import psutil
from run import ROOT, MODELS, adapter, score, contains
from expand import V2, digest
from shadow_verifier import features

BACKENDS=['identity','rules','pcs','pcs_guarded','gector','coedit','bonsai']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def checkpoint_rows(path):
    if not path.exists():return []
    data=path.read_bytes()
    if data and not data.endswith(b'\n'):
        data=data[:data.rfind(b'\n')+1];path.write_bytes(data)
    return [json.loads(s) for s in data.splitlines()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=BACKENDS,required=True);p.add_argument('--limit',choices=[190,1000],type=int,required=True)
    p.add_argument('--library',required=True);a=p.parse_args()
    corpus=V2/'corpus.jsonl';cases=[json.loads(s) for s in corpus.read_text().splitlines()]
    assert len(cases)==1000 and len({r['id'] for r in cases})==1000
    assert all(all(contains(c['input'],s) for s in c['protected']) for c in cases)
    output=V2/(a.backend+'.jsonl');meta_path=output.with_suffix('.meta.json');status=V2/(a.backend+'.status.json')
    signature=dict(corpus_sha256=sha(corpus),adapter_sha256=sha(Path(__file__).with_name('run.py')),
        worker_sha256=sha(__file__),models=MODELS,native_library_sha256=sha(a.library) if a.backend in ['rules','bonsai'] else None,
        repeats='3 per first190;1 per remaining810',encoder_threads=4)
    rows=checkpoint_rows(output)
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['signature']!=signature:raise RuntimeError('Resume signature changed; use a new run directory')
    else:
        if rows:raise RuntimeError('Missing run metadata')
        meta=dict(backend=a.backend,signature=signature,platform=platform.platform(),sessions=[])
    assert [r['id'] for r in rows]==[c['id'] for c in cases[:len(rows)]], 'Non-prefix resume data'
    if len(rows)>=a.limit:print('Already complete',a.backend,a.limit);return
    process=psutil.Process();peaks=[process.memory_info().rss];done=threading.Event()
    def memory():
        while not done.wait(.05):
            try:peaks.append(sum(x.memory_info().rss for x in [process]+process.children(recursive=True)))
            except psutil.Error:pass
    threading.Thread(target=memory,daemon=True).start()
    atomic(meta_path,meta);atomic(status,dict(backend=a.backend,state='loading',completed=len(rows),target=a.limit,pid=os.getpid()))
    start=time.perf_counter();infer,close=adapter(a.backend,a.library);load=time.perf_counter()-start
    start=time.perf_counter();infer(cases[0]);first=time.perf_counter()-start
    session=dict(started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage=a.limit,load_seconds=load,first_inference_seconds=first,starting_count=len(rows))
    try:
        with output.open('a') as out:
            for i in range(len(rows),a.limit):
                case=cases[i];durations=[];outputs=[];errors=[];warnings=[]
                for _ in range(3 if i<190 else 1):
                    start=time.perf_counter()
                    try:result=infer(case)
                    except Exception as e:result='';errors.append(type(e).__name__+': '+str(e))
                    durations.append((time.perf_counter()-start)*1000);outputs.append(result)
                    warnings.append(getattr(infer,'last_metadata',{}).copy())
                record=dict(id=case['id'],source=case['source'],category=case['category'],family=case['family'],
                    output=outputs[0],outputs=outputs,latency_ms=durations,errors=errors,adapter_warnings=warnings,
                    shadow=features(case['input'],outputs[0],case.get('context','')),protected_evaluated=len(case['protected']),
                    spoken_protected_evaluated=len(case.get('source_protected',[])),
                    spoken_protected_failed=sum(not contains(outputs[0],s) for s in case.get('source_protected',[])),**score(case,outputs[0]))
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
                atomic(status,dict(backend=a.backend,state='running',completed=i+1,target=a.limit,pid=os.getpid(),last_case=case['id']))
                if (i+1)%10==0:print(a.backend,f'{i+1}/{a.limit}',round(statistics.median(durations),1),flush=True)
    finally:
        session['peak_process_tree_rss_bytes']=max(peaks);done.set();close()
        meta['sessions'].append(session);atomic(meta_path,meta)
    atomic(status,dict(backend=a.backend,state='complete',completed=a.limit,target=a.limit,pid=os.getpid()))

if __name__=='__main__':main()
