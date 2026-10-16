#!/usr/bin/env python3
"""Fresh, normal-priority timing pass after the competing experiment was paused."""
import argparse,json,math,os,random,re,statistics,threading,time
from pathlib import Path
import psutil
from run import ROOT,adapter,score
from expand import V2
from expanded_worker import atomic,sha

DEST=V2/'latency-paused-peer'
PUBLIC=ROOT/'benchmarks/formatting/expanded-v2/latency-paused-peer'
def percentile(values,p):return sorted(values)[math.ceil(p*len(values))-1]
def model_invoked(name,text):
    if name in ['identity','rules']:return False
    if name!='pcs_guarded':return True
    # Annotation only: the adapter makes the actual bypass decision.
    return not (re.search(r'https?://|@|[`_/]|\d|\b\w*[a-z][A-Z]\w*\b',text) or (text[:1].isupper() and text.rstrip().endswith(('.', '?', '!'))))
def stats(rows):
    times=[r['latency_ms'] for r in rows]
    return dict(calls=len(rows),median_ms=statistics.median(times),p95_ms=percentile(times,.95),min_ms=min(times),max_ms=max(times),errors=sum(r['error'] for r in rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True);p.add_argument('--library',required=True);a=p.parse_args()
    DEST.mkdir(exist_ok=True);PUBLIC.mkdir(exist_ok=True)
    target=PUBLIC/(a.backend+'.jsonl');meta_path=PUBLIC/(a.backend+'.json')
    if meta_path.exists():raise SystemExit('Timing run already exists; do not silently overwrite it')
    cases=[json.loads(s) for s in (V2/'corpus.jsonl').read_text().splitlines()][:190]
    process=psutil.Process();done=threading.Event();peaks=[process.memory_info().rss]
    def memory():
        while not done.wait(.05):
            try:peaks.append(sum(p.memory_info().rss for p in [process]+process.children(recursive=True)))
            except psutil.Error:pass
    threading.Thread(target=memory,daemon=True).start()
    start=time.perf_counter();infer,close=adapter(a.backend,a.library);load=time.perf_counter()-start
    rows=[];raw_path=DEST/(a.backend+'.jsonl')
    try:
        start=time.perf_counter();infer(cases[0]);first=(time.perf_counter()-start)*1000
        for index in [63,126]:infer(cases[index])
        with raw_path.open('w') as raw,target.open('w') as public:
            for round_index in range(5):
                order=list(range(190));random.Random(20261015+round_index).shuffle(order)
                for index in order:
                    case=cases[index];error=False;output=''
                    start=time.perf_counter_ns();started=time.time()
                    try:output=infer(case)
                    except Exception:error=True
                    duration=(time.perf_counter_ns()-start)/1e6
                    words=len(case['input'].split())
                    row=dict(id=case['id'],round=round_index,latency_ms=duration,started_unix=started,
                        input_characters=len(case['input']),input_words=words,output_characters=len(output),error=error,
                        invokes_model=model_invoked(a.backend,case['input']),exact=score(case,output)['exact'])
                    raw.write(json.dumps(dict(row,output=output),ensure_ascii=False)+'\n');raw.flush()
                    public.write(json.dumps(row)+'\n');public.flush();os.fsync(public.fileno());rows.append(row)
                print(a.backend,'round',round_index+1,stats(rows[-190:]),flush=True)
    finally:
        close();done.set()
    summary=dict(backend=a.backend,cases=190,repeats=5,measured_calls=len(rows),cpu_nice=process.nice(),
        peer_experiment='user reports paused',started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        load_seconds=load,first_call_ms=first,warmup_calls=3,peak_process_tree_rss_bytes=max(peaks),
        corpus_sha256=sha(V2/'corpus.jsonl'),adapter_sha256=sha(Path(__file__).with_name('run.py')),worker_sha256=sha(__file__),
        native_library_sha256=sha(a.library) if a.backend in ['rules','bonsai'] else None,overall=stats(rows),
        by_round={str(i):stats([r for r in rows if r['round']==i]) for i in range(5)},
        by_input_words={key:stats([r for r in rows if predicate(r['input_words'])]) for key,predicate in
                        [('1-10',lambda n:n<=10),('11-25',lambda n:10<n<=25),('26+',lambda n:n>25)]})
    active=[r for r in rows if r['invokes_model']]
    summary['model_invocation_calls']=len(active)
    if active:summary['model_invocation_latency']=stats(active)
    atomic(meta_path,summary)

if __name__=='__main__':main()
