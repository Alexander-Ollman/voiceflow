#!/usr/bin/env python3
"""Publish aggregate + redacted evidence; human corpus text/predictions stay local."""
import argparse, hashlib, importlib.util, json, math, re, statistics
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from run import ROOT, canonical
from expand import V2,REV
from expanded_worker import BACKENDS,atomic

OUT=ROOT/'benchmarks/formatting/expanded-v2'

def interval(rows,field='exact',iterations=2000):
    groups=defaultdict(lambda:[0,0])
    for r in rows:
        groups[r['family']][0]+=bool(r[field]);groups[r['family']][1]+=1
    a=np.array(list(groups.values()),dtype=float);rng=np.random.default_rng(20261015)
    weights=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=iterations)
    values=(weights@a[:,0])/(weights@a[:,1])
    return [round(float(x),4) for x in np.percentile(values,[2.5,97.5])]

def group_summary(rows):
    ts=sorted(t for r in rows for t in r['latency_ms'])
    return dict(cases=len(rows),family_clusters=len({r['family'] for r in rows}),exact=sum(r['exact'] for r in rows),
      exact_rate=sum(r['exact'] for r in rows)/len(rows),exact_family_bootstrap_95=interval(rows),
      changed=sum(r['changed'] for r in rows),errors=sum(bool(r['errors']) or r['empty'] for r in rows),
      protected_failed_cases=sum(bool(r['protected_failures']) for r in rows),
      protected_cases=sum(r['protected_evaluated']>0 for r in rows),
      acceptable_input_cases=sum(r['input_acceptable'] for r in rows),
      acceptable_input_off_reference=sum(r['input_acceptable'] and not r['exact'] for r in rows),
      spoken_protected_spans=sum(r['spoken_protected_evaluated'] for r in rows),
      spoken_protected_failed=sum(r['spoken_protected_failed'] for r in rows),
      unstable_cases=sum(len(set(r['outputs']))>1 for r in rows),
      warm_median_ms=round(statistics.median(ts),2),warm_p95_ms=round(ts[math.ceil(.95*len(ts))-1],2),
      case_median_latency_median_ms=round(statistics.median(statistics.median(r['latency_ms']) for r in rows),2),
      case_median_latency_p95_ms=round(sorted(statistics.median(r['latency_ms']) for r in rows)[math.ceil(.95*len(rows))-1],2),
      limit_warning_cases=sum(any(any(d.values()) for d in r['adapter_warnings']) for r in rows))

def paired_difference(left,right):
    assert [r['id'] for r in left]==[r['id'] for r in right]
    groups=defaultdict(lambda:[0,0])
    for a,b in zip(left,right):groups[a['family']][0]+=int(a['exact'])-int(b['exact']);groups[a['family']][1]+=1
    values=np.array(list(groups.values()));rng=np.random.default_rng(20261015)
    weights=rng.multinomial(len(values),np.full(len(values),1/len(values)),size=2000)
    draws=(weights@values[:,0])/(weights@values[:,1])
    return dict(exact_difference=float(values[:,0].sum()/values[:,1].sum()),family_bootstrap_95=[round(float(x),4) for x in np.percentile(draws,[2.5,97.5])])

def errant_wi(cases,results,ann):
    from errant.commands.compare_m2 import simplify_edits,process_edits,evaluate_edits,computeFScore
    args=SimpleNamespace(dt=False,ds=False,cse=False,single=False,multi=False,filt=[],beta=.5,verbose=False)
    selected=[c for c in cases if c['source']=='wi-dev'];source_docs={c['id']:ann.parse(c['source_tokens']) for c in selected}
    gold={c['id']:process_edits(simplify_edits(c['gold_m2']),args) for c in selected}
    summary={};redacted={}
    for name,rows in results.items():
        byid={r['id']:r for r in rows};total=dict(tp=0,fp=0,fn=0);percase=[]
        for c in selected:
            hyp=ann.parse(byid[c['id']]['output'],tokenise=True)
            edits=ann.annotate(source_docs[c['id']],hyp)
            hyp_dict=process_edits([[e.o_start,e.o_end,e.type,e.c_str,0] for e in edits],args)
            counts,_=evaluate_edits(hyp_dict,gold[c['id']],total,c['id'],c['source_tokens'],args)
            for k in total:total[k]+=counts[k]
            percase.append(dict(id=c['id'],**counts))
        p,r,f=computeFScore(total['tp'],total['fp'],total['fn'],.5)
        summary[name]=dict(cases=len(selected),precision=p,recall=r,f0_5=f,**total)
        redacted[name]=percase
    if 'identity' in summary and summary['identity']['fp'] != 0:
        raise RuntimeError('Identity control has false edits; audit tokenizer/gold alignment before reporting')
    return summary,redacted

def wi_edit_intervals(cases, counts):
    families={c['id']:c['family'] for c in cases if c['source']=='wi-dev'}
    keys=sorted(set(families.values()));position={k:i for i,k in enumerate(keys)}
    rng=np.random.default_rng(20261015)
    weights=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=2000)
    values={}
    for name,rows in counts.items():
        clusters=np.zeros((len(keys),3))
        for row in rows:clusters[position[families[row['id']]]]+=np.array([row['tp'],row['fp'],row['fn']])
        boot=weights@clusters
        denominator=1.25*boot[:,0]+boot[:,1]+.25*boot[:,2]
        values[name]=np.divide(1.25*boot[:,0],denominator,out=np.ones_like(denominator),where=denominator!=0)
    return {name:dict(document_clusters=len(keys),f0_5_bootstrap_95=[round(float(x),4) for x in np.percentile(draws,[2.5,97.5])],
                 vs_bonsai_f0_5_bootstrap_95=[round(float(x),4) for x in np.percentile(draws-values['bonsai'],[2.5,97.5])]) for name,draws in values.items()}

def jfleg_gleu(cases,results,ann,limit):
    jf=V2.parent/'jfleg-source'
    spec=importlib.util.spec_from_file_location('pinned_jfleg_gleu',jf/'eval/gleu.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    selected=[c for c in cases if c['source']=='jfleg'];work=V2/f'gleu-{limit}';work.mkdir(exist_ok=True)
    source=work/'source';source.write_text(''.join(c['source_tokens']+'\n' for c in selected))
    refs=[]
    for k in range(4):
        path=work/f'ref{k}';path.write_text(''.join(c['reference_tokens'][k]+'\n' for c in selected));refs.append(str(path))
    summary={}
    for name,rows in results.items():
        byid={r['id']:r for r in rows};hyp=work/name
        hyp.write_text(''.join(' '.join(t.text for t in ann.nlp.tokenizer(byid[c['id']]['output']))+'\n' for c in selected))
        gleu=module.GLEU(4);gleu.load_sources(str(source));gleu.load_references(refs)
        scores=list(gleu.run_iterations(num_iterations=500,source=str(source),hypothesis=str(hyp),per_sent=False))[0]
        summary[name]=dict(cases=len(selected),mean=float(scores[0]),reference_sampling_std=float(scores[1]),reference_sampling_interval=scores[2])
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,choices=[190,1000],required=True);a=p.parse_args()
    cases=[json.loads(s) for s in (V2/'corpus.jsonl').read_text().splitlines()][:a.limit]
    results={name:[json.loads(s) for s in (V2/(name+'.jsonl')).read_text().splitlines()][:a.limit] for name in BACKENDS}
    for name,rows in results.items():assert len(rows)==a.limit and [r['id'] for r in rows]==[c['id'] for c in cases],name
    for rows in results.values():
        for c,r in zip(cases,rows):r['input_acceptable']=canonical(c['input']) in [canonical(ref) for ref in c['references']]
    summary=dict(stage=a.limit,corpus_sha256=hashlib.sha256((V2/'corpus.jsonl').read_bytes()).hexdigest(),sources=dict(Counter(c['source'] for c in cases)),
        categories=dict(Counter(c['category'] for c in cases)),distinct_text_context_inputs=len({(c['input'],c.get('context','')) for c in cases}),
        family_clusters=len({c['family'] for c in cases}),backends={})
    for name,rows in results.items():
        summary['backends'][name]=dict(overall=group_summary(rows),by_source={s:group_summary([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
            by_category={s:group_summary([r for r in rows if r['category']==s]) for s in sorted({r['category'] for r in rows})},
            vs_bonsai=paired_difference(rows,results['bonsai']))
        unique=[];seen=set()
        for c,r in zip(cases,rows):
            key=(c['input'],c.get('context',''))
            if key not in seen:unique.append(r);seen.add(key)
        summary['backends'][name]['deduplicated_text_context']=group_summary(unique)
        speech_byid={c['id']:c for c in cases if c['source']=='synthetic_tts_parakeet'}
        summary['backends'][name]['speech_by_voice']={v:group_summary([r for r in rows if r['id'] in speech_byid and speech_byid[r['id']]['voice']==v]) for v in sorted({c['voice'] for c in speech_byid.values()})}
        summary['backends'][name]['speech_by_noise']={v:group_summary([r for r in rows if r['id'] in speech_byid and ('clean' if speech_byid[r['id']]['noise_snr_db'] is None else 'snr20')==v]) for v in sorted({'clean' if c['noise_snr_db'] is None else 'snr20' for c in speech_byid.values()})}
    import errant
    ann=errant.load('en')
    summary['wi_errant'],counts=errant_wi(cases,results,ann)
    summary['wi_edit_uncertainty']=wi_edit_intervals(cases,counts)
    summary['jfleg_gleu']=jfleg_gleu(cases,results,ann,a.limit)
    for name,rows in results.items():
        # Never commit restricted human inputs, reference spans, model text, errors containing text, or context.
        safe=[{k:r[k] for k in ['id','source','category','family','exact','changed','empty','latency_ms','adapter_warnings','input_acceptable','protected_evaluated','spoken_protected_evaluated','spoken_protected_failed']}|
              dict(error_count=len(r['errors']),protected_failure_count=len(r['protected_failures']),
                   unstable=len(set(r['outputs']))>1) for r in rows]
        (OUT/f'{name}-{a.limit}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in safe))
    atomic(OUT/f'summary-{a.limit}.json',summary)
    atomic(OUT/f'wi-errant-counts-{a.limit}.json',counts)
    print('Scored stage',a.limit,flush=True)
    for name in BACKENDS:print(name,summary['backends'][name]['overall']['exact'],summary['wi_errant'][name]['f0_5'],summary['jfleg_gleu'][name]['mean'],flush=True)

if __name__=='__main__':main()
