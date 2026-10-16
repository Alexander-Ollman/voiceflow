#!/usr/bin/env python3
"""Freeze a nested 190/1000-case research suite before observing model outputs."""
import hashlib, json, re, tarfile, urllib.request, subprocess
from pathlib import Path
from run import ROOT, CACHE, canonical

V2 = CACHE/'expanded-v2'
REV = 'ee06ff806a208aba815ac45313f4e750a48330a5'
ARCHIVE = 'https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz'

def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def write(path, rows): path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def row(id,category,text,refs,source,family,protected=None,context='',**extra):
    return dict(id=id,category=category,input=text,references=refs,source=source,family=family,
                protected=protected or [],context=context,tts=False,**extra)
def select(rows,n): return sorted(rows,key=lambda r:digest('voiceflow-v2-20261015:'+r['id']))[:n]
def balanced(rows,n):
    groups={key:select([r for r in rows if r['category']==key],len(rows)) for key in sorted({r['category'] for r in rows})}
    chosen=[]
    while len(chosen)<n:
        for group in groups.values():
            if group and len(chosen)<n: chosen.append(group.pop(0))
    return chosen

def challenges():
    rows=[]
    names=['Aurora','Beacon','Cedar','Delta','Ember','Falcon','Garnet','Harbor','Indigo','Juniper']
    plural=['settings','files','reports','meetings','projects','tickets','records','folders','messages','requests']
    singular=['setting','file','report','meeting','project','ticket','record','folder','message','request']
    for i,name in enumerate(names):
        url=f'https://example.org/{name.lower()}?version={i+1}'
        amount=f'${i+1},250.50'
        cases=[
          ('preserve','clock',f'The {name} meeting starts at {i+1}:30 PM.',[f'{i+1}:30 PM']),
          ('preserve','amount',f'The {name} invoice total is {amount}.',[amount]),
          ('preserve','url',f'Open {url}.',[url]),
          ('preserve','email',f'Email {name.lower()}@example.org.',[f'{name.lower()}@example.org']),
          ('preserve','version',f'Deploy {name} version v2.{i+1}.7.',[f'v2.{i+1}.7']),
          ('preserve','identifier',f'Keep userID{i+1} separate from userid{i+1}.',[f'userID{i+1}',f'userid{i+1}']),
          ('preserve','negation',f'Do not delete the {name} backup.',['not']),
          ('preserve','code',f'Run git status --short in the {name} folder.',['git status --short']),
          ('preserve','brand',f'{name} uses VoiceFlow on an iPhone.',['VoiceFlow','iPhone']),
          ('preserve','unicode',f'José and Siobhán reviewed the {name} document.',['José','Siobhán']),
        ]
        for category,family,text,protected in cases:
            rows.append(row(f'challenge-{family}-{i:02}',category,text,[text],'controlled',family,protected))
        punct=[
          ('question',f'can you send the {name.lower()} report',f'Can you send the {name} report?'),
          ('statement',f'i will review the {name.lower()} report tomorrow',f'I will review the {name} report tomorrow.'),
          ('comma',f'yes the {name.lower()} report is ready',f'Yes, the {name} report is ready.'),
          ('thanks',f'thanks for reviewing the {name.lower()} report',f'Thanks for reviewing the {name} report.'),
          ('join',f'the {name.lower()} test passed now we can deploy',f'The {name} test passed. Now we can deploy.'),
          ('when',f'when will the {name.lower()} meeting start',f'When will the {name} meeting start?'),
          ('proper',f'i work on the {name.lower()} project in seattle',f'I work on the {name} project in Seattle.'),
        ]
        for family,text,ref in punct:
            refs=[ref]
            if family=='join':refs.append(f'The {name} test passed; now we can deploy.')
            # Name capitalization is explicit task context, not inferred from a hidden reference.
            context=f'Formatting context: {name} is the exact project name. Preserve wording and add punctuation.'
            rows.append(row(f'challenge-punct-{family}-{i:02}','punctuation',text,refs,'controlled','punct-'+family,context=context))
        grammars=[
          ('agreement',f'These is the {name} {plural[i]}.',f'These are the {name} {plural[i]}.'),
          ('habit',f'She review the {name} {singular[i]} every day.',f'She reviews the {name} {singular[i]} every day.'),
          ('have',f'I has a question about the {name} {singular[i]}.',f'I have a question about the {name} {singular[i]}.'),
          ('past',f'We was ready for the {name} meeting.',f'We were ready for the {name} meeting.'),
          ('negative',f'He did not reviewed the {name} {singular[i]}.',f'He did not review the {name} {singular[i]}.'),
          ('infinitive',f'I want to reviewing the {name} {singular[i]}.',f'I want to review the {name} {singular[i]}.'),
          ('gerund',f'I look forward to discuss the {name} {singular[i]}.',f'I look forward to discussing the {name} {singular[i]}.'),
        ]
        for family,text,ref in grammars:
            rows.append(row(f'challenge-grammar-{family}-{i:02}','grammar',text,[ref],'controlled','grammar-'+family))
        norms=[
          ('fillers',f'um I uh need the {name} report.',f'I need the {name} report.'),
          ('repeat',f'The the {name} microphone is working.',f'The {name} microphone is working.'),
          ('paragraph',f'Review the {name} report new paragraph Thank you.',f'Review the {name} report\n\nThank you.'),
          ('money',f'The {name} budget is {singular[i]}',None),
        ]
        amounts=['one','two','three','four','five','six','seven','eight','nine','ten']
        norms[-1]=('money',f'The {name} budget is {amounts[i]} thousand dollars.',f'The {name} budget is ${(i+1)*1000:,}.')
        for family,text,ref in norms:
            rows.append(row(f'challenge-normalize-{family}-{i:02}','normalization',text,[ref],'controlled','normalization-'+family))
        terms=[('kubectl','cube control'),('Postgres','post grass'),('GitHub','get hub'),('PyTorch','pie torch')]
        for term,mishear in terms:
            text=f'Use {mishear} for the {name} project.'
            context=f'Current document vocabulary: {term}. {name} is the project name. Correct technical terms only when supported by context.'
            rows.append(row(f'challenge-context-{term}-{i:02}','semantic',text,[f'Use {term} for the {name} project.'],'controlled','context-'+term,context=context))
        # Four structure families, distinct contents; retain only 37 below.
        items=[f'{name} report',f'{name} notes',f'{name} checklist']
        for family in ['keep-bullets','keep-numbering','make-bullets','single-line']:
            if family=='keep-bullets': text='\n'.join('- '+x for x in items);refs=[text];ctx=''
            elif family=='keep-numbering':text='\n'.join(f'{j+1}. {x}' for j,x in enumerate(items));refs=[text];ctx=''
            elif family=='make-bullets':
                text=', '.join(items)+'.';refs=['\n'.join(mark+' '+x for x in items) for mark in ['-','*','•']]
                ctx='Output formatting preference: return one bullet per named item. Preserve every item and its order. The target supports multiline plain text.'
            else:
                text=', '.join(items)+'.';refs=[text]
                ctx='Output formatting preference: keep this in one line with commas. The target is a single-line field; preserve every named item and its order.'
            rows.append(row(f'challenge-list-{family}-{i:02}','lists',text,refs,'controlled','list-'+family,context=ctx,list_items=items))
    lists=[r for r in rows if r['category']=='lists']
    rows=[r for r in rows if r['category']!='lists']+balanced(lists,37)
    assert len(rows)==357,len(rows)
    return rows

def main():
    V2.mkdir(parents=True,exist_ok=True)
    corpus=V2/'corpus.jsonl'
    if corpus.exists() or (V2/'selection.json').exists():raise SystemExit('Frozen corpus exists. Use its manifest; do not silently regenerate.')
    jf=CACHE/'jfleg-source'
    if not jf.exists():subprocess.run(['git','clone','https://github.com/keisks/jfleg.git',str(jf)],check=True)
    subprocess.run(['git','-C',str(jf),'checkout',REV],check=True,capture_output=True)
    archive=CACHE/'wi-locness.tar.gz'
    if not archive.exists():
        with urllib.request.urlopen(ARCHIVE) as res:archive.write_bytes(res.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != 'd5cbf68cda3da0c3af69dd672614d07287bfe996b87da0c75051d5349d76c666':
        raise RuntimeError('W&I archive changed; review before creating a new suite')
    from sacremoses import MosesDetokenizer
    detok=MosesDetokenizer(lang='en')
    natural=[]
    for split in ['dev','test']:
        texts=(jf/split/(split+'.src')).read_text().splitlines()
        refs=[(jf/split/(split+f'.ref{k}')).read_text().splitlines() for k in range(4)]
        candidates=[]
        for i,text in enumerate(texts):
            if not 5<=len(text.split())<=60:continue
            candidates.append(row(f'jfleg-{split}-{i:04}','grammar',detok.detokenize(text.split()),
                [detok.detokenize(r[i].split()) for r in refs],'jfleg',f'jfleg-{split}-{i:04}',dataset_split=split,source_index=i,
                source_tokens=text,reference_tokens=[r[i] for r in refs]))
        natural+=select(candidates,100)
    bea=[]
    norm=lambda s: re.sub(r'\W','',s).casefold()
    with tarfile.open(archive) as tar:
        # W&I only. LOCNESS has separate redistribution constraints and is not sampled.
        for level,n in [('A',70),('B',65),('C',65)]:
            docs=[json.loads(s) for s in tar.extractfile(f'wi+locness/json/{level}.dev.json').read().decode().splitlines()]
            docs=[(d['id'],norm(d['text'])) for d in docs]
            blocks=tar.extractfile(f'wi+locness/m2/{level}.dev.gold.bea19.m2').read().decode().strip().split('\n\n')
            candidates=[]
            for i,block in enumerate(blocks):
                lines=block.splitlines();tokens=lines[0][2:].split()
                if not 5<=len(tokens)<=60:continue
                edited=tokens.copy();annotations=[]
                for line in lines[1:]:
                    parts=line[2:].split('|||');start,end=map(int,parts[0].split())
                    if start<0 or parts[1] in ['noop','UNK','Um']:continue
                    annotations.append((start,end,parts[2].split()))
                for start,end,replacement in sorted(annotations,reverse=True):edited[start:end]=replacement
                source=' '.join(tokens);needle=norm(source)
                matches=[id for id,body in docs if needle in body]
                if len(matches)!=1:continue
                candidates.append(row(f'wi-dev-{level}-{i:04}','grammar',detok.detokenize(tokens),[detok.detokenize(edited)],
                    'wi-dev',f'wi-doc-{matches[0]}',dataset_split='dev',cefr=level,source_index=i,source_tokens=source,
                    reference_tokens=[' '.join(edited)],gold_m2=block))
            bea+=select(candidates,n)
    seed=[dict(json.loads(s),family='seed-'+json.loads(s)['id']) for s in (ROOT/'benchmarks/formatting/fixtures.jsonl').read_text().splitlines()]
    stress=challenges()
    text190=seed+select(natural,40)+select(bea,40)+balanced(stress,37)
    text1000=text190+[r for r in seed+natural+bea+stress if r['id'] not in {x['id'] for x in text190}]
    assert len(text1000)==800
    # Base speech sources are distinct IDs; related speech/text cases retain the same family.
    speech=seed+balanced(stress,117)+select(natural,20)+select(bea,20)
    assert len(speech)==200 and len({r['id'] for r in speech})==200
    plan=[]
    for i,c in enumerate(speech):
        plan.append(dict(c,id='speech-'+c['id'],parent_id=c['id'],source='synthetic_tts_parakeet',spoken_input=c['input'],
            voice=['Samantha','Daniel','Karen','Rishi'][i%4],noise_snr_db=20 if (i//4)%4==3 else None))
    # Both tiers frozen before inference: 160 text +30 speech; final 800 text +200 speech.
    prefix_ids=[r['id'] for r in text190]+[r['id'] for r in balanced(plan,30)]
    all_ids=prefix_ids+[r['id'] for r in text1000+plan if r['id'] not in set(prefix_ids)]
    manifest=dict(version=2,seed='voiceflow-v2-20261015',tier_190_ids=prefix_ids,tier_1000_ids=all_ids,
        sources={'jfleg':dict(url='https://github.com/keisks/jfleg',revision=REV,selection='100 dev +100 test, stable hash, 5–60 tokens'),
                 'wi':dict(url=ARCHIVE,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),selection='W&I dev only:70 A,65 B,65 C; LOCNESS excluded',terms='noncommercial research; no dataset text or predictions committed')},
        notes=['No model output used to select examples or tune gates.','Text/speech pairs and template variants are related; family-aware intervals required.','External model training overlap is unknown; local frozen evaluation is not proof of uncontaminated model pretraining.'])
    write(V2/'text.jsonl',text1000);write(V2/'speech-plan.jsonl',plan)
    (V2/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # No human dataset text in the tracked selection/provenance file.
    out=ROOT/'benchmarks/formatting/expanded-v2';out.mkdir(exist_ok=True)
    (out/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen 800 text cases and 200 speech sources; first tier160+30; final1000.')

if __name__=='__main__':main()
