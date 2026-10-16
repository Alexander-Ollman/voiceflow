#!/usr/bin/env python3
"""Collect four-voice, clean/20dB synthetic speech; resume without replacing rows."""
import argparse, base64, hashlib, io, json, os, socket, subprocess, time, wave
from pathlib import Path
import numpy as np
from run import ROOT, CACHE, contains
from expand import V2, write
from collect_asr import request

def perturb(path, snr, seed):
    with wave.open(str(path),'rb') as wav:
        assert wav.getnchannels()==1 and wav.getsampwidth()==2 and wav.getframerate()==16000
        audio=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float64)
    rng=np.random.default_rng(seed)
    clipping=0
    if snr is not None:
        active=audio[np.abs(audio)>200]
        rms=np.sqrt(np.mean(active**2)) if len(active) else 1
        audio+=rng.normal(0,rms/(10**(snr/20)),len(audio))
        clipping=int(np.sum(np.abs(audio)>32767))
    audio=np.pad(np.clip(audio,-32768,32767).astype('<i2'),(4000,4000))
    with wave.open(str(path),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(audio.tobytes())
    return clipping

def main():
    p=argparse.ArgumentParser();p.add_argument('--python',required=True);a=p.parse_args()
    plan=[json.loads(s) for s in (V2/'speech-plan.jsonl').read_text().splitlines()]
    target=V2/'speech.jsonl';existing=[json.loads(s) for s in target.read_text().splitlines()] if target.exists() else []
    done={r['id'] for r in existing}
    audio_dir=V2/'audio';audio_dir.mkdir(exist_ok=True)
    sock=Path(f'/tmp/vf-expanded-{os.getpid()}.sock')
    env=dict(os.environ,VOICEFLOW_PARAKEET_SOCKET=str(sock),VOICEFLOW_PARAKEET_PID=str(V2/'parakeet.pid'),VOICEFLOW_PARAKEET_LOG=str(V2/'parakeet.log'))
    with (V2/'parakeet-stderr.log').open('a') as log:
        child=subprocess.Popen([a.python,str(ROOT/'scripts/parakeet_asr_daemon.py')],env=env,stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+30
            while not sock.exists():
                if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('ASR daemon failed to start')
                time.sleep(.1)
            from huggingface_hub import snapshot_download
            model=snapshot_download('mlx-community/parakeet-tdt-0.6b-v2',revision='8ae155301e23d820d82aa60d24817c900e69e487',local_files_only=True)
            request(sock,{'command':'preload','model_id':model})
            for i,c in enumerate(plan):
                if c['id'] in done:continue
                aiff=audio_dir/(c['id']+'.aiff');wav=aiff.with_suffix('.wav')
                subprocess.run(['say','-v',c['voice'],'-o',str(aiff),'-f','-'],input=c['spoken_input'],text=True,check=True)
                subprocess.run(['afconvert','-f','WAVE','-d','LEI16@16000','-c','1',str(aiff),str(wav)],check=True)
                clipped=perturb(wav,c['noise_snr_db'],20261015+i)
                result=request(sock,{'command':'transcribe','audio':base64.b64encode(wav.read_bytes()).decode()})
                raw=result['text']
                r=dict(c,input=raw,source_protected=c['protected'],protected=[s for s in c['protected'] if contains(raw,s)],
                    audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),asr_elapsed_ms=result['elapsed_ms'],clipped_samples=clipped)
                # Avoid official written-source token fields on derived audio examples.
                for key in ['source_tokens','reference_tokens','gold_m2']:r.pop(key,None)
                with target.open('a') as out:out.write(json.dumps(r,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
                aiff.unlink();print(f"{i+1}/200 {c['id']} {c['voice']} snr={c['noise_snr_db']} chars={len(raw)}",flush=True)
        finally:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    rows=[json.loads(s) for s in (V2/'text.jsonl').read_text().splitlines()]+[json.loads(s) for s in target.read_text().splitlines()]
    by_id={r['id']:r for r in rows};selection=json.loads((V2/'selection.json').read_text())
    ordered=[by_id[id] for id in selection['tier_1000_ids']]
    assert len(ordered)==1000 and len(by_id)==1000
    write(V2/'corpus.jsonl',ordered)
    print('Frozen complete 1000-case corpus, with nested190 prefix.',flush=True)

if __name__=='__main__':main()
