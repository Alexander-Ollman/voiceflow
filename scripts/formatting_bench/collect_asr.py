#!/usr/bin/env python3
"""Generate public synthetic ASR fixtures, with a separately owned Parakeet daemon."""
import argparse, base64, hashlib, json, os, socket, struct, subprocess, time
from pathlib import Path
from run import ROOT, CACHE, contains

def request(path, data):
    def read(conn, n):
        result = b''
        while len(result) < n:
            chunk = conn.recv(n - len(result))
            if not chunk: raise EOFError('Daemon disconnected')
            result += chunk
        return result
    with socket.socket(socket.AF_UNIX) as conn:
        conn.settimeout(300); conn.connect(str(path))
        payload = json.dumps(data).encode(); conn.sendall(struct.pack('>I', len(payload)) + payload)
        result = json.loads(read(conn, struct.unpack('>I', read(conn, 4))[0]))
        if result.get('status') != 'ok': raise RuntimeError(result)
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', required=True, help='Python with parakeet-mlx installed')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    # Unique socket/PID files; never touch the application's service.
    sock = Path(f'/tmp/vf-bench-{os.getpid()}.sock')
    env = dict(os.environ, VOICEFLOW_PARAKEET_SOCKET=str(sock), VOICEFLOW_PARAKEET_PID=str(CACHE/'parakeet.pid'),
               VOICEFLOW_PARAKEET_LOG=str(CACHE/'parakeet.log'))
    log = (CACHE/'parakeet-stderr.log').open('w')
    process = subprocess.Popen([args.python, str(ROOT/'scripts/parakeet_asr_daemon.py')], env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 30
        while not sock.exists():
            if process.poll() is not None or time.monotonic() > deadline: raise RuntimeError('Daemon startup failed')
            time.sleep(.1)
        from huggingface_hub import snapshot_download
        model_path = snapshot_download('mlx-community/parakeet-tdt-0.6b-v2', revision='8ae155301e23d820d82aa60d24817c900e69e487', local_files_only=True)
        request(sock, {'command':'preload', 'model_id':model_path})
        rows = []
        fixtures = [json.loads(s) for s in (ROOT/'benchmarks/formatting/fixtures.jsonl').read_text().splitlines()]
        for case in fixtures:
            if not case['tts']: continue
            audio = CACHE/(case['id']+'.aiff'); wav = audio.with_suffix('.wav')
            subprocess.run(['say','-v','Samantha','-o',str(audio),case['input']], check=True)
            subprocess.run(['afconvert','-f','WAVE','-d','LEI16@16000','-c','1',str(audio),str(wav)], check=True)
            result = request(sock, {'command':'transcribe','audio':base64.b64encode(wav.read_bytes()).decode()})
            raw = result['text']
            rows.append(dict(case, id='asr-'+case['id'], source='synthetic_tts_parakeet', spoken_input=case['input'], input=raw,
                protected=[p for p in case['protected'] if contains(raw,p)], audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                asr_elapsed_ms=result['elapsed_ms']))
            print(case['id'], repr(raw), flush=True)
        target = ROOT/'benchmarks/formatting/asr-fixtures.jsonl'
        target.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()

if __name__ == '__main__': main()
