#!/usr/bin/env python3
"""Build the frozen ASR evaluation corpus for the Nemotron vs Parakeet v2 comparison.

Two parts:
  1. Synthetic dictation: authored sentences with exact written references,
     spoken by macOS `say` voices (6 English accents), 16 kHz mono int16,
     0.25 s silence pad each side. One voice per sentence also gets a 20 dB SNR
     white-noise copy. Deterministic voice rotation and noise seeds.
  2. Real read speech: hf-internal-testing/librispeech_asr_dummy (73 clips of
     LibriSpeech validation-clean). References are upper-case, unpunctuated,
     so only normalized WER is meaningful for this part.

Writes to VoiceFlowApp/build/asr-eval/audio/ and manifest.jsonl (gitignored).

Usage: python corpus.py   (needs numpy; pyarrow + soundfile for the LibriSpeech part)
"""
import hashlib, io, json, subprocess, sys, wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VoiceFlowApp/build/asr-eval'
AUDIO = OUT / 'audio'
SR = 16000

VOICES = ['Samantha', 'Daniel', 'Karen', 'Moira', 'Rishi', 'Tessa']

# (id, category, written reference). Short = expected < 2 s of speech.
SENTENCES = [
    ('short-01', 'short', 'Yes.'),
    ('short-02', 'short', 'Sounds good.'),
    ('short-03', 'short', 'Thanks so much!'),
    ('short-04', 'short', 'Call me back.'),
    ('short-05', 'short', 'Not today.'),
    ('short-06', 'short', 'Delete that.'),
    ('short-07', 'short', 'Got it, thanks.'),
    ('short-08', 'short', 'See you tomorrow.'),
    ('short-09', 'short', 'Ship it.'),
    ('short-10', 'short', 'Can you hear me?'),
    ('short-11', 'short', 'Running late.'),
    ('short-12', 'short', 'Looks great to me.'),
    ('med-01', 'medium', 'Please send the report by Friday.'),
    ('med-02', 'medium', 'I do not want to delete the backup.'),
    ('med-03', 'medium', 'Hello, how are you doing this morning?'),
    ('med-04', 'medium', 'Can we move the meeting to next Tuesday afternoon?'),
    ('med-05', 'medium', 'The deployment failed again, so I rolled it back.'),
    ('med-06', 'medium', 'Remind me to buy milk, eggs, and bread on the way home.'),
    ('med-07', 'medium', "I think we should wait until the client replies before we change anything."),
    ('med-08', 'medium', 'Our quarterly revenue grew faster than we expected.'),
    ('med-09', 'medium', "Let's schedule a quick call with Sarah and Michael."),
    ('med-10', 'medium', 'The new build is ready for testing on the staging server.'),
    ('med-11', 'medium', 'Could you review my pull request when you get a chance?'),
    ('med-12', 'medium', 'Please turn off the lights in the conference room.'),
    ('med-13', 'medium', 'We need to update the documentation before the release.'),
    ('med-14', 'medium', 'Honestly, I did not expect the results to be this good.'),
    ('num-01', 'numeric', 'The meeting starts at 3:30 PM.'),
    ('num-02', 'numeric', 'The invoice total is $5,000.'),
    ('num-03', 'numeric', 'My flight leaves on March 14th at 7 AM.'),
    ('num-04', 'numeric', 'We sold 250 units in the first week.'),
    ('long-01', 'long', "Hi Jordan, thanks for sending over the draft. I read through it last night and I think the structure works well, but the second section needs more detail about the budget. Can we talk about it tomorrow morning?"),
    ('long-02', 'long', "Quick update on the project: the backend migration is finished, the mobile app is still in review, and we are waiting on legal to approve the new terms of service. I expect everything to be ready by the end of the month."),
    ('long-03', 'long', "When you get a chance, please check whether the analytics dashboard is loading correctly for customers in Europe. A few people reported that the charts are empty, and I suspect it is related to the time zone change we made yesterday."),
    ('long-04', 'long', "I wanted to follow up on our conversation from last week. We decided to postpone the launch until the performance issues are fixed, and the team agreed to focus on reliability for the next two sprints."),
    ('long-05', 'long', "Dear Professor Williams, I am writing to ask whether it would be possible to extend the deadline for my research paper. I have been unwell for most of this week and I would appreciate a few extra days to finish the analysis."),
    ('long-06', 'long', "Note to self: compare the three vendors on price, support quality, and integration effort, then write a short summary for the leadership meeting on Thursday."),
]


def pad_and_noise(pcm: np.ndarray, snr_db, seed) -> np.ndarray:
    audio = pcm.astype(np.float64)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        active = audio[np.abs(audio) > 200]
        rms = np.sqrt(np.mean(active ** 2)) if len(active) else 1.0
        audio = audio + rng.normal(0, rms / (10 ** (snr_db / 20)), len(audio))
    audio = np.clip(audio, -32768, 32767).astype('<i2')
    return np.pad(audio, (SR // 4, SR // 4))


def write_wav(path: Path, pcm: np.ndarray):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), 'rb') as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')


def tts(text, voice, dest: Path):
    aiff = dest.with_suffix('.aiff')
    subprocess.run(['say', '-v', voice, '-o', str(aiff), text], check=True)
    subprocess.run(['afconvert', '-f', 'WAVE', '-d', f'LEI16@{SR}', '-c', '1', str(aiff), str(dest)], check=True)
    aiff.unlink()


def main():
    AUDIO.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (sid, cat, ref) in enumerate(SENTENCES):
        voices = [VOICES[(i + k * 2) % len(VOICES)] for k in range(3)]
        for k, voice in enumerate(voices):
            variants = [(None, '')] + ([(20, '-snr20')] if k == 0 else [])
            raw = AUDIO / f'_raw-{sid}-{voice}.wav'
            tts(ref, voice, raw)
            base = read_wav(raw); raw.unlink()
            for snr, suffix in variants:
                cid = f'tts-{sid}-{voice.lower()}{suffix}'
                pcm = pad_and_noise(base, snr, 20261107 + i * 10 + k)
                path = AUDIO / f'{cid}.wav'; write_wav(path, pcm)
                rows.append(dict(id=cid, part='tts', category=cat, voice=voice, snr_db=snr, reference=ref,
                                 cased_reference=True, duration_s=round(len(pcm) / SR, 3), path=str(path),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(sid, voices, flush=True)

    try:
        import pyarrow.parquet as pq, soundfile as sf
        from huggingface_hub import hf_hub_download
        p = hf_hub_download('hf-internal-testing/librispeech_asr_dummy', 'clean/validation-00000-of-00001.parquet',
                            repo_type='dataset')
        table = pq.read_table(p).to_pylist()
        for r in table:
            data, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='int16')
            assert sr == SR
            cid = f"libri-{r['id']}"
            path = AUDIO / f'{cid}.wav'; write_wav(path, data.astype('<i2'))
            rows.append(dict(id=cid, part='librispeech', category='real-read', voice=f"spk{r['speaker_id']}", snr_db=None,
                             reference=r['text'], cased_reference=False, duration_s=round(len(data) / SR, 3),
                             path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f'librispeech dummy: {len(table)} clips', flush=True)
    except Exception as e:  # pragma: no cover
        print('LibriSpeech part skipped:', e, file=sys.stderr)

    (OUT / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'{len(rows)} clips -> {OUT / "manifest.jsonl"}')


if __name__ == '__main__':
    main()
