#!/usr/bin/env python3
"""Download pinned assets; never execute Hugging Face remote model code."""
import hashlib, json, subprocess, urllib.request
from run import CACHE, MODELS
from huggingface_hub import snapshot_download

CACHE.mkdir(parents=True, exist_ok=True)
for name, model in MODELS.items():
    snapshot_download(model['repo'], revision=model['revision'], allow_patterns=['*.json','*.txt','*.model','*.yaml','*.onnx','*.safetensors'])
source = CACHE / 'gector-source'
if not source.exists(): subprocess.run(['git','clone','https://github.com/gotutiyan/gector',str(source)],check=True)
subprocess.run(['git','-C',str(source),'checkout',MODELS['gector']['source_revision']],check=True)
revision = MODELS['gector']['verb_dictionary_revision']
with urllib.request.urlopen(f'https://raw.githubusercontent.com/grammarly/gector/{revision}/data/verb-form-vocab.txt') as response:
    (CACHE/'verb-form-vocab.txt').write_bytes(response.read())
print('Prepared pinned models and reference implementation.')
