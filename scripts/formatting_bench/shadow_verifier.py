#!/usr/bin/env python3
"""Reference-free edit features, not a calibrated probability or production gate."""
import argparse, difflib, json, re
from collections import Counter
from pathlib import Path

WORDS = re.compile(r"\w+(?:['’]\w+)?")
IDENTIFIERS = re.compile(r'https?://[^\s]+|[\w.+-]+@[\w.-]+|\b\w*[a-z][A-Z]\w*\b|\b\d+(?:\.\d+){2,}\b')
NEGATIONS = {'not', 'never', 'no', "don't", "doesn't", "isn't", "can't", "won't", "didn't", "shouldn't"}

def features(source, candidate, context=''):
    left, right = WORDS.findall(source), WORDS.findall(candidate)
    a,b=[w.casefold() for w in left],[w.casefold() for w in right]
    context_words=set(w.casefold() for w in WORDS.findall(context))
    edits=[]
    for op,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if op=='equal': continue
        edits.append(dict(operation=op, source_words=left[i:j], candidate_words=right[k:l],
            context_supported_words=[w for w in right[k:l] if w.casefold() in context_words]))
    def numbers(text):
        # Ordered-list markers express structure, not changed numeric content.
        text=re.sub(r'(?m)^\s*\d+[.)]\s+', '', text)
        return re.findall(r'\d+(?:[.,:/-]\d+)*',text)
    def items(text):
        return [m.group(1) for m in re.finditer(r'(?m)^\s*(?:[-*•]|\d+[.)])\s+(.+)$',text)]
    old_items,new_items=items(source),items(candidate)
    return dict(edits=edits, word_edit_count=len(edits),
        identifiers_lost=[s for s in IDENTIFIERS.findall(source) if s not in candidate],
        numeric_content_changed=Counter(numbers(source))!=Counter(numbers(candidate)),
        negation_changed=Counter(w for w in a if w in NEGATIONS)!=Counter(w for w in b if w in NEGATIONS),
        source_list_items=len(old_items), candidate_list_items=len(new_items),
        existing_list_item_count_changed=bool(old_items) and len(old_items)!=len(new_items),
        # Structure-only changes retain lexical content and ordering.
        punctuation_or_structure_only=a==b,
        context_available=bool(context.strip()))

def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    data=json.loads(a.results.read_text())
    a.output.write_text(json.dumps(dict(mode='shadow_features_only_no_probability',backend=data['backend'],
        rows=[dict(id=r['id'],features=features(r['input'],r['output'],r.get('context',''))) for r in data['rows']]),indent=2)+'\n')

if __name__=='__main__': main()
