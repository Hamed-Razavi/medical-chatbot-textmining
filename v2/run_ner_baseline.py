"""Transformer NER baseline for the IEEE Access revision.
Runs the biomedical NER model of Raza et al. (d4data/biomedical-ner-all) on the patient turns of the
120 annotated dialogues and maps 'Sign_symptom' entities to the 52 lexicon concepts.

Usage (on a computer with internet access):
    pip install transformers torch pandas
    python run_ner_baseline.py path/to/medical_conversations.csv
Output: ner_predictions.csv  (upload this file back to Claude)"""
import sys, re, json, pandas as pd
from transformers import pipeline
conv = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'medical_conversations.csv')[['conversations', 'disease']].dropna().reset_index(drop=True)
conv['dialogue_id'] = ['D%03d' % i for i in range(len(conv))]
ids = pd.read_csv('annotation/key_sampleA.csv').dialogue_id
sub = conv[conv.dialogue_id.isin(ids)].copy()
LEX = {c: [re.compile(p) for p in ps] for c, ps in json.load(open('symptom_lexicon_v2.json')).items()}
ner = pipeline('ner', model='d4data/biomedical-ner-all', aggregation_strategy='first')
def patient_text(c):
    return ' '.join(t.strip()[5:].strip() for t in c.split('</s>') if t.strip().startswith('User:'))
rows = []
for _, r in sub.iterrows():
    text = patient_text(r.conversations).replace('’', "'")
    ents = [e for e in ner(text) if e['entity_group'] == 'Sign_symptom' and e['score'] >= 0.5]
    spans = [e['word'].strip() for e in ents]
    concepts, unmapped = set(), []
    for sp in spans:
        hit = {c for c, ps in LEX.items() if any(p.search(sp.lower()) for p in ps)}
        concepts |= hit
        if not hit: unmapped.append(sp)
    rows.append({'dialogue_id': r.dialogue_id, 'ner_concepts': ';'.join(sorted(concepts)), 'ner_spans': ' | '.join(spans), 'unmapped_spans': ' | '.join(unmapped)})
pd.DataFrame(rows).to_csv('ner_predictions.csv', index=False)
print('Wrote ner_predictions.csv for', len(rows), 'dialogues')
