"""Recomputes inter-annotator agreement and extractor precision/recall/F1 from labels_sampleA.csv.
Run inside v2/ (after copying ../analysis.py and ../medical_conversations.csv there): python annotation/reproduce_metrics.py"""
import sys, json, pandas as pd
sys.path.insert(0, ".")
import extract as E
from sklearn.metrics import cohen_kappa_score
CON = list(json.load(open('symptom_lexicon_v2.json')))
lab = pd.read_csv('annotation/labels_sampleA.csv').fillna('')
conv = pd.read_csv('medical_conversations.csv')[['conversations', 'disease']].dropna().reset_index(drop=True).conversations
S = lambda x: {t.strip() for t in x.split(';') if t.strip() in CON}
A, B, G = lab.A_concepts.map(S), lab.B_concepts.map(S), lab.adjudicated_concepts.map(S)
print('kappa', round(cohen_kappa_score([int(c in a) for a in A for c in CON], [int(c in b) for b in B for c in CON]), 3))
P = [set(E.extract(conv[int(i[1:])])[0]) for i in lab.dialogue_id]
tp = sum(len(p & g) for p, g in zip(P, G)); fp = sum(len(p - g) for p, g in zip(P, G)); fn = sum(len(g - p) for p, g in zip(P, G))
p, r = tp / (tp + fp), tp / (tp + fn); print('precision', round(p, 3), 'recall', round(r, 3), 'F1', round(2 * p * r / (p + r), 3))
