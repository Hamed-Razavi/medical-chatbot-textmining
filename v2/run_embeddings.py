"""Sentence-embedding baseline for the IEEE Access revision (run on your own computer).
    pip install sentence-transformers pandas numpy
    python run_embeddings.py
Output: embeddings.npz  (upload this file back to Claude). Takes about 5-15 minutes on a laptop CPU."""
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
df = pd.read_csv('views_raw.csv').fillna('')
VIEWS = ['patient', 'patient_masked', 'bot', 'bot_masked', 'full', 'full_masked']
MODELS = {'minilm': 'sentence-transformers/all-MiniLM-L6-v2', 'mpnet': 'sentence-transformers/all-mpnet-base-v2'}
out = {'dialogue_id': df.dialogue_id.values.astype(str)}
for short, name in MODELS.items():
    m = SentenceTransformer(name)
    for v in VIEWS:
        print(short, v, flush=True)
        out[f'{short}__{v}'] = m.encode(df[v].tolist(), batch_size=32, show_progress_bar=True, normalize_embeddings=True).astype(np.float16)
np.savez_compressed('embeddings.npz', **out)
print('Wrote embeddings.npz')
