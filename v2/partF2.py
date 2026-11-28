"""Evaluation-bootstrap CIs for sentence-embedding K-means (K=24), same resamples as partD."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, adjusted_mutual_info_score as AMI
z = np.load('embeddings.npz', allow_pickle=False)
y = pd.factorize(pd.read_csv('medical_conversations.csv').disease)[0]
VIEWS = ['patient', 'patient_masked', 'bot', 'bot_masked', 'full', 'full_masked']
keys = [f'{m}__{v}' for m in ['minilm', 'mpnet'] for v in VIEWS]
labs = {k: [KMeans(24, n_init=10, random_state=s).fit_predict(z[k].astype(np.float32)) for s in range(5)] for k in keys}
rs = np.random.RandomState(11); idxs = [rs.choice(len(y), len(y)) for _ in range(1000)]
vals = {k: {'ari': [], 'ami': []} for k in keys}
for ix in idxs:
    for k in keys:
        vals[k]['ari'].append(np.mean([ARI(y[ix], l[ix]) for l in labs[k]])); vals[k]['ami'].append(np.mean([AMI(y[ix], l[ix]) for l in labs[k]]))
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
point = {k: {'ari': float(np.mean([ARI(y, l) for l in labs[k]])), 'ami': float(np.mean([AMI(y, l) for l in labs[k]]))} for k in keys}
OUT = {'point': point, 'ci': {k: {m: ci(a) for m, a in d.items()} for k, d in vals.items()}, 'diff': {}}
for mdl in ['minilm', 'mpnet']:
    for a, b in [('full_masked', 'patient_masked'), ('bot_masked', 'patient_masked'), ('full', 'patient'), ('patient', 'patient_masked')]:
        A, Bk = f'{mdl}__{a}', f'{mdl}__{b}'
        OUT['diff'][f'{A}-{b}'] = {m: {'point': point[A][m] - point[Bk][m], 'ci': ci(np.subtract(vals[A][m], vals[Bk][m]))} for m in ['ari', 'ami']}
json.dump(OUT, open('results_partF2.json', 'w'), indent=1)
for k, d in OUT['diff'].items(): print(k, {m: [round(d[m]['point'], 3)] + [round(x, 3) for x in d[m]['ci']] for m in d})
