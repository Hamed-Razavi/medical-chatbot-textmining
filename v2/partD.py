"""Evaluation-bootstrap CIs for K-means (K=24): cluster once per seed on all dialogues, resample dialogues for the metric."""
import json, warnings, numpy as np
warnings.filterwarnings('ignore')
import common as C
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, adjusted_mutual_info_score as AMI
y = C.y; labs = {}
for v, col in C.VIEWS.items():
    X = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True).fit_transform(C.df[col].str.join(' '))
    labs[v] = [KMeans(24, n_init=10, random_state=s).fit_predict(X) for s in range(5)]
B = 1000; rs = np.random.RandomState(11); idxs = [rs.choice(len(y), len(y)) for _ in range(B)]
vals = {v: {'ari': [], 'ami': []} for v in C.VIEWS}
for ix in idxs:
    for v in C.VIEWS:
        vals[v]['ari'].append(np.mean([ARI(y[ix], l[ix]) for l in labs[v]])); vals[v]['ami'].append(np.mean([AMI(y[ix], l[ix]) for l in labs[v]]))
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
point = {v: {'ari': float(np.mean([ARI(y, l) for l in labs[v]])), 'ami': float(np.mean([AMI(y, l) for l in labs[v]]))} for v in C.VIEWS}
OUT = {'point': point, 'ci': {v: {m: ci(a) for m, a in d.items()} for v, d in vals.items()}}
pairs = [('full', 'patient'), ('full_masked', 'patient_masked'), ('full_masked_lenmatched', 'patient_masked'), ('bot', 'patient'), ('bot_masked', 'patient_masked'), ('patient', 'patient_masked')]
OUT['diff'] = {f'{a}-{b}': {m: {'point': point[a][m] - point[b][m], 'ci': ci(np.subtract(vals[a][m], vals[b][m]))} for m in ['ari', 'ami']} for a, b in pairs}
json.dump(OUT, open('results_partD.json', 'w'), indent=1)
for v in C.VIEWS: print(v, {m: [round(point[v][m], 3)] + [round(x, 3) for x in OUT['ci'][v][m]] for m in ['ari', 'ami']})
for k, d in OUT['diff'].items(): print(k, {m: [round(d[m]['point'], 3)] + [round(x, 3) for x in d[m]['ci']] for m in d})
