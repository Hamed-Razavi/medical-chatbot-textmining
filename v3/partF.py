"""Sentence-embedding baselines (MiniLM, MPNet) with K-means and BERTopic at K=24."""
import json, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, adjusted_mutual_info_score as AMI
z = np.load('emb_in/embeddings.npz', allow_pickle=False)
raw = pd.read_csv('emb_pkg/views_raw.csv').fillna('')
y = pd.factorize(pd.read_csv('medical_conversations.csv').disease)[0]
VIEWS = ['patient', 'patient_masked', 'bot', 'bot_masked', 'full', 'full_masked']
out = {'kmeans': {}, 'bertopic': {}}; t0 = time.time()
for m in ['minilm', 'mpnet']:
    for v in VIEWS:
        X = z[f'{m}__{v}'].astype(np.float32)
        r = []
        for s in range(5):
            lab = KMeans(24, n_init=10, random_state=s).fit_predict(X); r.append((ARI(y, lab), AMI(y, lab)))
        r = np.array(r); out['kmeans'][f'{m}__{v}'] = {'ari': r[:, 0].mean(), 'ari_sd': r[:, 0].std(ddof=1), 'ami': r[:, 1].mean(), 'ami_sd': r[:, 1].std(ddof=1)}
        print(f'[{time.time()-t0:5.0f}s] km', m, v, np.round(r.mean(0), 3), flush=True)
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
for v in ['patient', 'patient_masked', 'full', 'full_masked']:
    X = z[f'mpnet__{v}'].astype(np.float32); docs = raw[v].tolist(); r = []
    for s in range(3):
        tm = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=s),
                      hdbscan_model=HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
                      nr_topics=24, calculate_probabilities=False, verbose=False)
        topics, _ = tm.fit_transform(docs, X)
        topics = np.array(topics); outl = (topics == -1).mean()
        t2 = np.array(tm.reduce_outliers(docs, topics, strategy='embeddings', embeddings=X))
        r.append((ARI(y, topics), AMI(y, topics), ARI(y, t2), AMI(y, t2), outl, len(set(topics) - {-1})))
    r = np.array(r)
    out['bertopic'][v] = dict(zip(['ari_raw', 'ami_raw', 'ari', 'ami', 'outlier_share', 'n_topics'], r.mean(0).tolist())) | {'ari_sd': r[:, 2].std(ddof=1)}
    print(f'[{time.time()-t0:5.0f}s] bertopic', v, np.round(r.mean(0), 3), flush=True)
json.dump(out, open('results_partF.json', 'w'), indent=1); print('DONE')
