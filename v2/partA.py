"""Revision analyses, part A: views, clustering, topic models, supervised reference, baselines, sensitivity."""
import json, warnings, gzip, numpy as np, pandas as pd, time
warnings.filterwarnings('ignore')
import common as C
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import (silhouette_score, adjusted_rand_score as ARI, adjusted_mutual_info_score as AMI,
                             normalized_mutual_info_score as NMI, f1_score, accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from scipy.stats import binomtest
t0 = time.time(); df = C.df; y = C.y; OUT = {}
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)
docs = {v: df[c].str.join(' ') for v, c in C.VIEWS.items()}
def tfidf(v): return TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True).fit_transform(docs[v])

# ---- 1. K-means: label-free K selection + external metrics, all views
KS = [5, 10, 15, 20, 24, 30, 40]; km = {}
for v in C.VIEWS:
    X = tfidf(v); rows = []
    for k in KS:
        r = {'k': k, 'sil': [], 'ari': [], 'ami': [], 'nmi': []}
        for s in range(5):
            lab = KMeans(k, n_init=10, random_state=s).fit_predict(X)
            r['sil'].append(silhouette_score(X, lab, metric='cosine')); r['ari'].append(ARI(y, lab)); r['ami'].append(AMI(y, lab)); r['nmi'].append(NMI(y, lab))
        rows.append({kk: (float(np.mean(vv)) if isinstance(vv, list) else vv) for kk, vv in r.items()} | {kk + '_sd': float(np.std(vv)) for kk, vv in r.items() if isinstance(vv, list)})
    t = pd.DataFrame(rows); kfree = int(t.loc[t.sil.idxmax(), 'k'])
    km[v] = {'sweep': t.round(4).to_dict('records'), 'k_labelfree': kfree}
    log('kmeans', v, 'k*', kfree, t.loc[t.k == 24, ['ari', 'ami']].round(3).values.tolist())
OUT['kmeans'] = km

# ---- 2. Bootstrap CIs over dialogues (refit K-means on each resample), paired across views, K = 24
B = 200; boot = {v: {'ari': [], 'ami': []} for v in C.VIEWS}
mats = {v: tfidf(v) for v in C.VIEWS}
for b in range(B):
    idx = np.random.RandomState(1000 + b).choice(len(y), len(y), replace=True)
    for v in C.VIEWS:
        lab = KMeans(24, n_init=3, random_state=b).fit_predict(mats[v][idx])
        boot[v]['ari'].append(ARI(y[idx], lab)); boot[v]['ami'].append(AMI(y[idx], lab))
    if b % 50 == 0: log('boot', b)
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
OUT['kmeans_boot'] = {v: {m: {'mean': float(np.mean(a)), 'ci': ci(a)} for m, a in d.items()} for v, d in boot.items()}
pairs = [('full', 'patient'), ('full_masked', 'patient_masked'), ('full_masked_lenmatched', 'patient_masked'), ('bot', 'patient'), ('bot_masked', 'patient_masked'), ('patient', 'patient_masked')]
OUT['kmeans_boot_diff'] = {f'{a}-{b}': {m: {'mean': float(np.mean(np.subtract(boot[a][m], boot[b][m]))), 'ci': ci(np.subtract(boot[a][m], boot[b][m])),
                                            'p_le0': float(np.mean(np.subtract(boot[a][m], boot[b][m]) <= 0))} for m in ['ari', 'ami']} for a, b in pairs}
log('boot done')

# ---- 3. Additional unsupervised baselines (K = 24): NMF, LSA + K-means, GloVe mean-embedding + K-means
glove = {}
with gzip.open('emb/glove100.gz', 'rt', encoding='utf8') as f:
    next(f)
    for line in f:
        p = line.rstrip().split(' ');
        glove[p[0]] = np.asarray(p[1:], dtype=np.float32)
log('glove loaded', len(glove))
def emb(toks):
    vs = [glove[w] for w in toks if w in glove]
    return np.mean(vs, 0) if vs else np.zeros(100, np.float32)
base = {}
for v in ['patient', 'full', 'patient_masked', 'full_masked']:
    X = tfidf(v); r = {}
    res = []
    for s in range(5):
        W = NMF(24, init='nndsvda', random_state=s, max_iter=500).fit_transform(X); lab = W.argmax(1); res.append((ARI(y, lab), AMI(y, lab)))
    r['NMF'] = np.mean(res, 0).tolist() + np.std(res, 0).tolist()
    Z = normalize(TruncatedSVD(100, random_state=0).fit_transform(X)); res = []
    for s in range(5):
        lab = KMeans(24, n_init=10, random_state=s).fit_predict(Z); res.append((ARI(y, lab), AMI(y, lab)))
    r['LSA+KMeans'] = np.mean(res, 0).tolist() + np.std(res, 0).tolist()
    E = normalize(np.vstack(df[C.VIEWS[v]].apply(emb))); res = []
    for s in range(5):
        lab = KMeans(24, n_init=10, random_state=s).fit_predict(E); res.append((ARI(y, lab), AMI(y, lab)))
    r['GloVe+KMeans'] = np.mean(res, 0).tolist() + np.std(res, 0).tolist()
    r['oov_rate'] = float(1 - np.mean([np.mean([w in glove for w in t]) if t else 1 for t in df[C.VIEWS[v]]]))
    base[v] = r; log('baselines', v, {k: np.round(vv[:2], 3).tolist() for k, vv in r.items() if k != 'oov_rate'})
OUT['baselines_k24'] = base

# ---- 4. LDA on all views: K = 15 and 24, three seeds; C_V, NPMI, AMI of dominant topic
lda = {}
for v, col in C.VIEWS.items():
    texts = df[col].tolist(); d = Dictionary(texts); d.filter_extremes(no_below=2, no_above=0.5); corp = [d.doc2bow(t) for t in texts]
    lda[v] = {}
    for k in [15, 24]:
        cv, npmi, ami, nmi = [], [], [], []
        for s in (1, 2, 3):
            m = LdaModel(corp, id2word=d, num_topics=k, passes=20, iterations=200, alpha='auto', eta='auto', random_state=s)
            cv.append(CoherenceModel(model=m, texts=texts, dictionary=d, coherence='c_v', topn=10).get_coherence())
            npmi.append(CoherenceModel(model=m, texts=texts, dictionary=d, coherence='c_npmi', topn=10).get_coherence())
            dom = [max(m.get_document_topics(b_, minimum_probability=0), key=lambda x: x[1])[0] for b_ in corp]
            ami.append(AMI(y, dom)); nmi.append(NMI(y, dom))
        lda[v][k] = {'cv': [float(np.mean(cv)), float(np.std(cv))], 'npmi': [float(np.mean(npmi)), float(np.std(npmi))],
                     'ami': [float(np.mean(ami)), float(np.std(ami))], 'nmi': [float(np.mean(nmi)), float(np.std(nmi))]}
    log('lda', v, {k: [round(lda[v][k]['cv'][0], 3), round(lda[v][k]['ami'][0], 3)] for k in lda[v]})
OUT['lda_views'] = lda

# ---- 5. Supervised reference: nested CV (TF-IDF fitted inside folds, C tuned in inner CV), 5x3 repeated
def nested(textser, ngram=(1, 2), Xmat=None):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42); f1s, accs, Cs = [], [], []
    oof = np.empty(len(y), dtype=object)
    for i, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y)):
        if Xmat is None:
            pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=ngram), LogisticRegression(max_iter=4000))
            gs = GridSearchCV(pipe, {'logisticregression__C': [0.1, 1, 10, 100]}, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring='f1_macro')
            gs.fit(textser.iloc[tr], y[tr]); p = gs.predict(textser.iloc[te])
        else:
            gs = GridSearchCV(LogisticRegression(max_iter=4000), {'C': [0.1, 1, 10, 100]}, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring='f1_macro')
            gs.fit(Xmat[tr], y[tr]); p = gs.predict(Xmat[te])
        f1s.append(f1_score(y[te], p, average='macro')); accs.append(accuracy_score(y[te], p)); Cs.append(list(gs.best_params_.values())[0])
        if i < 5: oof[te] = p
    return {'f1': [float(np.mean(f1s)), float(np.std(f1s))], 'acc': [float(np.mean(accs)), float(np.std(accs))], 'C_chosen': pd.Series(Cs).value_counts().to_dict()}, oof
sup, oofs = {}, {}
for v in C.VIEWS:
    sup[v], oofs[v] = nested(docs[v]); log('LR', v, np.round(sup[v]['f1'], 3))
# symptom-feature model (52 concepts, patient-reported, extractor v2)
import extract as E
rep = df.conversations.apply(lambda c: E.extract(c)[0])
concepts = sorted(E.LEX); Xs = np.array([[int(c in r) for c in concepts] for r in rep])
sup['symptom_concepts_52'], oofs['symptom_concepts_52'] = nested(None, Xmat=Xs); log('LR symptoms', sup['symptom_concepts_52']['f1'])
OUT['supervised'] = sup
def boot_f1(p, B=1000):
    r = np.random.RandomState(0); v = []
    for _ in range(B):
        i = r.choice(len(y), len(y)); v.append(f1_score(y[i], p[i], average='macro'))
    return ci(v)
OUT['supervised_oof_ci'] = {v: {'f1_oof': float(f1_score(y, p, average='macro')), 'ci': boot_f1(p)} for v, p in oofs.items()}
def mcnemar(a, b):
    ca, cb = (oofs[a] == y), (oofs[b] == y); n01 = int((~ca & cb).sum()); n10 = int((ca & ~cb).sum())
    return {'a_only_correct': n10, 'b_only_correct': n01, 'p_exact': float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0}
OUT['mcnemar'] = {f'{a} vs {b}': mcnemar(a, b) for a, b in [('full', 'patient'), ('full_masked', 'patient_masked'), ('full_masked_lenmatched', 'patient_masked'),
                                                              ('patient', 'patient_masked'), ('bot_masked', 'patient_masked'), ('patient_masked', 'symptom_concepts_52')]}
log('mcnemar', OUT['mcnemar'])

# ---- 6. Preprocessing sensitivity (patient view, masked LR + K-means K=24)
import re
def prep_variant(text, filler=True, minlen=3, lemma=True):
    stop = C.STOP if filler else C.SW
    t = re.sub(r'[^a-z\s]', ' ', re.sub(r"'", '', text.lower().replace('’', "'")))
    toks = [w for w in t.split() if len(w) >= minlen and w not in stop]
    if lemma: toks = [C.lem.lemmatize(w) for w in toks]; toks = [w for w in toks if w not in stop and C.lem.lemmatize(w, 'v') not in stop]
    return [w for w in toks if w not in C.DIS_WORDS]
sens = {}
for name, kw in {'default': {}, 'no_filler_list': {'filler': False}, 'min_len_2': {'minlen': 2}, 'no_lemmatization': {'lemma': False}}.items():
    toks = df.user_text.apply(lambda t: prep_variant(t, **kw)); d_ = toks.str.join(' ')
    X = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True).fit_transform(d_)
    labs = [KMeans(24, n_init=10, random_state=s).fit_predict(X) for s in range(5)]
    skf = StratifiedKFold(5, shuffle=True, random_state=42); f1 = []
    for tr, te in skf.split(d_, y):
        p = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)), LogisticRegression(C=10, max_iter=4000)).fit(d_.iloc[tr], y[tr]).predict(d_.iloc[te]); f1.append(f1_score(y[te], p, average='macro'))
    sens[name] = {'ari': float(np.mean([ARI(y, l) for l in labs])), 'ami': float(np.mean([AMI(y, l) for l in labs])), 'f1_masked': float(np.mean(f1)), 'vocab': int(X.shape[1])}
    log('sens', name, sens[name])
OUT['preproc_sensitivity'] = sens
json.dump(OUT, open('results_partA.json', 'w'), indent=1, default=float)
log('DONE')
