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

import extract as E
from sklearn.metrics import f1_score
from scipy.stats import binomtest
A = json.load(open('results_partA.json'))
sup_pm, oof_pm = nested(docs['patient_masked'])
rep = df.conversations.apply(lambda c: E.extract(c)[0]); concepts = sorted(E.LEX)
Xs = np.array([[int(c in r) for c in concepts] for r in rep])
sup_s, oof_s = nested(None, Xmat=Xs)
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
def boot_f1(p, B=1000):
    r = np.random.RandomState(0); v = []
    for _ in range(B):
        i = r.choice(len(y), len(y)); v.append(f1_score(y[i], p[i], average='macro'))
    return ci(v)
A['supervised']['symptom_concepts_52'] = sup_s
A['supervised_oof_ci']['symptom_concepts_52'] = {'f1_oof': float(f1_score(y, oof_s, average='macro')), 'ci': boot_f1(oof_s)}
ca, cb = (oof_pm == y), (oof_s == y); n10 = int((ca & ~cb).sum()); n01 = int((~ca & cb).sum())
A['mcnemar']['patient_masked vs symptom_concepts_52'] = {'a_only_correct': n10, 'b_only_correct': n01, 'p_exact': float(binomtest(n10, n10 + n01, 0.5).pvalue)}
A['_note_symptom_model'] = 'recomputed with frozen extractor v2'
json.dump(A, open('results_partA.json', 'w'), indent=1, default=float)
print(sup_pm['f1'], sup_s, A['mcnemar']['patient_masked vs symptom_concepts_52'])
