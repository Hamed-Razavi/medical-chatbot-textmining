"""LDA sweep with AMI (patient and full views) and dialogue-bootstrap CIs for LDA AMI at K = 15 and 24."""
import json, warnings, numpy as np
warnings.filterwarnings('ignore')
import common as C
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from sklearn.metrics import adjusted_mutual_info_score as AMI, normalized_mutual_info_score as NMI
y = C.y; OUT = {}
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
for v in ['patient', 'full']:
    texts = C.df[C.VIEWS[v]].tolist(); d = Dictionary(texts); d.filter_extremes(no_below=2, no_above=0.5); corp = [d.doc2bow(t) for t in texts]
    rows = []
    for k in [5, 10, 15, 20, 24, 30]:
        cv, npmi, ami, nmi, boots = [], [], [], [], []
        for s in (1, 2, 3):
            m = LdaModel(corp, id2word=d, num_topics=k, passes=20, iterations=200, alpha='auto', eta='auto', random_state=s)
            cv.append(CoherenceModel(model=m, texts=texts, dictionary=d, coherence='c_v', topn=10).get_coherence())
            npmi.append(CoherenceModel(model=m, texts=texts, dictionary=d, coherence='c_npmi', topn=10).get_coherence())
            dom = np.array([max(m.get_document_topics(b, minimum_probability=0), key=lambda x: x[1])[0] for b in corp])
            ami.append(AMI(y, dom)); nmi.append(NMI(y, dom))
            if s == 1:
                r = np.random.RandomState(0)
                for _ in range(500):
                    i = r.choice(len(y), len(y)); boots.append(AMI(y[i], dom[i]))
        rows.append({'k': k, 'cv': np.mean(cv), 'cv_sd': np.std(cv), 'npmi': np.mean(npmi), 'npmi_sd': np.std(npmi), 'ami': np.mean(ami), 'ami_sd': np.std(ami),
                     'nmi': np.mean(nmi), 'ami_boot_ci_seed1': ci(boots)})
        print(v, rows[-1], flush=True)
    OUT[v] = rows
json.dump(OUT, open('results_partC.json', 'w'), indent=1, default=float)
print('DONE')
