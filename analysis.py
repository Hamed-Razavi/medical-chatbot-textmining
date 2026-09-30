import re, json, random, numpy as np, pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, f1_score, accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

SEED = 42; random.seed(SEED); np.random.seed(SEED)
OUT = {}
df = pd.read_csv('medical_conversations.csv')[['conversations', 'disease']].dropna().reset_index(drop=True)

# ---------- 1. Dataset statistics ----------
turns = df.conversations.str.split('</s>').apply(lambda t: [x.strip() for x in t if x.strip()])
df['n_turns'] = turns.str.len()
df['user_text'] = turns.apply(lambda t: ' '.join(x.split(':', 1)[1] for x in t if x.startswith('User:')))
df['bot_text'] = turns.apply(lambda t: ' '.join(x.split(':', 1)[1] for x in t if x.startswith('Bot:')))
df['full_text'] = turns.apply(lambda t: ' '.join(x.split(':', 1)[1] if ':' in x else x for x in t))
df['n_user'] = turns.apply(lambda t: sum(x.startswith('User:') for x in t))
df['n_words'] = df.full_text.str.split().str.len()
df['n_user_words'] = df.user_text.str.split().str.len()
def names_label(row):
    keys = {'common cold': ['cold'], 'bronchial asthma': ['asthma'], 'urinary tract infection': ['urinary tract infection', 'uti'],
            'gastroesophageal reflux disease': ['gerd', 'reflux'], 'peptic ulcer disease': ['ulcer'], 'dimorphic hemorrhoids': ['hemorrhoid'],
            'chicken pox': ['chickenpox', 'chicken pox'], 'fungal infection': ['fungal'], 'drug reaction': ['drug reaction', 'reaction to'],
            'cervical spondylosis': ['spondylosis'], 'varicose veins': ['varicose']}.get(row.disease, [row.disease.split()[-1]])
    b = row.bot_text.lower(); u = row.user_text.lower()
    return pd.Series({'bot_names': any(k in b for k in keys), 'user_names': any(k in u for k in keys)})
df[['bot_names', 'user_names']] = df.apply(names_label, axis=1)
OUT['stats'] = {
    'n_dialogues': len(df), 'n_classes': df.disease.nunique(),
    'per_class_min': int(df.disease.value_counts().min()), 'per_class_max': int(df.disease.value_counts().max()),
    'turns_mean': round(df.n_turns.mean(), 2), 'turns_sd': round(df.n_turns.std(), 2), 'turns_min': int(df.n_turns.min()), 'turns_max': int(df.n_turns.max()),
    'turns_median': float(df.n_turns.median()),
    'user_turns_mean': round(df.n_user.mean(), 2), 'words_mean': round(df.n_words.mean(), 1), 'words_sd': round(df.n_words.std(), 1),
    'user_words_mean': round(df.n_user_words.mean(), 1), 'user_words_sd': round(df.n_user_words.std(), 1),
    'vocab_full': len(set(' '.join(df.full_text).lower().split())),
    'bot_names_label_pct': round(100 * df.bot_names.mean(), 1), 'user_names_label_pct': round(100 * df.user_names.mean(), 1),
    'duplicates': int(df.conversations.duplicated().sum())}

# ---------- 2. Preprocessing ----------
SW = set(stopwords.words('english'))
FILLER = {'yes', 'no', 'ive', 'im', 'theyre', 'dont', 'cant', 'id', 'ill', 'youre', 'thats', 'its', 'also', 'really', 'lot', 'bit',
          'like', 'get', 'got', 'getting', 'feel', 'feeling', 'feels', 'felt', 'sometimes', 'lately', 'recently', 'could', 'would', 'might',
          'should', 'thing', 'think', 'know', 'sure', 'way', 'ok', 'okay', 'thanks', 'thank', 'user', 'bot', 'much', 'even', 'well',
          'every', 'time', 'day', 'days', 'week', 'weeks', 'noticed', 'notice', 'go', 'going', 'make', 'one', 'still', 'something', 'anything',
          'try', 'help', 'good', 'idea', 'see', 'doctor', 'recommend', 'possible', 'sound', 'sounds', 'seem', 'seems', 'experiencing', 'experience',
          'symptom', 'sign', 'common', 'usually', 'often', 'lot', 'able', 'need', 'use', 'using', 'take', 'taking', 'want', 'let', 'let', 'mean', 'havent', 'hasnt', 'shes', 'hes', 'wasnt', 'isnt', 'doesnt', 'didnt', 'wont', 'weve', 'theyve', 'youve', 'arent', 'couldnt', 'wouldnt', 'kind', 'right', 'soon', 'away', 'keep', 'come', 'ago', 'last', 'long', 'change', 'cause'}
STOP = SW | FILLER
lem = WordNetLemmatizer()
def prep(text):
    t = text.lower().replace('’', "'")
    t = re.sub(r"'", '', t)             # I've -> ive (then removed as filler)
    t = re.sub(r'[^a-z\s]', ' ', t)
    toks = [lem.lemmatize(w) for w in t.split() if len(w) > 2 and w not in STOP]  # stop-words removed before lemmatisation (avoids 'was'->'wa')
    return [w for w in toks if w not in STOP and lem.lemmatize(w, 'v') not in STOP]
df['tok_user'] = df.user_text.apply(prep)
df['tok_full'] = df.full_text.apply(prep)
y = df.disease.values

def purity(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred); return cm.max(axis=0).sum() / cm.sum()

# ---------- 3. LDA ----------
def lda_eval(tok_col, ks=(5, 10, 15, 20, 24, 30), seeds=(1, 2, 3)):
    texts = df[tok_col].tolist(); d = Dictionary(texts); d.filter_extremes(no_below=2, no_above=0.5)
    corpus = [d.doc2bow(t) for t in texts]; res = []
    for k in ks:
        cvs, npmis, nmis = [], [], []
        for s in seeds:
            m = LdaModel(corpus, id2word=d, num_topics=k, passes=20, iterations=200, alpha='auto', eta='auto', random_state=s)
            cvs.append(CoherenceModel(
    model=m,
    texts=texts,
    dictionary=d,
    coherence='c_v',
    topn=10,
    processes=1
).get_coherence())
            npmis.append(CoherenceModel(
    model=m,
    texts=texts,
    dictionary=d,
    coherence='c_npmi',
    topn=10,
    processes=1
).get_coherence())
            dom = [max(m.get_document_topics(b, minimum_probability=0), key=lambda x: x[1])[0] for b in corpus]
            nmis.append(normalized_mutual_info_score(y, dom))
        res.append({'k': k, 'cv_mean': np.mean(cvs), 'cv_sd': np.std(cvs), 'npmi_mean': np.mean(npmis), 'nmi_mean': np.mean(nmis), 'nmi_sd': np.std(nmis)})
    return pd.DataFrame(res), d, corpus, texts
lda_user, d_u, c_u, t_u = lda_eval('tok_user')
lda_full, _, _, _ = lda_eval('tok_full')
OUT['lda_user'] = lda_user.round(3).to_dict('records'); OUT['lda_full'] = lda_full.round(3).to_dict('records')
# Reproduce original setting (sklearn-style K=5 via gensim default, raw tokens, no stopword removal) for transparency
raw_tokens = df.conversations.str.replace('</s>', ' ').str.lower().str.replace(r'[^\w\s]', '', regex=True).str.split()
d_raw = Dictionary(raw_tokens); c_raw = [d_raw.doc2bow(t) for t in raw_tokens]
m_raw = LdaModel(c_raw, id2word=d_raw, num_topics=5, passes=5, random_state=42)
OUT['lda_original_cv'] = CoherenceModel(
    model=m_raw,
    texts=raw_tokens,
    dictionary=d_raw,
    coherence='c_v',
    processes=1
).get_coherence()# final interpretable models (user text) at K=5 and best K
best_k = int(lda_user.sort_values('cv_mean', ascending=False).iloc[0].k)
OUT['lda_best_k'] = best_k
def topics_for(k, seed=1):
    m = LdaModel(c_u, id2word=d_u, num_topics=k, passes=20, iterations=200, alpha='auto', eta='auto', random_state=seed)
    dom = np.array([max(m.get_document_topics(b, minimum_probability=0), key=lambda x: x[1])[0] for b in c_u])
    tops = []
    for i in range(k):
        words = [w for w, _ in m.show_topic(i, 8)]
        labs = Counter(y[dom == i]).most_common(2); n = int((dom == i).sum())
        tops.append({'topic': i + 1, 'words': ', '.join(words), 'n_docs': n,
                     'top_diseases': '; '.join(f'{a} ({b})' for a, b in labs)})
    return tops
OUT['lda_topics_k5'] = topics_for(5)
OUT['lda_topics_best'] = topics_for(best_k)

# ---------- 4. K-Means ----------
def km_eval(tok_col, ks=(5, 10, 15, 20, 24, 30)):
    docs = df[tok_col].str.join(' ')
    X = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True).fit_transform(docs); res = []
    for k in ks:
        sils, aris, nmis, purs = [], [], [], []
        for s in range(5):
            lab = KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X)
            sils.append(silhouette_score(X, lab, metric='cosine')); aris.append(adjusted_rand_score(y, lab))
            nmis.append(normalized_mutual_info_score(y, lab)); purs.append(purity(y, lab))
        res.append({'k': k, 'sil': np.mean(sils), 'sil_sd': np.std(sils), 'ari': np.mean(aris), 'ari_sd': np.std(aris),
                    'nmi': np.mean(nmis), 'nmi_sd': np.std(nmis), 'purity': np.mean(purs)})
    return pd.DataFrame(res), X
km_user, X_user = km_eval('tok_user'); km_full, X_full = km_eval('tok_full')
OUT['km_user'] = km_user.round(3).to_dict('records'); OUT['km_full'] = km_full.round(3).to_dict('records')
# original configuration reproduction
Xo = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english').fit_transform(
    df.conversations.str.replace('</s>', ' ').str.lower().str.replace(r'[^\w\s]', '', regex=True))
lo = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(Xo)
OUT['km_original'] = {'sil_euclid': silhouette_score(Xo, lo), 'sil_cosine': silhouette_score(Xo, lo, metric='cosine'),
                      'ari': adjusted_rand_score(y, lo), 'nmi': normalized_mutual_info_score(y, lo)}

# ---------- 5. Supervised baseline ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED); sup = {}
for name, col in [('user', 'tok_user'), ('full', 'tok_full')]:
    pipe = make_pipeline(TfidfVectorizer(min_df=1, sublinear_tf=True, ngram_range=(1, 2)), LogisticRegression(max_iter=3000, C=10))
    accs, f1s = [], []
    for tr, te in skf.split(df, y):
        pipe.fit(df[col].str.join(' ').iloc[tr], y[tr]); p = pipe.predict(df[col].str.join(' ').iloc[te])
        accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average='macro'))
    sup[name] = {'acc': np.mean(accs), 'acc_sd': np.std(accs), 'f1': np.mean(f1s), 'f1_sd': np.std(f1s)}
# user text with disease-name mentions masked (tests reliance on explicit naming)
DIS_WORDS = {'allergy', 'allergic', 'asthma', 'malaria', 'impetigo', 'varicose', 'diabetes', 'diabetic', 'psoriasis', 'pneumonia', 'jaundice',
             'migraine', 'uti', 'arthritis', 'ulcer', 'spondylosis', 'cervical', 'chickenpox', 'chicken', 'pox', 'typhoid', 'hemorrhoid', 'dimorphic',
             'hypertension', 'gerd', 'reflux', 'acne', 'fungal', 'cold', 'dengue', 'flu', 'drug', 'reaction', 'bronchial', 'infection', 'urinary', 'tract'}
masked = df.tok_user.apply(lambda t: ' '.join(w for w in t if w not in DIS_WORDS))
pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)), LogisticRegression(max_iter=3000, C=10))
pm = cross_val_predict(pipe, masked, y, cv=skf)
sup['user_masked'] = {'acc': accuracy_score(y, pm), 'f1': f1_score(y, pm, average='macro')}
OUT['supervised'] = sup
from sklearn.metrics import classification_report
rep = classification_report(y, pm, output_dict=True, zero_division=0)
OUT['masked_per_class_f1'] = {k: round(v['f1-score'], 3) for k, v in rep.items() if k in set(y)}

# ---------- 6. Symptom lexicon normalisation + Apriori ----------
LEX = json.load(open('symptom_lexicon.json'))
NEG = re.compile(r"\b(no|not|never|without|dont|don't|haven't|havent|hasn't|didn't|didnt|isn't|nothing)\b")
def extract_symptoms(text):
    t = text.lower().replace('’', "'"); found = set()
    for clause in re.split(r'[.,;!?]| but | and ', t):
        negated = bool(NEG.search(clause))
        for concept, pats in LEX.items():
            if any(re.search(p, clause) for p in pats) and not negated:
                found.add(concept)
    return sorted(found)
df['symptoms'] = df.user_text.apply(extract_symptoms)
cov = (df.symptoms.str.len() > 0).mean()
OUT['lexicon'] = {'n_concepts': len(LEX), 'coverage_pct': round(100 * cov, 1), 'mean_per_dialogue': round(df.symptoms.str.len().mean(), 2),
                  'sd_per_dialogue': round(df.symptoms.str.len().std(), 2), 'n_distinct_found': len(set(sum(df.symptoms, [])))}
freq = Counter(sum(df.symptoms, [])); OUT['top_symptoms'] = freq.most_common(15)
te = TransactionEncoder(); S = pd.DataFrame(te.fit(df.symptoms).transform(df.symptoms), columns=te.columns_)
MINSUP = 0.02
fi = apriori(S, min_support=MINSUP, use_colnames=True, max_len=3)
rules = association_rules(fi, metric='confidence', min_threshold=0.5)
rules = rules[rules.lift > 1.5].copy()
rules['disease_support'] = rules.apply(lambda r: Counter(y[S[list(r.antecedents | r.consequents)].all(axis=1).values]).most_common(3), axis=1)
rules['n_diseases'] = rules.apply(lambda r: len(set(y[S[list(r.antecedents | r.consequents)].all(axis=1).values])), axis=1)
rules = rules.sort_values(['lift', 'confidence'], ascending=False)
OUT['apriori'] = {'min_support': MINSUP, 'min_conf': 0.5, 'min_lift': 1.5, 'n_itemsets': len(fi), 'n_rules': len(rules)}
fmt = lambda s: ' + '.join(sorted(s))
OUT['rules_top'] = [{'ante': fmt(r.antecedents), 'cons': fmt(r.consequents), 'sup': round(r.support, 3), 'conf': round(r.confidence, 3),
                     'lift': round(r.lift, 2), 'n_dis': int(r.n_diseases), 'dis': '; '.join(f'{a} ({b})' for a, b in r.disease_support)}
                    for _, r in rules.head(40).iterrows()]
# fever-headache specifically
fh = S.get('fever', pd.Series(False, index=S.index)) & S.get('headache', pd.Series(False, index=S.index))
OUT['fever_headache'] = {'support': round(fh.mean(), 3), 'conf_fever_to_headache': round(fh.sum() / S['fever'].sum(), 3),
                         'conf_headache_to_fever': round(fh.sum() / S['headache'].sum(), 3),
                         'lift': round(fh.mean() / (S['fever'].mean() * S['headache'].mean()), 2), 'n': int(fh.sum()),
                         'diseases': Counter(y[fh.values]).most_common()}
# cross-disease symptoms: number of diseases in which each symptom appears in >=10% of dialogues
prof = S.groupby(y).mean()
OUT['cross_disease'] = sorted([(c, int((prof[c] >= 0.10).sum())) for c in prof.columns], key=lambda x: -x[1])[:12]
df[['disease', 'user_text', 'symptoms']].to_csv('symptoms_extracted.csv', index=False)
# sample for manual validation
#df.sample(100, random_state=SEED)[['disease', 'user_text', 'symptoms']].to_csv('validation_sample_100.csv', index=False)

json.dump(OUT, open('results.json', 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o))
lda_user.to_csv('lda_user.csv', index=False); km_user.to_csv('km_user.csv', index=False); km_full.to_csv('km_full.csv', index=False); lda_full.to_csv('lda_full.csv', index=False)
print(json.dumps(OUT, indent=1, default=str)[:12000])
