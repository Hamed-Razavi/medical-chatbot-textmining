"""Second corpus: CovidDialog-English (train+dev+test). Label-independent measures, applied identically to both corpora."""
import json, re, sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import common as C, extract as E
D = sum((json.load(open(f'medd/english-{s}.json')) for s in ['train', 'dev', 'test']), [])
SPANISH = re.compile(r'[¿¡]|\b(tengo|pero|est[aá]n?|qu[eé]|para|una)\b')
D = [d for d in D if not SPANISH.search(' '.join(d['utterances']))]
COND = (r"\b(covid[- ]?19|covid|coronavirus|corona ?virus|corona|pneumonia|influenza|flu|common cold|head cold|chest cold|a cold|strep( throat)?|"
        r"sinusitis|bronchitis|allerg(y|ies|ic rhinitis)|hay fever|asthma|cellulitis|abscess|embolism|pharyngitis|tonsillitis|thrush|"
        r"reflux|gerd|heartburn|urti|sepsis|tuberculosis|tb|copd|empyema|hpv|anxiety|depression|diabetes|arthritis|heart failure|"
        r"(viral|bacterial|sinus|chest|lung|ear|throat|respiratory|tract|systemic|fungal|skin|urinary) infections?|"
        r"acne|dengue|malaria|typhoid|jaundice|hepatitis|migraine|psoriasis|impetigo|chicken ?pox|varicella|hypertension|high blood pressure|"
        r"ulcer|hemorrhoids?|piles|varicose veins?|spondylosis|uti|drug reaction|eczema|dermatitis|gastroenteritis|food poisoning)\b")
def norm(t): return t.replace('’', "'").lower()
def conds(t): return {m.group(0) for m in re.finditer(COND, norm(t))}
def measures(dialogs):
    rows = []
    for turns in dialogs:
        P = ' '.join(t for s, t in turns if s == 'P'); R = ' '.join(t for s, t in turns if s == 'B')
        cp, cr = conds(P), conds(R)
        conv = ' </s> '.join(('User: ' if s == 'P' else 'Bot: ') + t for s, t in turns)
        rep, first = E.extract(conv.replace('’', "'"))
        rows.append(dict(n_turns=len(turns), pat_words=len(P.split()), resp_words=len(R.split()), resp_names=bool(cr), pat_names=bool(cp), resp_new=bool(cr - cp),
                         any_concept=bool(rep), n_concepts=len(rep), resp_first=sum(1 for c in rep if first.get(c) == 'bot'), elliptical=sum(1 for v in rep.values() if v == 'confirmed')))
    return pd.DataFrame(rows)
cd = measures([[('P' if u.startswith('patient:') else 'B', u.split(':', 1)[1].strip()) for u in d['utterances']] for d in D])
sy = measures([E.turns_of(c) for c in C.df.conversations])
def summ(x):
    return {'n': len(x), 'multi_exchange': int((x.n_turns > 2).sum()), 'turns_mean': round(x.n_turns.mean(), 2), 'pat_words_median': float(x.pat_words.median()), 'resp_words_median': float(x.resp_words.median()),
            'resp_names_pct': round(100 * x.resp_names.mean(), 1), 'pat_names_pct': round(100 * x.pat_names.mean(), 1), 'resp_new_pct': round(100 * x.resp_new.mean(), 1),
            'coverage_pct': round(100 * x.any_concept.mean(), 1), 'concepts_mean': round(x.n_concepts.mean(), 2), 'concepts_total': int(x.n_concepts.sum()),
            'resp_first_pct': round(100 * x.resp_first.sum() / max(1, x.n_concepts.sum()), 1)}
res = {'coviddialog': summ(cd), 'synthetic': summ(sy), 'cond_regex': COND}
# 2x2 tests (Fisher) for responder naming and coverage
from scipy.stats import fisher_exact
for k in ['resp_names', 'pat_names', 'any_concept']:
    t = [[cd[k].sum(), (~cd[k]).sum()], [sy[k].sum(), (~sy[k]).sum()]]
    res[f'fisher_{k}'] = fisher_exact(t)[1]
# frequent out-of-lexicon symptom words among CovidDialog patient texts without any concept
nocon = [norm(' '.join(u.split(':', 1)[1] for u in d['utterances'] if u.startswith('patient:'))) for d, a in zip(D, cd.any_concept) if not a]
res['no_concept_n'] = len(nocon)
probe = {'loss of smell/taste': r'\b(smell|taste)\b', 'chest tightness/heaviness': r'\b(tight|heavy|heaviness|tightness)\b.{0,20}\bchest|\bchest\b.{0,20}\b(tight|heavy|tightness)', 'body aches/sore body': r'\b(body (ache|aches|pain|pains)|sore body|body (feels|is) sore)\b', 'swollen glands': r'\bswollen glands?\b', 'post-nasal drip': r'\b(post ?nasal drip|nasal drip)\b'}
allpat = [norm(' '.join(u.split(':', 1)[1] for u in d['utterances'] if u.startswith('patient:'))) for d in D]
res['probe'] = {k: int(sum(bool(re.search(p, t)) for t in allpat)) for k, p in probe.items()}
json.dump(res, open('results_partG.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != 'cond_regex'}, indent=1, default=float))
