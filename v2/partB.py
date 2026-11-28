"""Revision analyses, part B: extractor v2, speaker attribution, ablations, itemset-level rules with
significance testing, bootstrap lift CIs, disease-stratified association, threshold sensitivity."""
import json, warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from collections import Counter
from scipy.stats import fisher_exact, norm
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import extract as E
df = pd.read_csv('medical_conversations.csv')[['conversations', 'disease']].dropna().reset_index(drop=True)
y = df.disease.values; N = len(df); OUT = {}
ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]

# ---- 1. Extraction: v2 and ablations
def run(neg=None, ell=True):
    r = df.conversations.apply(lambda c: E.extract(c, negation=neg, elliptical=ell)); return r.apply(lambda t: t[0]), r.apply(lambda t: t[1])
rep, first = run()
rep_noneg, _ = run(neg=False); rep_noell, _ = run(ell=False)
v1 = pd.read_csv('../symptoms_extracted.csv').symptoms.apply(eval).apply(set)
sets = {'v2': rep.apply(set), 'v2_no_negation': rep_noneg.apply(set), 'v2_no_elliptical': rep_noell.apply(set), 'v1': v1}
def summ(s): n = s.apply(len); return {'coverage_pct': float(100 * (n > 0).mean()), 'mean': float(n.mean()), 'sd': float(n.std()), 'total_mentions': int(n.sum()), 'n_dialogues_none': int((n == 0).sum())}
OUT['extraction'] = {k: summ(s) for k, s in sets.items()}
OUT['v1_v2_diff'] = {'added_by_v2': int(sum(len(a - b) for a, b in zip(sets['v2'], v1))), 'removed_by_v2': int(sum(len(b - a) for a, b in zip(sets['v2'], v1)))}
OUT['negation_effect'] = {'concepts_removed_by_negation': int(sum(len(a - b) for a, b in zip(sets['v2_no_negation'], sets['v2']))),
                          'dialogues_affected': int(sum(a != b for a, b in zip(sets['v2_no_negation'], sets['v2'])))}
prov = Counter(p for r in rep for p in r.values()); OUT['provenance'] = dict(prov)
# none-dialogues by disease (v2)
OUT['none_by_disease_v2'] = Counter(y[(sets['v2'].apply(len) == 0).values]).most_common(8)

# ---- 2. Speaker attribution: who first mentioned each patient-reported symptom; bot-only mentions
att = Counter(); per_d = {}
for r, f, d in zip(rep, first, y):
    for c in r:
        k = 'patient_initiated' if f.get(c) == 'patient' else 'bot_prompted'
        att[k] += 1; per_d.setdefault(d, Counter())[k] += 1
    for c, who in f.items():
        if c not in r and who == 'bot': att['bot_only_not_reported'] += 1
OUT['attribution'] = dict(att)
tot = att['patient_initiated'] + att['bot_prompted']
OUT['attribution_share'] = {'patient_initiated_pct': 100 * att['patient_initiated'] / tot, 'bot_prompted_pct': 100 * att['bot_prompted'] / tot}
OUT['attribution_by_disease'] = {d: round(100 * c['patient_initiated'] / max(1, sum(c.values())), 1) for d, c in sorted(per_d.items())}
# share of ALL symptom concepts appearing anywhere in dialogue that the patient reports
allc = sum(len(f) for f in first); OUT['concepts_anywhere'] = int(allc); OUT['concepts_patient_reported'] = int(sum(len(r) for r in rep))

# ---- 3. Prevalence spread
te = TransactionEncoder(); S = pd.DataFrame(te.fit(sets['v2']).transform(sets['v2']), columns=te.columns_)
prof = S.groupby(y).mean()
OUT['freq_top'] = S.sum().sort_values(ascending=False).head(15).astype(int).to_dict()
OUT['spread_ge10pct'] = sorted([(c, int((prof[c] >= .10).sum())) for c in prof.columns], key=lambda x: -x[1])[:10]

# ---- 4. Rules at itemset level + significance
def mine(S, minsup=0.02, minconf=0.5, minlift=1.5, maxlen=3):
    fi = apriori(S, min_support=minsup, use_colnames=True, max_len=maxlen)
    if fi.empty: return fi, pd.DataFrame()
    ru = association_rules(fi, metric='confidence', min_threshold=0.0)
    return fi, ru
def disease_profile(items):
    m = S[list(items)].all(axis=1).values; c = Counter(y[m]); n = m.sum()
    return n, c
def mh_or(a_col, c_col):
    """Mantel-Haenszel odds ratio of association between antecedent and consequent, stratified by disease (RGB CI)."""
    num = den = 0.0; P = Q = R_ = Ss = PR = PS_QR = QS = 0.0; Rs = Ss_ = 0.0; terms = []
    for d in np.unique(y):
        m = (y == d); A = a_col[m]; Cc = c_col[m]; n = m.sum()
        a = np.sum(A & Cc); b = np.sum(A & ~Cc); c = np.sum(~A & Cc); dd = np.sum(~A & ~Cc)
        if n == 0: continue
        R = a * dd / n; Sx = b * c / n; Pi = (a + dd) / n; Qi = (b + c) / n
        terms.append((R, Sx, Pi, Qi))
    Rsum = sum(t[0] for t in terms); Ssum = sum(t[1] for t in terms)
    if Rsum == 0 or Ssum == 0: return {'or_mh': float('nan'), 'ci': [float('nan')] * 2}
    orr = Rsum / Ssum
    v = (sum(t[2] * t[0] for t in terms) / (2 * Rsum ** 2) + sum(t[2] * t[1] + t[3] * t[0] for t in terms) / (2 * Rsum * Ssum) + sum(t[3] * t[1] for t in terms) / (2 * Ssum ** 2))
    se = np.sqrt(v); return {'or_mh': float(orr), 'ci': [float(np.exp(np.log(orr) - 1.96 * se)), float(np.exp(np.log(orr) + 1.96 * se))]}
fi, ru = mine(S)
cand = ru.copy(); m_tested = len(cand)            # every rule generated from frequent itemsets is a tested hypothesis
def fisher_row(r):
    A = S[list(r.antecedents)].all(axis=1).values; Cc = S[list(r.consequents)].all(axis=1).values
    a = np.sum(A & Cc); b = np.sum(A & ~Cc); c = np.sum(~A & Cc); d = np.sum(~A & ~Cc)
    return fisher_exact([[a, b], [c, d]], alternative='greater')[1]
cand['p'] = cand.apply(fisher_row, axis=1)
cand = cand.sort_values('p'); mtot = len(cand)
cand['p_bonf'] = np.minimum(cand.p * mtot, 1)
rk = np.arange(1, mtot + 1); bh = cand.p.values * mtot / rk; cand['q_bh'] = np.minimum.accumulate(bh[::-1])[::-1].clip(max=1)
sel = cand[(cand.confidence >= 0.5) & (cand.lift > 1.5)].copy()
sel['itemset'] = sel.apply(lambda r: frozenset(r.antecedents | r.consequents), axis=1)
items = sel.groupby('itemset').agg(lift=('lift', 'max'), conf=('confidence', 'max'), support=('support', 'first'), p_bonf=('p_bonf', 'min'), q_bh=('q_bh', 'min'), n_rules=('lift', 'size')).reset_index()
prof_rows = []
for _, r in items.iterrows():
    n, c = disease_profile(r.itemset); top = c.most_common(1)[0][1] / n
    prof_rows.append({'n_dis': len(c), 'dom_share': top})
items = pd.concat([items, pd.DataFrame(prof_rows)], axis=1)
# disease-stratified association for each itemset (best rule): MH OR
def best_rule(iset):
    rr = sel[sel.itemset == iset].sort_values('lift', ascending=False).iloc[0]; return rr
strat = []
for _, r in items.iterrows():
    br = best_rule(r.itemset); A = S[list(br.antecedents)].all(axis=1).values; Cc = S[list(br.consequents)].all(axis=1).values
    strat.append(mh_or(A, Cc))
items['or_mh'] = [s['or_mh'] for s in strat]; items['or_mh_ci'] = [s['ci'] for s in strat]
OUT['rules'] = {'n_frequent_itemsets': int(len(fi)), 'n_rules_tested': int(mtot), 'n_rules_selected': int(len(sel)), 'n_itemsets_selected': int(len(items)),
                'itemsets_single_disease': int((items.n_dis == 1).sum()), 'itemsets_dom80': int((items.dom_share >= .8).sum()),
                'itemsets_sig_bonf': int((items.p_bonf < .05).sum()), 'itemsets_sig_bh': int((items.q_bh < .05).sum()),
                'itemsets_mh_ci_above1': int(sum(1 for c in items.or_mh_ci if c[0] > 1)), 'itemsets_mh_undefined': int(items.or_mh.isna().sum())}
# null reference: single-disease share among ALL frequent itemsets of size>=2 (no lift filter)
fi2 = fi[fi.itemsets.apply(len) >= 2]
OUT['rules']['all_freq_itemsets_ge2'] = int(len(fi2)); OUT['rules']['all_freq_single_disease'] = int(sum(len(disease_profile(s)[1]) == 1 for s in fi2.itemsets))
# bootstrap lift CIs for the selected itemsets (best rule)
B = 1000; rs = np.random.RandomState(0); idxs = [rs.choice(N, N) for _ in range(B)]
Sv = S.values.astype(bool); col = {c: i for i, c in enumerate(S.columns)}
def lift_boot(ante, cons):
    a_ = [col[c] for c in ante]; c_ = [col[c] for c in cons]; out = []
    for ix in idxs:
        X = Sv[ix]; A = X[:, a_].all(1); Cc = X[:, c_].all(1)
        pa, pc, pac = A.mean(), Cc.mean(), (A & Cc).mean(); out.append(pac / (pa * pc) if pa * pc > 0 else np.nan)
    return ci(out)
fmt = lambda s: ' + '.join(sorted(x.replace('_', ' ') for x in s))
table = []
for _, r in items.sort_values('lift', ascending=False).iterrows():
    br = best_rule(r.itemset); n, c = disease_profile(r.itemset)
    table.append({'rule': f'{fmt(br.antecedents)} => {fmt(br.consequents)}', 'support': round(br.support, 3), 'conf': round(br.confidence, 2), 'lift': round(br.lift, 2),
                  'lift_ci': [round(x, 2) for x in lift_boot(br.antecedents, br.consequents)], 'p_bonf': float(br.p_bonf), 'q_bh': float(br.q_bh),
                  'or_mh': None if np.isnan(r.or_mh) else round(r.or_mh, 2), 'or_mh_ci': [None if np.isnan(x) else round(x, 2) for x in r.or_mh_ci],
                  'labels': '; '.join(f'{a} ({b})' for a, b in c.most_common(3)), 'n_dis': len(c)})
OUT['itemset_table'] = table

# ---- 5. Fever-headache
f, h = S['fever'].values, S['headache'].values
obs = int((f & h).sum()); exp = f.sum() * h.sum() / N
fl = []; 
for ix in idxs:
    ff, hh = f[ix], h[ix]; fl.append((ff & hh).mean() / (ff.mean() * hh.mean()))
OUT['fever_headache'] = {'n_fever': int(f.sum()), 'n_headache': int(h.sum()), 'observed': obs, 'expected': float(exp), 'lift': float(obs / exp), 'lift_ci': ci(fl),
                         'fisher_two_sided_p': float(fisher_exact([[obs, f.sum() - obs], [h.sum() - obs, N - f.sum() - h.sum() + obs]])[1]),
                         'mh_or': mh_or(f.astype(bool), h.astype(bool)), 'diseases': Counter(y[(f & h).astype(bool)]).most_common()}

# ---- 6. Threshold sensitivity (itemset level)
grid = []
for ms in [0.01, 0.015, 0.02, 0.03]:
    fi_, ru_ = mine(S, minsup=ms)
    for mc in [0.3, 0.5, 0.7]:
        for ml in [1.2, 1.5, 2.0]:
            s_ = ru_[(ru_.confidence >= mc) & (ru_.lift > ml)]
            its = set(frozenset(a | b) for a, b in zip(s_.antecedents, s_.consequents))
            single = sum(len(disease_profile(i)[1]) == 1 for i in its); dom = sum(max(disease_profile(i)[1].values()) / disease_profile(i)[0] >= .8 for i in its)
            grid.append({'minsup': ms, 'minconf': mc, 'minlift': ml, 'n_itemsets': len(its), 'single_pct': round(100 * single / len(its), 1) if its else None, 'dom80_pct': round(100 * dom / len(its), 1) if its else None})
OUT['threshold_grid'] = grid

# ---- 7. Same rule analysis under ablations (negation off / elliptical off / v1)
abl = {}
for k in ['v2_no_negation', 'v2_no_elliptical', 'v1']:
    S_ = pd.DataFrame(TransactionEncoder().fit(sets[k]).transform(sets[k]), columns=TransactionEncoder().fit(sets[k]).columns_)
    fi_, ru_ = mine(S_); s_ = ru_[(ru_.confidence >= .5) & (ru_.lift > 1.5)]
    its = set(frozenset(a | b) for a, b in zip(s_.antecedents, s_.consequents))
    abl[k] = {'n_itemsets': len(its), 'only_here': len(its - set(items.itemset)), 'missing_here': len(set(items.itemset) - its)}
OUT['ablation_rules'] = abl
# yellow rule check
OUT['yellow_impetigo_rule_present'] = any(('yellow_skin_eyes' in i and 'blisters' in i) for i in items.itemset)
OUT['yellow_by_disease'] = Counter(y[S['yellow_skin_eyes'].values]).most_common() if 'yellow_skin_eyes' in S else []
pd.DataFrame({'disease': y, 'symptoms_v2': [sorted(s) for s in sets['v2']], 'provenance': [dict(r) for r in rep]}).to_csv('symptoms_extracted_v2.csv', index=False)
json.dump(OUT, open('results_partB.json', 'w'), indent=1, default=lambda o: list(o) if isinstance(o, (set, frozenset)) else float(o))
print(json.dumps({k: OUT[k] for k in ['extraction', 'v1_v2_diff', 'negation_effect', 'provenance', 'attribution', 'attribution_share', 'rules', 'fever_headache', 'ablation_rules', 'yellow_impetigo_rule_present', 'spread_ge10pct']}, indent=1, default=str))
