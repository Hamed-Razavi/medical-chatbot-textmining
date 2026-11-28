import re, numpy as np, pandas as pd
src = open('analysis.py').read()
_ns = {}
exec(src[src.index('import re'):src.index('# ---------- 3. LDA')], _ns)   # reuse exact v1 preprocessing (dataset stats, prep, STOP)
df = _ns['df']; prep = _ns['prep']; STOP = _ns['STOP']; SW = _ns['SW']; FILLER = _ns['FILLER']; lem = _ns['lem']
s = src.index('DIS_WORDS = {'); DIS_WORDS = eval(src[s + 12:src.index('}', s) + 1])
s = src.index('keys = {'); KEYS = eval(src[s + 7:src.index('}', s) + 1])
y = df.disease.values
df['tok_bot'] = df.bot_text.apply(prep)
mask = lambda toks: [w for w in toks if w not in DIS_WORDS]
df['tok_user_m'] = df.tok_user.apply(mask); df['tok_full_m'] = df.tok_full.apply(mask); df['tok_bot_m'] = df.tok_bot.apply(mask)
rng = np.random.RandomState(7)
def length_match(full, user):
    k = min(len(user), len(full))
    return list(rng.choice(full, size=k, replace=False)) if k else []
df['tok_full_lm'] = [length_match(f, u) for f, u in zip(df.tok_full_m, df.tok_user_m)]   # masked full view, length-matched to patient view
VIEWS = {'patient': 'tok_user', 'full': 'tok_full', 'bot': 'tok_bot', 'patient_masked': 'tok_user_m',
         'full_masked': 'tok_full_m', 'bot_masked': 'tok_bot_m', 'full_masked_lenmatched': 'tok_full_lm'}
