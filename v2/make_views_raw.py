"""Builds views_raw.csv (untokenized patient/bot/full text, with and without disease names) for run_embeddings.py."""
import re, pandas as pd
import common as C
pat = re.compile(r"\b(" + "|".join(sorted(C.DIS_WORDS, key=len, reverse=True)) + r")(s|es)?\b", re.I)
m = lambda t: re.sub(r'\s+', ' ', pat.sub(' ', t)).strip()
df = C.df
out = pd.DataFrame({'dialogue_id': ['D%03d' % i for i in range(len(df))], 'patient': df.user_text, 'bot': df.bot_text})
out['full'] = df.full_text if 'full_text' in df else (df.user_text + ' ' + df.bot_text)
for v in ['patient', 'bot', 'full']: out[v + '_masked'] = out[v].map(m)
out.to_csv('views_raw.csv', index=False)
