"""Symptom extractor v2 (frozen before manual annotation).
Changes from v1: ConText-style negation scope (forward, backward and pseudo triggers; scope ends at
termination terms or a new clause), resolution of elliptical replies to chatbot questions, speaker
attribution of first mention, and three lexicon fixes (hives, wheezy, context-restricted 'yellow')."""
import re, json
LEX = json.load(open('symptom_lexicon_v2.json'))
PATS = {c: [re.compile(p) for p in ps] for c, ps in LEX.items()}
FORWARD = [r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bwithout\b", r"\bden(y|ies|ied)\b", r"\bneither\b", r"\bnor\b",
           r"\bnone\b", r"\bhaven'?t\b", r"\bhasn'?t\b", r"\bhadn'?t\b", r"\bdon'?t\b", r"\bdoesn'?t\b", r"\bdidn'?t\b",
           r"\bisn'?t\b", r"\baren'?t\b", r"\bwasn'?t\b", r"\bweren'?t\b", r"\bwon'?t\b", r"\bcan'?t\b", r"\bcannot\b",
           r"\bfree of\b", r"\babsence of\b", r"\bno signs? of\b", r"\bnegative for\b"]
PSEUDO = [r"\bnot only\b", r"\bnot sure\b", r"\bno idea\b", r"\bnot (getting|feeling) (any )?better\b", r"\bnot improving\b",
          r"\bno improvement\b", r"\b(doesn'?t|didn'?t|don'?t|isn'?t) (seem to )?help(ing)?\b", r"\bdon'?t know\b",
          r"\bno matter\b", r"\bcan'?t (seem to )?(breathe|sleep|stop|catch)\b", r"\bcannot (breathe|sleep)\b",
          r"\bnot (hungry|eating much)\b", r"\bno appetite\b", r"\bcan'?t (shake|get rid of)\b", r"\bwon'?t (go away|stop)\b",
          r"\bdoesn'?t (go away|stop)\b", r"\bnot go(ing)? away\b"]
BACKWARD = [r"\b(is|has|have|are) (now )?gone\b", r"\bwent away\b", r"\bresolved\b", r"\bcleared up\b", r"\bdisappeared\b"]
TERMINATE = r"\b(but|however|although|though|except|yet|aside from|apart from|which|whereas)\b|[.;!?]|\band (i|my|it|the|there|then|now|also|sometimes|lately|recently|i'?ve|i'?m)\b"
FAMILY_HISTORY = r"\b(runs in (my|the) family|family history|my (mother|mom|father|dad|brother|sister|grandmother|grandfather|parents) (has|had|have|also has|also had))\b"
AFFIRM = re.compile(r"^(yes|yeah|yep|yup|sure|definitely|absolutely|i do|i have|i am|i'?ve (had|noticed|been)|it does|it is|they do|they are|a little|a bit|sometimes|occasionally|actually,? yes|now that you mention it)\b")
NEGREPLY = re.compile(r"^(no|nope|not really|not at all|i don'?t|i haven'?t|i'?m not|it doesn'?t|none)\b")
NEGATION_ON = True

def norm(t):
    return t.lower().replace('’', "'").replace('‘', "'")

def clauses(text):
    """Split into clause segments at termination terms; ConText-style scope = one segment."""
    t = norm(text); spans = []; last = 0
    for m in re.finditer(TERMINATE, t):
        spans.append(t[last:m.start()]); last = m.start() if m.group(0).startswith('and') else m.end()
    spans.append(t[last:]); return [s for s in spans if s.strip()]

def find(text, negation=None):
    """Return (affirmed, negated) concept sets for one turn."""
    neg_on = NEGATION_ON if negation is None else negation
    aff, neg = set(), set()
    for cl in clauses(text):
        if re.search(FAMILY_HISTORY, cl):
            continue
        masked = cl
        for p in PSEUDO:
            masked = re.sub(p, lambda m: '_' * len(m.group(0)), masked)
        ftrig = [m.start() for p in FORWARD for m in re.finditer(p, masked)]
        btrig = [m.start() for p in BACKWARD for m in re.finditer(p, masked)]
        for c, ps in PATS.items():
            for p in ps:
                for m in p.finditer(cl):
                    inside = lambda pos: m.start() <= pos < m.end()
                    is_neg = neg_on and (any(t < m.start() and not inside(t) for t in ftrig) or any(t >= m.end() for t in btrig))
                    (neg if is_neg else aff).add(c)
    return aff, neg

def turns_of(conv):
    out = []
    for t in conv.split('</s>'):
        t = t.strip()
        if not t: continue
        if t.startswith('User:'): out.append(('P', t[5:].strip()))
        elif t.startswith('Bot:'): out.append(('B', t[4:].strip()))
        else: out.append(('?', t))
    return out

def extract(conv, negation=None, elliptical=True):
    """Patient-reported concepts with provenance, and speaker of first mention for every concept."""
    tr = turns_of(conv); reported = {}; first = {}; prev_bot = ''
    for spk, txt in tr:
        aff, neg = find(txt, negation)
        if spk == 'B':
            for c in aff | neg:
                first.setdefault(c, 'bot')
            prev_bot = txt; continue
        stated = aff
        for c in stated:
            reported.setdefault(c, 'stated'); first.setdefault(c, 'patient')
        if elliptical and prev_bot and '?' in prev_bot and AFFIRM.search(norm(txt)) and not NEGREPLY.search(norm(txt)):
            qs = ' '.join(s for s in re.split(r'(?<=[?.!])\s+', prev_bot) if s.strip().endswith('?'))
            qaff, _ = find(qs, negation=False)
            chosen = (qaff & stated) if (qaff & stated) else qaff
            for c in chosen - neg:
                reported.setdefault(c, 'confirmed')
        for c in neg:
            first.setdefault(c, 'patient')
        prev_bot = ''
    return reported, first
