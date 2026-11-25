# Revision v2 (IEEE Access resubmission)

Scripts in this folder reproduce every number in the revised manuscript. They reuse the preprocessing in `../analysis.py` (run from the repository root, with `medical_conversations.csv` in the root).

| Script | Output | Purpose |
|---|---|---|
| `common.py` | — | Shared preprocessing; builds the seven text views (patient, bot, full; each with and without the 37 masked disease tokens; length-matched masked full view) |
| `extract.py` + `symptom_lexicon_v2.json` | — | Frozen symptom extractor: 52 concepts, ConText-style negation, elliptical-reply resolution, speaker of first mention. SHA-256 hashes in `extractor_freeze.txt` |
| `partA.py` | `results/results_partA.json` | K-means sweep and label-free K, NMF/LSA/GloVe baselines, LDA on all views, nested logistic regression, McNemar tests, preprocessing sensitivity |
| `partB.py` | `results/results_partB.json`, `results/symptoms_extracted_v2.csv` | Extraction and ablations, attribution, itemset rules with Fisher/Bonferroni/BH, bootstrap lift CIs, Mantel–Haenszel stratification, threshold grid |
| `partC.py` | `results/results_partC.json` | LDA K sweep with C_V, NPMI, AMI |
| `partD.py` | `results/results_partD.json` | Bootstrap CIs over dialogues and paired view differences |
| `partE.py` | updates partA | Logistic regression on the 52 symptom concepts with the frozen extractor |
| `make_annot.py` | annotation workbooks | Draws the 120-dialogue validation sample and the 40 no-concept dialogues (seed 2026) |
| `run_ner_baseline.py` | `ner_predictions.csv` | Transformer NER baseline (d4data/biomedical-ner-all) |
| `fig_*.py` | `figures/` | Figures 1–4 |

Run from the repository root, e.g. `python v2/partA.py`; outputs are written to the working directory. Order: `partA.py`, `partC.py`, `partB.py`, `partD.py`, `partE.py`, then the figure scripts.

Environment: Python 3.13, pandas 3.0, NLTK 3.10.4, gensim 4.4.0, scikit-learn 1.9.1, mlxtend 0.25.0, SciPy 1.18.1. GloVe vectors: glove-wiki-gigaword-100 (gensim-data).
