# Revision v2 (IEEE Access resubmission)

Everything needed to reproduce the revised manuscript. **Run the scripts inside this folder** after copying `../analysis.py` and `../medical_conversations.csv` into it (`common.py` reuses the preprocessing in `analysis.py`; `extract.py` reads `symptom_lexicon_v2.json` from the working directory).

## Frozen extractor
`extract.py` + `symptom_lexicon_v2.json`: 52 concepts, ConText-style negation, elliptical-reply resolution, speaker of first mention. Frozen before manual annotation; SHA-256 hashes in `extractor_freeze.txt` (`sha256sum extract.py symptom_lexicon_v2.json` to check).

## Scripts (run in this order)
| Script | Output (`results/`) | Purpose |
|---|---|---|
| `partA.py` | `results_partA.json` | Seven views; K-means sweep and label-free K; NMF/LSA/GloVe baselines (needs `emb/glove100.gz`, glove-wiki-gigaword-100 from gensim-data); LDA on all views; nested logistic regression; McNemar tests; preprocessing sensitivity |
| `partC.py` | `results_partC.json` | LDA K sweep (C_V, NPMI, AMI) |
| `partB.py` | `results_partB.json`, `symptoms_extracted_v2.csv` | Extraction and ablations; speaker attribution; itemset rules with Fisher/Bonferroni/BH, bootstrap lift CIs, Mantel–Haenszel stratification, threshold grid |
| `partD.py` | `results_partD.json` | Bootstrap CIs over dialogues; paired view differences |
| `partE.py` | updates `results_partA.json` | Logistic regression on the 52 symptom concepts |
| `make_views_raw.py` → `run_embeddings.py` | `views_raw.csv` → `embeddings.npz` | Untokenized views; MiniLM and MPNet sentence embeddings (needs internet for the models) |
| `partF.py`, `partF2.py` | `results_partF.json`, `results_partF2.json` | Embedding K-means and BERTopic; bootstrap CIs for embedding views |
| `partG.py` | `results_partG.json` | Second corpus: CovidDialog-English (download `english-train.json`, `english-dev.json`, `english-test.json` from the CovidDialog release into `medd/`) |
| `run_ner_baseline.py` | `annotation/ner_predictions.csv` | Transformer NER baseline (d4data/biomedical-ner-all) on Sample A |
| `fig_*.py` | `figures/` | Figures 1–4 |

## Manual validation (`annotation/`)
- `ANNOTATION_GUIDELINES.md`: instructions and concept definitions given to both annotators.
- `labels_sampleA.csv`: 120 synthetic dialogues (5 per disease); each annotator's concepts, out-of-lexicon symptoms and comments, and the adjudicated labels with the reason for each adjudication. Concept names are normalized (lower case, underscores, sorted); one data-entry typo in annotator B's file (`frequent_urinationreduced_alertness`) was read as `frequent_urination`.
- `labels_sampleB.csv`: 40 synthetic dialogues without any concept from the preprint extractor; codes S/E/M/O.
- `labels_sampleC_coviddialog.csv`: 40 CovidDialog patient messages without an extracted symptom; codes Q/D/V/S. `coviddialog_index` refers to the concatenation of the train, dev, and test files after removing the one non-English dialogue (see `partG.py`); `sampleC_ids.json` lists the same indices. **CovidDialog text is not redistributed here.**
- `key_sampleA.csv`, `key_sampleB.csv`: dialogue ID → disease label.
- `reproduce_metrics.py`: recomputes Cohen's kappa (0.994) and the extractor's precision/recall/F1 (0.882/0.915/0.898) from `labels_sampleA.csv`.
- `results/annot_results_raw.json`, `results/annot_results.json`: all validation numbers reported in the paper.

The synthetic dialogues (`../medical_conversations.csv`) are from the MedicalConversations2Disease dataset (Apache 2.0). The CovidDialog corpus is available from its original release under its own license.
