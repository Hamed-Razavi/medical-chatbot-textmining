# Reproduction

pip install pandas nltk gensim scikit-learn mlxtend matplotlib
python -c "import nltk; \[nltk.download(p) for p in \['stopwords','wordnet']]"
python analysis.py        # all LDA / K-Means / classifier / lexicon / Apriori results -> results.json
python figs.py; python pipe.py   # Figures 1-3

Place medical\_conversations.csv in this folder first.

