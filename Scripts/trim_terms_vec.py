import pandas as pd
import re
from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import csv
from pathlib import Path

DATA_DIR = Path("C:/Users/sirda/Dropbox (Personal)/Documents/Fall '25/Data Mining/PubMed Project/Data")
CO_FILE = DATA_DIR / "co-mentions.csv"
PAPERS_FILE = DATA_DIR / "papers.csv"
AGG_FILE = DATA_DIR / "aggregated.csv"
TERMS_FILE = DATA_DIR / "terms.csv"

terms_df = pd.read_csv(TERMS_FILE)

# Some entries are multi-line aliases; split them
def extract_aliases(entry):
    if pd.isna(entry):
        return []
    return re.split(r'[\n|]', entry)

terms_df["alias_list"] = terms_df["entries"].apply(extract_aliases)
terms = [t.strip() for sublist in terms_df["alias_list"].dropna() for t in sublist if t.strip()]
terms = list(set(terms))


model = KeyedVectors.load_word2vec_format("BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True)
embeddings = {t: model[t] for t in terms if t in model}


X = np.vstack(list(embeddings.values()))
labels = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.4,
    affinity='cosine',
    linkage='average'
).fit_predict(X)

# Map back to terms
cluster_map = {}
for label, term in zip(labels, embeddings.keys()):
    cluster_map.setdefault(label, []).append(term)

canonical = {}
for cluster_id, cluster_terms in cluster_map.items():
    rep = min(cluster_terms, key=len)  # simplest heuristic
    for t in cluster_terms:
        canonical[t] = rep

co_mentions = pd.read_csv(AGG_FILE)
co_mentions["drug_name_canonical"] = co_mentions["drug_name"].map(canonical).fillna(co_mentions["drug_name"])
co_mentions["disease_name_canonical"] = co_mentions["disease_name"].map(canonical).fillna(co_mentions["disease_name"])
