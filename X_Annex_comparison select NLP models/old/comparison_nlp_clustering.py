#############################################
# CLUSTERING + DEDUPLICATION (LABEL = 1 ONLY)
#############################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer, AutoModel
import torch

#############################################
# 📥 LOAD DATA
#############################################

df = pd.read_excel("data/annotated_final.xlsx")

# ✅ KEEP ONLY MISSING PERSON CASES
df_cases = df[df["label"] == 1].copy()

texts = df_cases["text_clean"].astype(str)

#############################################
# 🧠 CLASSIC ML (TF-IDF + DBSCAN)
#############################################

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(texts)

db = DBSCAN(eps=0.6, min_samples=3, metric="cosine")
clusters_cl = db.fit_predict(X)

cl_score = (
    silhouette_score(X, clusters_cl)
    if len(set(clusters_cl)) > 1 else 0
)

#############################################
# 🤖 TRANSFORMER EMBEDDINGS
#############################################

model_name = "aubmindlab/bert-base-arabert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def embed(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        out = model(**inputs).last_hidden_state.mean(dim=1)

    return out.cpu().numpy()

embeddings = np.vstack([embed(t) for t in texts])

kmeans = KMeans(n_clusters=5, random_state=42)
clusters_tr = kmeans.fit_predict(embeddings)

tr_score = silhouette_score(embeddings, clusters_tr)

#############################################
# 🧠 LLM BASELINE (NO REAL CLUSTERING)
#############################################

llm_score = 0.5  # placeholder (no ground truth available)

#############################################
# 📊 OUTPUT ROW
#############################################

cluster_row = {
    "Task": "Clustering (label=1 only)",
    "Classic ML": cl_score,
    "Transformer": tr_score,
    "LLM": llm_score
}

print(cluster_row)