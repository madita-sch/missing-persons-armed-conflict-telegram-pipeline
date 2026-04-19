from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# =========================================================
# LOAD MODEL ONCE
# =========================================================
_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================================================
# BUILD COMPOSITE TEXT (IMPROVED + CLEANER WEIGHTING)
# =========================================================
def build_weighted_text(row):
    """
    Combines:
    - text_clean (base meaning)
    - names (very strong signal)
    - location (strong signal)
    - dates (weak temporal signal)
    """

    text = str(row.get("text_clean", "")).strip()
    name = str(row.get("names", "")).strip()
    location = str(row.get("location", "")).strip()
    date = str(row.get("dates", "")).strip()

    parts = []

    NOISE_PHRASES = [
    "كما وصلني",
    "كما وصلنا",
    "مناشدة",
    "من لديه معلومات",
    "يرجى التواصل",
    "الرجاء النشر"
    ]

    for phrase in NOISE_PHRASES:
        text = text.replace(phrase, "")
    # Strong semantic anchors
    if name:
        parts.append(f"{name} {name} {name}")  # triple weight

    if location:
        parts.append(f"{location} {location}")  # double weight

    # weak temporal signal
    if date:
        parts.append(date)

    # base content
    if text:
        parts.append(text)

    return " ".join(parts).strip()


# =========================================================
# MAIN CLUSTERING FUNCTION (SIMILARITY-BASED LIKE YOUR TF-IDF VERSION)
# =========================================================
def cluster_cases(df, threshold=0.65):
    """
    Clusters cases using cosine similarity over SBERT embeddings.
    Uses explicit threshold grouping (like TF-IDF approach).
    """

    if df.empty:
        df["cluster_id"] = []
        return df

    # -----------------------------------------------------
    # 1. BUILD INPUT TEXTS
    # -----------------------------------------------------
    texts = df.apply(build_weighted_text, axis=1).tolist()

    # -----------------------------------------------------
    # 2. EMBEDDINGS (normalized => cosine-ready)
    # -----------------------------------------------------
    embeddings = _model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # -----------------------------------------------------
    # 3. COSINE SIMILARITY MATRIX
    # -----------------------------------------------------
    similarity_matrix = cosine_similarity(embeddings)

    # -----------------------------------------------------
    # 3.5 NAME-BASED PENALTY (FIXED + ROBUST VERSION)
    # -----------------------------------------------------
    from difflib import SequenceMatcher

    def name_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()


    names = df["names"].fillna("").astype(str).tolist()
    n = len(df)

    for i in range(n):
        for j in range(i + 1, n):

            name_i = names[i].strip()
            name_j = names[j].strip()

            # skip empty names
            if not name_i or not name_j:
                continue

            # if names are sufficiently different → penalize similarity
            if name_similarity(name_i, name_j) < 0.75:
                similarity_matrix[i, j] *= (0.3 + 0.7 * name_similarity(name_i, name_j))
                similarity_matrix[j, i] = similarity_matrix[i, j]
    # -----------------------------------------------------
    # 4. THRESHOLD-BASED CLUSTERING (LIKE YOUR REFERENCE)
    # -----------------------------------------------------
    cluster_ids = [-1] * n
    current_cluster = 0

    for i in range(n):
        if cluster_ids[i] != -1:
            continue

        cluster_ids[i] = current_cluster

        for j in range(i + 1, n):
            if cluster_ids[j] == -1 and similarity_matrix[i, j] >= threshold:
                cluster_ids[j] = current_cluster

        current_cluster += 1

    df["cluster_id"] = cluster_ids

    return df