from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
from difflib import SequenceMatcher

# =========================================================
# LOAD MODEL ONCE
# =========================================================
_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# =========================================================
# NAME NORMALIZATION
# =========================================================
def normalize_name(name):
    name = str(name).lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name

# =========================================================
# BUILD ENTITY MAP (GLOBAL ENTITY MATCHING)
# =========================================================
def build_entity_map(names, threshold=0.85):
    canonical = []
    entity_map = {}
    entity_id_map = {}

    id_counter = 0

    for name in names:
        norm = normalize_name(name)

        if not norm:
            entity_map[name] = None
            continue

        matched_id = None

        for canon in canonical:
            sim = SequenceMatcher(None, norm, canon).ratio()
            if sim >= threshold:
                matched_id = entity_id_map[canon]
                break

        if matched_id is None:
            canonical.append(norm)

            matched_id = f"entity_{id_counter:04d}"
            entity_id_map[norm] = matched_id
            id_counter += 1

        entity_map[name] = matched_id

    return entity_map

# =========================================================
# BUILD COMPOSITE TEXT (USING ENTITY ID)
# =========================================================
def build_weighted_text(row):
    text = str(row.get("text_clean", "")).strip()
    entity = str(row.get("entity_id", "")).strip()
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

    # Strong anchor: entity
    if entity:
        parts.append(f"{entity} {entity} {entity}")

    # Medium anchor: location
    if location:
        parts.append(f"{location} {location}")

    # Weak signal: date
    if date:
        parts.append(date)

    # Base text
    if text:
        parts.append(text)

    return " ".join(parts).strip()

# =========================================================
# MAIN CLUSTERING FUNCTION
# =========================================================
def cluster_cases(df, threshold=0.65):

    if df.empty:
        df["cluster_id"] = []
        return df

    # -----------------------------------------------------
    # 0. ENTITY MATCHING (NEW STEP)
    # -----------------------------------------------------
    names = df["names"].fillna("").astype(str).tolist()
    entity_map = build_entity_map(names)

    df["entity_id"] = df["names"].map(entity_map)

    # -----------------------------------------------------
    # 1. BUILD TEXTS
    # -----------------------------------------------------
    texts = df.apply(build_weighted_text, axis=1).tolist()

    # -----------------------------------------------------
    # 2. EMBEDDINGS
    # -----------------------------------------------------
    embeddings = _model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # -----------------------------------------------------
    # 3. COSINE SIMILARITY
    # -----------------------------------------------------
    similarity_matrix = cosine_similarity(embeddings)

    # -----------------------------------------------------
    # 4. ENTITY-BASED CONSTRAINT (REPLACES OLD PENALTY)
    # -----------------------------------------------------
    entity_ids = df["entity_id"].tolist()
    n = len(df)

    for i in range(n):
        for j in range(i + 1, n):

            e1 = entity_ids[i]
            e2 = entity_ids[j]

            if e1 and e2 and e1 != e2:
                # strong penalty if clearly different people
                similarity_matrix[i, j] *= 0.2
                similarity_matrix[j, i] = similarity_matrix[i, j]

    # -----------------------------------------------------
    # 5. THRESHOLD CLUSTERING
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