import pandas as pd
import re
from difflib import SequenceMatcher

# -----------------------------------------------------
# 1. NORMALIZE NAME
# -----------------------------------------------------
def normalize_name(name):
    name = str(name).lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name

# -----------------------------------------------------
# 2. NAME-BASED CLUSTERING (MAIN LOGIC)
# -----------------------------------------------------
def cluster_by_name(df, threshold=0.88):

    df = df.copy()

    names = df["names"].fillna("").astype(str).tolist()

    normalized = [normalize_name(n) for n in names]

    clusters = [-1] * len(df)
    cluster_id = 0

    for i in range(len(df)):

        if clusters[i] != -1:
            continue

        clusters[i] = cluster_id
        name_i = normalized[i]

        for j in range(i + 1, len(df)):

            if clusters[j] != -1:
                continue

            name_j = normalized[j]

            # skip empty names
            if not name_i or not name_j:
                continue

            # similarity between names
            sim = SequenceMatcher(None, name_i, name_j).ratio()

            if sim >= threshold:
                clusters[j] = cluster_id

        cluster_id += 1

    df["cluster_id"] = clusters

    return df
