import os
import pandas as pd
import numpy as np
import torch

from rapidfuzz import fuzz

# Import YOUR modules
from src.sequence_classification import predict
from src.ner import extract_entities
from src.clustering import cluster_texts
from src.translation import translate_texts


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = "data/nlp"
INPUT_PATH = os.path.join(BASE_DIR, "telegram_sample.xlsx")

ANNOTATED_PATH = "data/annotated_final.xlsx"  # used only for training beforehand
MODEL_PATH = "models/sequence_classifier"     # <-- must exist after training

OUTPUT_PATH = os.path.join(BASE_DIR, "nlp_results.xlsx")


# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_excel(INPUT_PATH)
df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)

df["text_clean"] = df["text"].fillna("").astype(str)


# =========================================================
# 2. CLASSIFICATION (TRANSFORMER)
# =========================================================
print("🔍 Running sequence classification...")

preds = predict(MODEL_PATH, df["text_clean"].tolist())

df["is_missing"] = preds
df["missing_prob"] = np.nan  # optional (not returned in current model)


# =========================================================
# 3. FILTER RELEVANT CASES
# =========================================================
df_cases = df[df["is_missing"] == 1].copy().reset_index(drop=True)

if len(df_cases) == 0:
    print("⚠️ No relevant cases found.")
    

# =========================================================
# 4. NER (from src)
# =========================================================
print("🧠 Running NER...")

ner_results = extract_entities(df_cases["text_clean"].tolist())

def parse_ner(entities):
    names, locs, dates = [], [], []

    for e in entities:
        if e["entity_group"] == "PER":
            names.append(e["word"])
        elif e["entity_group"] == "LOC":
            locs.append(e["word"])
        elif e["entity_group"] == "DATE":
            dates.append(e["word"])

    return pd.Series([
        "; ".join(names),
        "; ".join(locs),
        "; ".join(dates)
    ])

df_cases[["names", "locations", "dates"]] = pd.DataFrame(
    [parse_ner(e) for e in ner_results]
)


# =========================================================
# 5. TRANSLATION
# =========================================================
print("🌍 Translating...")

df_cases["translation_en"] = translate_texts(
    df_cases["text_clean"].tolist()
)


# =========================================================
# 6. ENTITY MATCHING (FUZZY)
# =========================================================
print("🔗 Matching similar cases...")

def match_score(a, b):
    return fuzz.token_set_ratio(str(a), str(b))

matches = []
for i in range(len(df_cases)):
    for j in range(i + 1, len(df_cases)):
        score = match_score(df_cases.loc[i, "text_clean"],
                            df_cases.loc[j, "text_clean"])
        if score > 85:
            matches.append((i, j, score))

df_matches = pd.DataFrame(matches, columns=["i", "j", "similarity"])


# =========================================================
# 7. CLUSTERING (EMBEDDINGS)
# =========================================================
print("🧩 Clustering...")

if len(df_cases) > 1:
    clusters = cluster_texts(df_cases["text_clean"].tolist(), n_clusters=5)
    df_cases["cluster_id"] = clusters
else:
    df_cases["cluster_id"] = -1


# =========================================================
# 8. SAVE RESULTS
# =========================================================
print("💾 Saving results...")

with pd.ExcelWriter(OUTPUT_PATH) as writer:
    df.to_excel(writer, sheet_name="all_predictions", index=False)
    df_cases.to_excel(writer, sheet_name="missing_cases", index=False)
    df_matches.to_excel(writer, sheet_name="entity_matches", index=False)

print("✅ NLP pipeline completed successfully")
print(f"Saved to: {OUTPUT_PATH}")