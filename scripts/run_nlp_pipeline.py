import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from rapidfuzz import fuzz

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers import MarianMTModel, MarianTokenizer

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = "data/nlp"
INPUT_PATH = os.path.join(BASE_DIR, "telegram_sample.xlsx")

ANNOTATED_PATH = "data/annotated_final.xlsx"
OUTPUT_PATH = os.path.join(BASE_DIR, "nlp_results.xlsx")

DEVICE = 0 if torch.cuda.is_available() else -1


# =========================================================
# 1. LOAD DATA (VERY SMALL SAMPLE ONLY)
# =========================================================
df = pd.read_excel(INPUT_PATH)
df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)

annot = pd.read_excel(ANNOTATED_PATH)
annot = annot[["text_clean", "label"]].dropna()


# =========================================================
# 2. CLASSICAL ML CLASSIFIER (TRAIN)
# =========================================================
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    analyzer="char_wb"
)

X_train = vectorizer.fit_transform(annot["text_clean"])
y_train = annot["label"]

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)


# =========================================================
# 3. CLASSIFY TELEGRAM SAMPLE
# =========================================================
df["text_clean"] = df["text"].fillna("").astype(str)
X_test = vectorizer.transform(df["text_clean"])

df["missing_prob"] = clf.predict_proba(X_test)[:, 1]
df["is_missing"] = (df["missing_prob"] > 0.5).astype(int)


# =========================================================
# 4. FILTER ONLY RELEVANT MESSAGES
# =========================================================
df_cases = df[df["is_missing"] == 1].copy().reset_index(drop=True)


# =========================================================
# 5. NER (TRANSFORMER - ARABIC)
# =========================================================
ner = pipeline(
    "ner",
    model="CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
    tokenizer="CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
    device=DEVICE
)

def extract_ner(text):
    try:
        ents = ner(text)
        names, locs, dates = [], [], []

        for e in ents:
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
    except:
        return pd.Series([None, None, None])

df_cases[["names", "locations", "dates"]] = df_cases["text_clean"].apply(extract_ner)


# =========================================================
# 6. TRANSLATION (AR -> EN)
# =========================================================
model_name = "Helsinki-NLP/opus-mt-ar-en"

translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
translator_model = MarianMTModel.from_pretrained(model_name)

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator_model.to(device_torch)

def translate(text):
    if not text:
        return ""

    batch = translator_tokenizer([text], return_tensors="pt", padding=True).to(device_torch)
    gen = translator_model.generate(**batch)
    return translator_tokenizer.decode(gen[0], skip_special_tokens=True)

df_cases["translation_en"] = df_cases["text_clean"].apply(translate)


# =========================================================
# 7. ENTITY MATCHING (FUZZY)
# =========================================================
def match_score(a, b):
    return fuzz.token_set_ratio(str(a), str(b))

# simple self-matching example (can extend later)
matches = []
for i in range(len(df_cases)):
    for j in range(i + 1, len(df_cases)):
        score = match_score(df_cases.loc[i, "text_clean"],
                            df_cases.loc[j, "text_clean"])
        if score > 85:
            matches.append((i, j, score))

df_matches = pd.DataFrame(matches, columns=["i", "j", "similarity"])


# =========================================================
# 8. CLUSTERING (TF-IDF + DBSCAN)
# =========================================================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_cases = tfidf.fit_transform(df_cases["text_clean"])

dbscan = DBSCAN(eps=0.6, min_samples=2, metric="cosine")
df_cases["cluster_id"] = dbscan.fit_predict(X_cases)


# =========================================================
# 9. SAVE RESULTS
# =========================================================
with pd.ExcelWriter(OUTPUT_PATH) as writer:
    df.to_excel(writer, sheet_name="all_predictions", index=False)
    df_cases.to_excel(writer, sheet_name="missing_cases", index=False)
    df_matches.to_excel(writer, sheet_name="entity_matches", index=False)

print("✅ NLP pipeline completed successfully")
print(f"Saved to: {OUTPUT_PATH}")
