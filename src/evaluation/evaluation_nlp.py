# Import libraries
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
from nltk.translate.bleu_score import sentence_bleu


# Create Arabic normalization function
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

def normalize_arabic(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(ARABIC_DIACRITICS, "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Function to normalize dataframe
def normalize_dataframe(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace("\u00a0", " ").str.strip()
        df[col] = df[col].replace("nan", "")
    return df


# Split entities that are separated by semicolon and normalize each for arabic
def split_entities(text):
    if pd.isna(text) or str(text).strip() == "":
        return []
    return [normalize_arabic(e) for e in str(text).split(";") if e.strip()]

# Create Fuzzy match function, robust to Arabic orthographic variation
def is_match(a, b, threshold=90):
    a = normalize_arabic(a)
    b = normalize_arabic(b)
    if a == b:
        return True
    if fuzz.token_sort_ratio(a, b) >= threshold:
        return True
    if fuzz.partial_ratio(a, b) >= threshold:
        return True
    return False

# Alignment-based P/R/F1 with fuzzy matching
def score_entities(pred_list, gold_list):
    # Both empty, then nothing to evaluate for this row
    if not pred_list and not gold_list:
        return None, None, None

    # One side empty, then all FP or all FN
    if not pred_list:
        return 0.0, 0.0, 0.0
    if not gold_list:
        return 0.0, 0.0, 0.0

    matched_gold = set()
    matched_pred = set()

    for i, p in enumerate(pred_list):
        for j, g in enumerate(gold_list):
            if j in matched_gold:
                continue
            if is_match(p, g):
                matched_pred.add(i)
                matched_gold.add(j)
                break

    tp = len(matched_gold)
    fp = len(pred_list) - len(matched_pred)
    fn = len(gold_list) - len(matched_gold)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


# Evaluate classification function
def evaluate_classification(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    try:
        y_true = merged["is_missing_gold"].astype(int)
        y_pred = merged["is_missing_pred"].astype(int)

        return {
            "task": "classification",
            "entity": "-",
            "accuracy":     accuracy_score(y_true, y_pred),
            "f1_macro":     f1_score(y_true, y_pred, average="macro"),
            "f1_weighted":  f1_score(y_true, y_pred, average="weighted"),
            "precision":    None,
            "recall":       None,
            "f1":           None,
        }
    except KeyError:
        return {
            "task": "classification",
            "entity": "-",
            "accuracy":     float("nan"),
            "f1_macro":     float("nan"),
            "f1_weighted":  float("nan"),
            "precision":    None,
            "recall":       None,
            "f1":           None,
        }


# Evaluate NER function
def evaluate_ner(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    try:
        results = []
        for col in ["names", "location", "dates", "age"]:
            precisions, recalls, f1s = [], [], []

            for _, row in merged.iterrows():
                pred = split_entities(row[f"{col}_pred"])
                gold = split_entities(row[f"{col}_gold"])
                p, r, f = score_entities(pred, gold)
                if p is None:          # both sides empty → skip row
                    continue
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)

            n = len(f1s)  # rows where at least one side was non-empty
            results.append({
                "task":        "NER",
                "entity":      col,
                "n_scored":    n,
                "accuracy":    None,
                "f1_macro":    None,
                "f1_weighted": None,
                "precision":   float(np.mean(precisions)) if n else float("nan"),
                "recall":      float(np.mean(recalls))    if n else float("nan"),
                "f1":          float(np.mean(f1s))        if n else float("nan"),
            })

        return results
    except KeyError:
        return [
            {
                "task":        "NER",
                "entity":      col,
                "n_scored":    0,
                "accuracy":    None,
                "f1_macro":    None,
                "f1_weighted": None,
                "precision":   float("nan"),
                "recall":      float("nan"),
                "f1":          float("nan"),
            }
            for col in ["names", "location", "dates", "age"]
        ]


# Evaluate Clustering function
def evaluate_clustering(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    try:
        return {
            "task":    "clustering",
            "entity":  "-",
            "ARI":     adjusted_rand_score(
                merged["cluster_id_gold"],
                merged["cluster_id_pred"],
            ),
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }
    except KeyError:
        return {
            "task":    "clustering",
            "entity":  "-",
            "ARI":     float("nan"),
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }


# Evaluate translation function
def evaluate_translation(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    try:
        scores = []
        for _, row in merged.iterrows():
            ref_text = str(row.get("text_clean_en_gold", "")).strip()
            pred_text = str(row.get("text_clean_en_pred", "")).strip()

            ref  = ref_text.split() if ref_text else []
            pred = pred_text.split() if pred_text else []

            if ref and pred:
                scores.append(sentence_bleu([ref], pred))

        return {
            "task":    "translation",
            "entity":  "-",
            "BLEU":    float(np.mean(scores)) if scores else 0.0,
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }
    except KeyError:
        return {
            "task":    "translation",
            "entity":  "-",
            "BLEU":    float("nan"),
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }


# Evaluate Pseudonymization function
def evaluate_pseudonymization(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    try:
        leakage = merged["text_clean_anon_pred"].str.contains(
            r"[A-Za-z\u0600-\u06FF]{4,}", na=False
        ).sum()
        total = len(merged)

        return {
            "task":      "pseudonymization",
            "entity":    "-",
            "leak_rate": leakage / total,
            "coverage":  1 - (leakage / total),
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }
    except KeyError:
        return {
            "task":      "pseudonymization",
            "entity":    "-",
            "leak_rate": float("nan"),
            "coverage":  float("nan"),
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }

