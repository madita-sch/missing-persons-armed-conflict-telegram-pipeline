# Import libraries
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, classification_report
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
    if not pred_list and not gold_list:
        return None, None, None
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


# Evaluate classification function — runs on ALL rows (by design)
def evaluate_classification(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    try:
        y_true = merged["is_missing_gold"].astype(int)
        y_pred = merged["is_missing_pred"].astype(int)

        # ADD THESE THREE LINES
        print("\n--- Per-class breakdown ---")
        print(classification_report(y_true, y_pred,
                                    target_names=["not missing", "missing"],
                                    digits=3))
        
        return {
            "task":        "classification",
            "entity":      "-",
            "n_eval":      len(merged),
            "accuracy":    accuracy_score(y_true, y_pred),
            "f1_macro":    f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision":   None,
            "recall":      None,
            "f1":          None,
        }
    except KeyError:
        return {
            "task":        "classification",
            "entity":      "-",
            "n_eval":      0,
            "accuracy":    float("nan"),
            "f1_macro":    float("nan"),
            "f1_weighted": float("nan"),
            "precision":   None,
            "recall":      None,
            "f1":          None,
        }


# Evaluate NER function — runs only on gold is_missing == 1 rows
def evaluate_ner(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    # Filter to rows where gold says is_missing == 1
    merged = merged[merged["is_missing_gold"] == 1]

    try:
        results = []
        for col in ["names", "location", "dates", "age"]:
            precisions, recalls, f1s = [], [], []

            for _, row in merged.iterrows():
                pred = split_entities(row[f"{col}_pred"])
                gold = split_entities(row[f"{col}_gold"])
                p, r, f = score_entities(pred, gold)
                if p is None:  # both sides empty → skip row
                    continue
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)

            n = len(f1s)
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


# Evaluate clustering function — runs only on gold is_missing == 1 rows
def evaluate_clustering(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))
    missing_only = merged[merged["is_missing_gold"] == 1].copy()
    missing_only["cluster_id_gold"] = missing_only["cluster_id_gold"].fillna(-1).astype(int)
    missing_only["cluster_id_pred"] = missing_only["cluster_id_pred"].fillna(-1).astype(int)

    # Normalise singleton cluster IDs to -1 before computing ARI.
    # Singletons receive arbitrary unique IDs that differ between pred and gold,
    # which ARI penalises even when the underlying partition is identical.
    # Normalising ensures only true multi-message clusters affect the score.
    def normalize_cluster_ids(series):
        counts = series.value_counts()
        singletons = set(counts[counts == 1].index)
        return series.apply(lambda x: -1 if x in singletons else x)

    missing_only["cluster_id_gold_norm"] = normalize_cluster_ids(missing_only["cluster_id_gold"])
    missing_only["cluster_id_pred_norm"] = normalize_cluster_ids(missing_only["cluster_id_pred"])

    try:
        return {
            "task":     "clustering",
            "entity":   "-",
            "n_eval":   len(missing_only),
            "ARI":      adjusted_rand_score(
                missing_only["cluster_id_gold_norm"],
                missing_only["cluster_id_pred_norm"],
            ),
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }
    except KeyError:
        return {
            "task":     "clustering",
            "entity":   "-",
            "n_eval":   0,
            "ARI":      float("nan"),
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }


# Evaluate translation function — runs only on gold is_missing == 1 rows
def evaluate_translation(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    # Filter to rows where gold says is_missing == 1
    missing_only = merged[merged["is_missing_gold"] == 1]

    try:
        scores = []
        for _, row in missing_only.iterrows():
            ref_text  = str(row.get("text_clean_en_gold", "")).strip()
            pred_text = str(row.get("text_clean_en_pred", "")).strip()

            ref  = ref_text.split()  if ref_text  else []
            pred = pred_text.split() if pred_text else []

            if ref and pred:
                scores.append(sentence_bleu([ref], pred))

        return {
            "task":     "translation",
            "entity":   "-",
            "n_eval":   len(missing_only),
            "BLEU":     float(np.mean(scores)) if scores else 0.0,
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }
    except KeyError:
        return {
            "task":     "translation",
            "entity":   "-",
            "n_eval":   0,
            "BLEU":     float("nan"),
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }


# Evaluate pseudonymization function — runs only on gold is_missing == 1 rows
def evaluate_pseudonymization(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))
    missing_only = merged[merged["is_missing_gold"] == 1].copy()

    try:
        total = len(missing_only)
        leaked = 0

        for _, row in missing_only.iterrows():
            anon_text = str(row.get("text_clean_anon_pred", "") or "")
            
            # Check if any gold name survived (fuzzy match against anonymized text)
            gold_names = [n.strip() for n in str(row.get("names_gold", "") or "").split(";") if n.strip()]
            name_leaked = any(
                name in anon_text or
                any(token in anon_text for token in name.split() if len(token) > 3)
                for name in gold_names
            )

            # Check if any raw phone number survived (digits 7+ long)
            phone_leaked = bool(re.search(r'\d{7,}', anon_text))

            if name_leaked or phone_leaked:
                leaked += 1

        return {
            "task":      "pseudonymization",
            "entity":    "-",
            "n_eval":    total,
            "leak_rate": leaked / total if total > 0 else float("nan"),
            "coverage":  1 - (leaked / total) if total > 0 else float("nan"),
            "accuracy":  None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }
    except KeyError:
        return {
            "task":      "pseudonymization",
            "entity":    "-",
            "n_eval":    0,
            "leak_rate": float("nan"),
            "coverage":  float("nan"),
            "accuracy":  None, "f1_macro": None, "f1_weighted": None,
            "precision": None, "recall": None, "f1": None,
        }
