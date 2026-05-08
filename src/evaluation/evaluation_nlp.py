# =========================
# IMPORTS
# =========================
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
from nltk.translate.bleu_score import sentence_bleu


# =========================
# ARABIC NORMALIZATION
# =========================
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


# =========================
# NORMALIZATION
# =========================
def normalize_dataframe(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace("\u00a0", " ").str.strip()
        df[col] = df[col].replace("nan", "")
    return df


# =========================
# ENTITY HELPERS
# =========================
def split_entities(text):
    """Split semicolon-separated entities, normalizing each for Arabic."""
    if pd.isna(text) or str(text).strip() == "":
        return []
    return [normalize_arabic(e) for e in str(text).split(";") if e.strip()]


def is_match(a, b, threshold=90):
    """Fuzzy match robust to Arabic orthographic variation."""
    a = normalize_arabic(a)
    b = normalize_arabic(b)
    if a == b:
        return True
    if fuzz.token_sort_ratio(a, b) >= threshold:
        return True
    if fuzz.partial_ratio(a, b) >= threshold:
        return True
    return False


def score_entities(pred_list, gold_list):
    """Alignment-based P/R/F1 with fuzzy matching.

    Returns (None, None, None) when both lists are empty so the caller
    can skip this row rather than averaging in a spurious 0.0.
    Returns (0, 0, 0) when only one side is empty (genuine miss/hallucination).
    """
    # Both empty → nothing to evaluate for this row
    if not pred_list and not gold_list:
        return None, None, None

    # One side empty → all FP or all FN
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


# =========================
# 1. CLASSIFICATION
# =========================
def evaluate_classification(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

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


# =========================
# 2. NER
# =========================
def evaluate_ner(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

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


# =========================
# 3. CLUSTERING
# =========================
def evaluate_clustering(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id")

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


# =========================
# 4. TRANSLATION
# =========================
def evaluate_translation(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id")

    scores = []
    for _, row in merged.iterrows():
        ref  = str(row["text_en_gold"]).split()
        pred = str(row["text_en_pred"]).split()
        if ref and pred:
            scores.append(sentence_bleu([ref], pred))

    return {
        "task":    "translation",
        "entity":  "-",
        "BLEU":    float(np.mean(scores)) if scores else 0.0,
        "accuracy": None, "f1_macro": None, "f1_weighted": None,
        "precision": None, "recall": None, "f1": None,
    }


# =========================
# 5. PSEUDONYMIZATION
# =========================
def evaluate_pseudonymization(df_pred, df_gold):
    merged = df_pred.merge(df_gold, on="id")

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


# =========================
# DEBUG HELPER
# =========================
def diff_check(df_pred, df_gold, cols):
    """Print how many rows differ between pred and gold per column."""
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))
    print("\n===== DIFF CHECK =====")
    for col in cols:
        if f"{col}_pred" in merged.columns and f"{col}_gold" in merged.columns:
            diff = (merged[f"{col}_pred"] != merged[f"{col}_gold"]).sum()
            print(f"  {col}: {diff} differing rows out of {len(merged)}")
    print()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # ── Load & normalize ──────────────────────────────────────────────
    PRED_PATH = "outputs/evaluation_results_test_dataset_OLD BUT WORKED.csv"
    GOLD_PATH = "data/evaluation_correct_dataset.csv"

    df_pred = normalize_dataframe(pd.read_csv(PRED_PATH))
    df_gold = normalize_dataframe(pd.read_csv(GOLD_PATH))

    print("Columns :", df_pred.columns.tolist())
    print("Shape   :", df_pred.shape)
    print(df_pred.head(2))

    # ── Sanity diff check ─────────────────────────────────────────────
    diff_check(df_pred, df_gold, ["is_missing", "names", "location", "dates", "age"])

    # ── Run evaluations ───────────────────────────────────────────────
    results = []
    results.append(evaluate_classification(df_pred, df_gold))
    results.extend(evaluate_ner(df_pred, df_gold))
    # results.append(evaluate_clustering(df_pred, df_gold))
    # results.append(evaluate_translation(df_pred, df_gold))
    # results.append(evaluate_pseudonymization(df_pred, df_gold))

    # ── Display & save ────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    print("\n===== FINAL EVALUATION RESULTS =====\n")
    print(results_df.to_string(index=False))

    results_df.to_csv("evaluation_results.csv", index=False, encoding="utf-8")
    print("\nSaved → evaluation_results.csv")
