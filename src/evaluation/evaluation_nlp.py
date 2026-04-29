# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
from nltk.translate.bleu_score import sentence_bleu

# Import your pipeline functions (adjust paths if needed)
# from classification import predict
# from ner import apply_ner_to_df
# from clustering import ...
# from translation import ...
# from pseudonymization import ...

# =========================
# 1. CLASSIFICATION EVAL (AraBERT)
# =========================
def evaluate_classification(model_path, df):
    texts = df["text_clean"].tolist()
    y_true = df["label"].values
    y_pred = predict(model_path, texts)

    return {
        "task": "classification",
        "model": "AraBERT",
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


# =========================
# 2. NER EVAL (LLM + regex)
# =========================
def _entity_f1(pred, gold):
    pred_set = set((pred or "").split("; "))
    gold_set = set((gold or "").split("; "))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def evaluate_ner(df_pred, df_gold):
    results = []

    for col in ["names", "location", "dates", "age"]:
        p_list, r_list, f_list = [], [], []

        for _, row in df_pred.iterrows():
            p, r, f = _entity_f1(row[col], df_gold[col])
            p_list.append(p)
            r_list.append(r)
            f_list.append(f)

        results.append({
            "task": "NER",
            "model": "LLaMA 3.3-70B + regex",
            "entity": col,
            "precision": np.mean(p_list),
            "recall": np.mean(r_list),
            "f1": np.mean(f_list),
        })

    return results


# =========================
# 3. CLUSTERING EVAL
# =========================
def evaluate_clustering(df, true_col="true_cluster", pred_col="cluster_id"):
    return {
        "task": "clustering",
        "model": "graph-based similarity",
        "ARI": adjusted_rand_score(df[true_col], df[pred_col])
    }


# =========================
# 4. TRANSLATION EVAL
# =========================
def evaluate_translation(df):
    scores = []

    for _, row in df.iterrows():
        ref = str(row["text_en_ref"]).split()
        pred = str(row["text_en"]).split()

        if len(ref) == 0 or len(pred) == 0:
            continue

        scores.append(sentence_bleu([ref], pred))

    return {
        "task": "translation",
        "model": "LLaMA 3.3-70B",
        "BLEU": np.mean(scores)
    }


# =========================
# 5. PSEUDONYMIZATION EVAL
# =========================
def evaluate_pseudonymization(df):
    total_rows = len(df)

    # detect leakage (rough heuristic: remaining Arabic/Latin names)
    leakage = df["text_clean_anon"].str.contains(r"[A-Za-z\u0600-\u06FF]{4,}", na=False).sum()

    return {
        "task": "pseudonymization",
        "model": "registry + rules",
        "leak_rate": leakage / total_rows,
        "coverage": 1 - (leakage / total_rows),
    }


# =========================
# 6. MASTER EVALUATION RUNNER
# =========================
def run_all_evaluations(
    df,
    model_path,
    df_ner_gold=None
):

    results = []

    # ---- Classification ----
    results.append(evaluate_classification(model_path, df))

    # ---- NER ----
    if df_ner_gold is not None:
        ner_df = apply_ner_to_df(df)
        results.extend(evaluate_ner(ner_df, df_ner_gold))

    # ---- Clustering ----
    results.append(evaluate_clustering(df))

    # ---- Translation ----
    results.append(evaluate_translation(df))

    # ---- Pseudonymization ----
    results.append(evaluate_pseudonymization(df))

    return pd.DataFrame(results)


# =========================
# 7. RUN SCRIPT
# =========================
if __name__ == "__main__":

    # Load dataset (adjust paths)
    df = pd.read_csv("data/annotated_final.csv")

    # Optional gold NER dataset
    df_ner_gold = pd.read_csv("data/evaluation_dataset.csv")

    model_path = "./model_output"

    results_df = run_all_evaluations(df, model_path, df_ner_gold)

    print("\n===== FINAL EVALUATION RESULTS =====\n")
    print(results_df)

    results_df.to_csv("evaluation_results.csv", index=False)