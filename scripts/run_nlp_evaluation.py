import pandas as pd
from src.evaluation.evaluation_nlp import (
    normalize_dataframe,
    evaluate_classification,
    evaluate_ner,
    evaluate_clustering,
    evaluate_translation,
    evaluate_pseudonymization,
)

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
    PRED_PATH = "outputs/pred_Gaza20249.csv"
    GOLD_PATH = "data/gold_Gaza20249.csv"

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
    results.append(evaluate_clustering(df_pred, df_gold))
    results.append(evaluate_translation(df_pred, df_gold))
    results.append(evaluate_pseudonymization(df_pred, df_gold))

    # ── Display & save ────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    print("\n===== FINAL EVALUATION RESULTS =====\n")
    print(results_df.to_string(index=False))

    results_df.to_csv("outputs/evaluation_nlp.csv", index=False, encoding="utf-8")
    print("\nSaved → outputs/evaluation_nlp.csv")

