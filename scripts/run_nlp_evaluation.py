# Import libraries
import pandas as pd
from src.evaluation.evaluation_nlp import (
    normalize_dataframe,
    evaluate_classification,
    evaluate_ner,
    evaluate_clustering,
    evaluate_translation,
    evaluate_pseudonymization,
)

# Configuration: replace with paths to predicted, gold NLP datasets, and output path for error analysis report
PRED_PATH = "outputs/pred_ALMAFKODEN.csv"
GOLD_PATH = "data/gold_ALMAFKODEN.csv"
OUTPUT_PATH = "outputs/evaluation_nlp_ALMAFKODEN.csv"
    

# Define debug helper
def diff_check(df_pred, df_gold, cols):
    # Print how many rows differ between pred and gold per column.
    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))
    print("\n - DIFF CHECK")
    for col in cols:
        if f"{col}_pred" in merged.columns and f"{col}_gold" in merged.columns:
            diff = (merged[f"{col}_pred"] != merged[f"{col}_gold"]).sum()
            print(f"  {col}: {diff} differing rows out of {len(merged)}")
    print()

# Run evaluations
if __name__ == "__main__":

    # Normalize data
    df_pred = normalize_dataframe(pd.read_csv(PRED_PATH))
    df_gold = normalize_dataframe(pd.read_csv(GOLD_PATH))

    # Fix types after normalization (normalize_dataframe casts everything to str)
    for df in [df_pred, df_gold]:
        df["is_missing"] = pd.to_numeric(df["is_missing"], errors="coerce").fillna(0).astype(int)
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(int)

    print("Columns :", df_pred.columns.tolist())
    print("Shape   :", df_pred.shape)
    print(df_pred.head(2))

    # Sanity diff check to see how many rows differ between pred and gold for key columns before running evaluations
    diff_check(df_pred, df_gold, ["is_missing", "names", "location", "dates", "age"])

    # Run evaluations
    results = []
    results.append(evaluate_classification(df_pred, df_gold))
    results.extend(evaluate_ner(df_pred, df_gold))
    results.append(evaluate_clustering(df_pred, df_gold))
    results.append(evaluate_translation(df_pred, df_gold))
    results.append(evaluate_pseudonymization(df_pred, df_gold))

    # Display & save results
    results_df = pd.DataFrame(results)
    print("Evaluation results:")
    print(results_df.to_string(index=False))
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")


