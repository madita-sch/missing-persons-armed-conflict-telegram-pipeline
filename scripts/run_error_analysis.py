# Import libraries
import pandas as pd
from src.evaluation.evaluation_nlp import normalize_dataframe
from src.evaluation.error_analysis_nlp import build_error_analysis_report

# Run error analysis and save results to CSV
if __name__ == "__main__":

    PRED_PATH = "outputs/pred_Gaza20249.csv"
    GOLD_PATH = "data/gold_Gaza20249.csv"
    OUTPUT_PATH = "outputs/error_analysis_nlp_Gaza20249.xlsx"

    df_pred = normalize_dataframe(pd.read_csv(PRED_PATH))
    df_gold = normalize_dataframe(pd.read_csv(GOLD_PATH))

    # Fix types after normalization
    for df in [df_pred, df_gold]:
        df["is_missing"] = pd.to_numeric(df["is_missing"], errors="coerce").fillna(0).astype(int)
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(int)

    report = build_error_analysis_report(df_pred, df_gold, output_path=OUTPUT_PATH)

    print("\n===== CLASSIFICATION SAMPLE =====")
    print(report["classification"].head(10))

    print("\n===== NER SAMPLE =====")
    print(report["ner"].head(10))

    print("\n===== TRANSLATION SAMPLE =====")
    print(report["translation"].head(10))

    print("\n===== PSEUDONYMIZATION SAMPLE =====")
    print(report["pseudonymization"].head(10))