import pandas as pd
from src.evaluation.evaluation_nlp import normalize_dataframe
from src.evaluation.error_analysis_nlp import build_error_analysis_report


if __name__ == "__main__":

    PRED_PATH = "outputs/pred_Gaza20249.csv"
    GOLD_PATH = "data/gold_Gaza20249.csv"

    df_pred = normalize_dataframe(pd.read_csv(PRED_PATH))
    df_gold = normalize_dataframe(pd.read_csv(GOLD_PATH))

    report = build_error_analysis_report(df_pred, df_gold)

    print("\n===== CLASSIFICATION SAMPLE =====")
    print(report["classification"].head(10))

    print("\n===== NER SAMPLE =====")
    print(report["ner"].head(10))

    print("\n===== TRANSLATION SAMPLE =====")
    print(report["translation"].head(10))

    print("\n===== PSEUDONYMIZATION SAMPLE =====")
    print(report["pseudonymization"].head(10))