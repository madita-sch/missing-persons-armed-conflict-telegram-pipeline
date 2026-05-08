import os
import pandas as pd
import traceback

from src.nlp.pseudonymization import pseudonymize_dataframe

INPUT_PATH  = "outputs/evaluation_results_test_dataset.csv"
OUTPUT_PATH = "outputs/evaluation_results_test_dataset_anon.csv"

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows. Missing cases: {df['is_missing'].sum()}")

# Ensure required columns exist
for col in ["names", "cluster_id"]:
    if col not in df.columns:
        print(f"WARNING: column '{col}' not found — filling with default")
        df[col] = "" if col == "names" else -1

df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)
df["names"]      = df["names"].fillna("")

df_missing = df[df["is_missing"] == 1].copy()
print(f"Running pseudonymization on {len(df_missing)} missing cases...")

try:
    df_missing, anon_map = pseudonymize_dataframe(
        df_missing,
        text_col="text_clean",
        names_col="names",
        cluster_col="cluster_id",
    )
    df.loc[df_missing.index, "text_clean_anon"] = df_missing["text_clean_anon"]
    print(f"Pseudonymization completed — {len(anon_map)} entities masked")

except Exception as e:
    print(f"\nPseudonymization failed: {e}")
    traceback.print_exc()
    print("\nSaving output anyway with empty text_clean_anon column...")
    df["text_clean_anon"] = ""

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"Saved to: {OUTPUT_PATH}")