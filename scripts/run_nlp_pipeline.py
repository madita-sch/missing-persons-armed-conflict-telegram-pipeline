# Import libraries
import os
import pandas as pd

from src.nlp.classification import predict
from src.nlp.ner import apply_ner_to_df
from src.nlp.translation import apply_translation_to_df
from src.nlp.clustering import normalize, build_graph, extract_clusters
from src.nlp.pseudonymization import pseudonymize_dataframe

# Configuration: replace with paths to input Telegram dataset, 
# output path for predictions, model path for classification, sample size
INPUT_PATH = "data/text_ALMAFKODEN/telegram_clean.xlsx"
OUTPUT_PATH = "outputs/pred_ALMAFKODEN.csv"
MODEL_PATH = "./model_output"
SAMPLE_SIZE = 100

# Define main pipeline function merging all steps (classification, NER, translation, clustering, pseudonymization)
# Adapt input and output paths, and sample_size as needed
def run_nlp_pipeline(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
    model_path=MODEL_PATH,
    sample_size=SAMPLE_SIZE,
    run_ner=True,
    run_translation=True,
    run_clustering=True,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Helper to save checkpoints
    def save_checkpoint(df, stage):
        checkpoint_path = output_path.replace(".csv", f"_checkpoint_{stage}.csv")
        df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
        print(f"[Checkpoint] Saved after '{stage}' → {checkpoint_path}")

    # Load preprocessed Telegram dataset
    df = pd.read_excel(input_path)
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    # Sequence classification
    df["is_missing"] = predict(model_path, df["text_clean"].tolist())
    print(f"Found {df['is_missing'].sum()} potential cases")
    save_checkpoint(df, "classification")           # Checkpoint saved after classification

    # NER
    if run_ner:
        try:
            df_missing = df[df["is_missing"] == 1]
            if len(df_missing) > 0:
                df_missing = apply_ner_to_df(df_missing, text_col="text_clean")
                for col in ["names", "location", "dates", "age"]:
                    df.loc[df_missing.index, col] = df_missing[col]
            else:
                print("No missing cases found")
        except Exception as e:
            print(f"NER failed, continuing pipeline: {e}")

    for col in ["names", "location", "dates", "age"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    save_checkpoint(df, "ner")                      # Checkpoint saved after NER

    # Translation
    if run_translation:
        try:
            df_missing = df[df["is_missing"] == 1].copy()
            if len(df_missing) > 0:
                df_missing = apply_translation_to_df(
                    df_missing,
                    text_col="text_clean",
                    extra_cols=["names", "location", "dates"],
                )
                for col in ["text_clean_en", "names_en", "location_en", "dates_en"]:
                    if col in df_missing.columns:
                        df.loc[df_missing.index, col] = df_missing[col]
                print(f"Translated {len(df_missing)} missing cases")
            else:
                print("No missing cases to translate")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Translation failed: {e}")

    save_checkpoint(df, "translation")              # Checkpoint saved after translation

    # Clustering
    if run_clustering:
        try:
            df["clean"] = df["names"].fillna("").apply(normalize)
            df["cluster_id"] = -1
            df_missing = df[df["is_missing"] == 1].copy()
            if not df_missing.empty:
                G = build_graph(df_missing, threshold=0.60)
                clusters = extract_clusters(G)
                df.loc[df_missing.index, "cluster_id"] = df_missing.index.map(clusters)
                print(f"Clusters found: {len(set(clusters.values()))}")
            else:
                print("No missing cases to cluster")
        except Exception as e:
            print(f"Clustering failed: {e}")
            df["cluster_id"] = -1

    save_checkpoint(df, "clustering")              # Checkpoint saved after clustering

    # Pseudonymization
    try:
        df_missing = df[df["is_missing"] == 1].copy()
        df_missing, anon_map = pseudonymize_dataframe(
            df_missing,
            text_col="text_clean",
            names_col="names",
            cluster_col="cluster_id"
        )
        df.loc[df_missing.index, "text_clean_anon"] = df_missing["text_clean_anon"]
        print(f"Pseudonymization completed ({len(anon_map)} entities masked)")
    except Exception as e:
        print(f"Pseudonymization failed: {e}")

    # Remove "clean" column used as helper column for clustering
    df = df.drop(columns=["clean"], errors="ignore")

    # Save final output
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("NLP Pipeline completed successfully")
    print(f"Saved to: {output_path}")

    return df

# Run the pipeline
if __name__ == "__main__":
    run_nlp_pipeline(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
        sample_size=SAMPLE_SIZE,
    )
    