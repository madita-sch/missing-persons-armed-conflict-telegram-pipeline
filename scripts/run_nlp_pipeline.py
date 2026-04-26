# Import libraries
import os
import pandas as pd

from src.nlp.classification import predict
from src.nlp.ner import apply_ner_to_df
from src.nlp.translation import apply_translation_to_df
from src.nlp.clustering import normalize, build_graph, extract_clusters
from src.nlp.pseudonymization import anonymize_dataframe

def run_nlp_pipeline(
    input_path="data/nlp/telegram_clean.xlsx",
    output_path="outputs/nlp_results.xlsx",
    model_path="./model_output",
    sample_size=10,
    run_ner=True,
    run_translation=True,
    run_clustering=True,
):
    # Load preprocessed Telegram dataset
    df = pd.read_excel(input_path)
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    # Sequence classification to find missing cases using the predict function
    df["is_missing"] = predict(model_path, df["text_clean"].tolist())
    print(f"Found {df['is_missing'].sum()} potential cases")

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

    # Ensure columns produced by NER exist
    for col in ["names", "location", "dates", "age"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    # Translation
    if run_translation:
        try:
            # Only translate missing cases
            df_missing = df[df["is_missing"] == 1].copy()

            if len(df_missing) > 0:
                df_missing = apply_translation_to_df(
                    df_missing,
                    text_col="text_clean",
                    extra_cols=["names", "location", "dates"],
                )
                # copy translated columns back to main df
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
    
    # Clustering
    if run_clustering:
        try:
            #Preprocessing for clustering
            df["clean"] = df["names"].fillna("").apply(normalize)
            df["cluster_id"] = -1

            # Only cluster missing cases
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

    # Pseudonymization
    try:
        # Only pseudonymize missing cases
        df_missing = df[df["is_missing"] == 1].copy()

        df_missing, anon_map = anonymize_dataframe(
            df_missing,
            text_col="text_clean",
            names_col="names",
            cluster_col="cluster_id"   # IMPORTANT: pass correct arg name
        )

        # merge back results
        df.loc[df_missing.index, "text_clean_anon"] = df_missing["text_clean_anon"]
        print(f"Pseudonymization completed ({len(anon_map)} entities masked)")

    except Exception as e:
        print(f"Pseudonymization failed: {e}")
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="all_predictions", index=False)

    print("NLP Pipeline completed successfully")
    print(f"Saved to: {output_path}")

    return df


if __name__ == "__main__":
    run_nlp_pipeline()

