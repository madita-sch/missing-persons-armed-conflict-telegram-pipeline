import os
import pandas as pd

from src.nlp.classification import predict
from src.nlp.ner import apply_ner_to_df, load_ner_model
from src.nlp.translation import translate_texts
from src.nlp.clustering import cluster_cases

def run_nlp_pipeline(
    input_path="data/nlp/telegram_clean.xlsx",
    output_path="outputs/nlp_results.xlsx",
    model_path="./model_output",
    sample_size=10,
    run_ner=True,
    run_translation=True,
    run_clustering=True,
):
    print("📥 Loading data...")
    df = pd.read_excel(input_path)
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    print("🔍 Running classification...")
    df["is_missing"] = predict(model_path, df["text_clean"].tolist())
    print(f"Found {df['is_missing'].sum()} potential cases")


    # -------------------------------------------------------
    # NER
    # -------------------------------------------------------
    if run_ner:
        try:
            print("🧠 Loading NER model...")
            tokenizer, model, ner_pipeline = load_ner_model()

            print("🧠 Running NER...")

            df_missing = df[df["is_missing"] == 1]

            if len(df_missing) > 0:
                df_missing = apply_ner_to_df(df_missing, text_col="text_clean")

                for col in ["names", "location", "dates"]:
                    df.loc[df_missing.index, col] = df_missing[col]
            else:
                print("⚠️ No missing cases found — skipping NER")

        except Exception as e:
            print(f"⚠️ NER failed, continuing pipeline: {e}")

    # Ensure columns exist
    for col in ["names", "location", "dates"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    # -------------------------------------------------------
    # TRANSLATION (ONLY AFTER NER)
    # -------------------------------------------------------
    if run_translation:
        try:
            print("🌐 Translating AFTER NER (selected fields only)...")

            df_missing = df[df["is_missing"] == 1]

            if len(df_missing) > 0:
                df.loc[df_missing.index, "text_en"] = translate_texts(
                    df_missing["text_clean"].tolist()
                )

                df.loc[df_missing.index, "names_en"] = translate_texts(
                    df_missing["names"].fillna("").tolist()
                )

                df.loc[df_missing.index, "location_en"] = translate_texts(
                    df_missing["location"].fillna("").tolist()
                )

                df.loc[df_missing.index, "dates_en"] = translate_texts(
                    df_missing["dates"].fillna("").tolist()
                )

                print(f"✅ Translated {len(df_missing)} missing cases")

            else:
                print("⚠️ No missing cases to translate")

        except Exception as e:
            print(f"⚠️ Translation failed: {e}")
    # -------------------------------------------------------
    # CLUSTERING
    # -------------------------------------------------------
    if run_clustering:
        try:
            print("🧩 Running clustering (missing cases only)...")

            df_missing = df[df["is_missing"] == 1]

            if len(df_missing) > 0:
                df_missing = cluster_cases(df_missing)
                df.loc[df_missing.index, "cluster_id"] = df_missing["cluster_id"]
                print(f"Clusters found: {df_missing['cluster_id'].nunique()}")
            else:
                print("⚠️ No missing cases to cluster")
                df["cluster_id"] = -1

        except Exception as e:
            print(f"⚠️ Clustering failed: {e}")
            df["cluster_id"] = -1

    # -------------------------------------------------------
    # SAVE OUTPUT
    # -------------------------------------------------------
    print("💾 Saving results...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="all_predictions", index=False)

    print("✅ Pipeline V2 completed successfully")
    print(f"📁 Saved to: {output_path}")

    return df


if __name__ == "__main__":
    run_nlp_pipeline(
        input_path="data/nlp/telegram_clean.xlsx",
        output_path="outputs/nlp_results.xlsx",
        model_path="./model_output",
        sample_size=10,
        run_ner=True,
        run_translation=True,
        run_clustering=True,
    )