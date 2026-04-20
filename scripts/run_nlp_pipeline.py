import os
import pandas as pd

from src.nlp.classification import predict
from src.nlp.ner import apply_ner_to_df, load_ner_model
#from src.nlp.translation import translate_texts
from src.nlp.clustering import cluster_cases, build_entity_map
from src.nlp.anonymization import anonymize_dataframe

def run_nlp_pipeline(
    input_path="data/nlp/telegram_clean.xlsx",
    output_path="outputs/nlp_results.xlsx",
    model_path="./model_output",
    sample_size=100,
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
            print("🌐 Translating AFTER NER (batched)...")

            df_missing = df[df["is_missing"] == 1].copy()

            if len(df_missing) > 0:
                # Ensure columns exist
                for col in ["names", "location", "dates"]:
                    if col not in df_missing.columns:
                        df_missing[col] = ""
                    else:
                        df_missing[col] = df_missing[col].fillna("")

                # --- 1. Combine all fields into ONE list ---
                combined_texts = (
                    df_missing["text_clean"].tolist() +
                    df_missing["names"].tolist() +
                    df_missing["location"].tolist() +
                    df_missing["dates"].tolist()
                )

                # --- 2. Translate ONCE ---
                translated_all = translate_texts(combined_texts, batch_size=8)

                # --- 3. Split back ---
                n = len(df_missing)

                text_en = translated_all[0:n]
                names_en = translated_all[n:2*n]
                location_en = translated_all[2*n:3*n]
                dates_en = translated_all[3*n:4*n]

                # --- 4. Assign back ---
                df.loc[df_missing.index, "text_en"] = text_en
                df.loc[df_missing.index, "names_en"] = names_en
                df.loc[df_missing.index, "location_en"] = location_en
                df.loc[df_missing.index, "dates_en"] = dates_en

                print(f"✅ Translated {len(df_missing)} missing cases (batched)")

            else:
                print("⚠️ No missing cases to translate")

        except Exception as e:
            print(f"⚠️ Translation failed: {e}")
    
        # -------------------------------------------------------
        # ENTITY MATCHING (NEW STEP - CRITICAL)
        # -------------------------------------------------------
        print("🧩 Running entity matching...")

        try:
            # Use translated names if available, fallback to raw
            name_source = df["names_en"] if "names_en" in df.columns else df["names"]

            name_source = name_source.fillna("").astype(str).tolist()

            entity_map = build_entity_map(name_source)

            df["entity_id"] = pd.Series(name_source).map(entity_map)

            # fallback safety
            df["entity_id"] = df["entity_id"].fillna("")

            print(f"✅ Entity matching completed ({len(entity_map)} unique entities)")

        except Exception as e:
            print(f"⚠️ Entity matching failed: {e}")
            df["entity_id"] = ""
    
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
    # ANONYMIZATION LAYER (NEW)
    # -------------------------------------------------------
    try:
        print("🕶️ Running anonymization layer...")

        df, anon_map = anonymize_dataframe(
            df,
            entity_col="entity_id",
            text_col="text_clean",
            names_col="names"
        )

        print(f"✅ Anonymization completed ({len(anon_map)} entities masked)")

    except Exception as e:
        print(f"⚠️ Anonymization failed: {e}")
    
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
        sample_size=100,
        run_ner=True,
        run_translation=True,
        run_clustering=True,
    )