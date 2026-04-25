import os
import pandas as pd

from src.nlp.classification import predict
from src.nlp.ner import apply_ner_to_df
#from src.nlp.translation import apply_translation_to_df
from src.nlp.clustering import cluster_by_name
from src.nlp.anonymization import anonymize_dataframe

def run_nlp_pipeline(
    input_path="data/nlp/telegram_clean.xlsx",
    output_path="outputs/nlp_results.xlsx",
    model_path="./model_output",
    sample_size=300,
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
            print("🧠 Running NER...")

            df_missing = df[df["is_missing"] == 1]

            if len(df_missing) > 0:
                df_missing = apply_ner_to_df(df_missing, text_col="text_clean")

                for col in ["names", "location", "dates", "age"]:
                    df.loc[df_missing.index, col] = df_missing[col]
            else:
                print("⚠️ No missing cases found — skipping NER")

        except Exception as e:
            print(f"⚠️ NER failed, continuing pipeline: {e}")

    # Ensure columns exist
    for col in ["names", "location", "dates", "age"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    # -------------------------------------------------------
    # TRANSLATION (ONLY AFTER NER)
    # -------------------------------------------------------

    # replace the entire translation block with this:
    if run_translation:
        try:
            print("🌐 Translating missing cases...")
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

                print(f"✅ Translated {len(df_missing)} missing cases")
            else:
                print("⚠️ No missing cases to translate")

        except Exception as e:
            import traceback
            traceback.print_exc()
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
    # CLUSTERING (NAME-FIRST SYSTEM)
    # -------------------------------------------------------
    if run_clustering:
        try:
            print("🧩 Running clustering (missing cases only)...")

            # ensure column exists
            if "cluster_id" not in df.columns:
                df["cluster_id"] = -1

            # isolate missing cases
            df_missing = df.loc[df["is_missing"] == 1].copy()

            if not df_missing.empty:

                # ✅ USE NAME-BASED CLUSTERING FUNCTION
                df_missing = cluster_by_name(df_missing)

                # assign back safely
                df.loc[df_missing.index, "cluster_id"] = df_missing["cluster_id"]

                # count clusters (excluding optional noise -1)
                n_clusters = df_missing["cluster_id"].nunique()

                print(f"Clusters found: {n_clusters}")

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
        sample_size=300,
        run_ner=True,
        run_translation=True,
        run_clustering=True,
    )

