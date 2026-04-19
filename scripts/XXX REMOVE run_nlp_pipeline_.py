import os
import pandas as pd
import numpy as np

from src.nlp.classification import predict
from src.nlp.ner import apply_ner_to_df, load_ner_model
from src.nlp.translation import translate_texts
from src.nlp.clustering import cluster_texts


def run_nlp_pipeline(
    input_path="data/nlp/telegram_sample.xlsx",
    output_path="outputs/nlp_results.xlsx",
    model_path="./model_output",
    sample_size=50,
    run_ner=True,
    run_translation=True,
    run_clustering=True
):

    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    print("📥 Loading data...")
    df = pd.read_excel(input_path)

    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    # =========================================================
    # 2. CLASSIFICATION
    # =========================================================
    print("🔍 Running classification...")

    df["is_missing"] = predict(model_path, df["text_clean"].tolist())

    print(f"Found {df['is_missing'].sum()} potential cases")

    # =========================================================
    # 3. NER (FIXED + SAFE)
    # =========================================================
    if run_ner:
        try:
            print("🧠 Loading NER model...")
            tokenizer, model, id2label = load_ner_model()

            print("🧠 Running NER...")

            df_missing = df[df["is_missing"] == 1].copy()

            if len(df_missing) > 0:
                df_missing = apply_ner_to_df(
                    df_missing,
                    tokenizer=tokenizer,
                    model=model,
                    id2label=id2label,
                    text_col="text_clean"
                )

                # merge back safely
                for col in ["names", "location", "dates"]:
                    df.loc[df_missing.index, col] = df_missing[col]
            else:
                print("⚠️ No missing cases found — skipping NER")

        except Exception as e:
            print(f"⚠️ NER failed, continuing pipeline: {e}")

    # =========================================================
    # 4. TRANSLATION
    # =========================================================
    #if run_translation:
     #   try:
      #      print("🌍 Translating...")
       #     df["translation_en"] = translate_texts(df["text_clean"].tolist())
        #except Exception as e:
         #   print(f"⚠️ Translation failed: {e}")
          #  df["translation_en"] = ""

    # =========================================================
    # 5. CLUSTERING
    # =========================================================
    #if run_clustering:
     #   try:
      #      print("🧩 Running semantic clustering...")
       #     df["cluster_id"] = cluster_texts(df["text_clean"].tolist())
        #except Exception as e:
         #   print(f"⚠️ Clustering failed: {e}")
          #  df["cluster_id"] = -1

    # =========================================================
    # 6. SAVE RESULTS
    # =========================================================
    print("💾 Saving results...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="all_predictions", index=False)

    print("✅ Pipeline completed successfully")
    print(f"📁 Saved to: {output_path}")

    return df


if __name__ == "__main__":

    run_nlp_pipeline(
        input_path="data/nlp/telegram_sample.xlsx",
        output_path="outputs/nlp_results.xlsx",
        model_path="./model_output",
        sample_size=50,
        run_ner=True,
        run_translation=True,
        run_clustering=True
    )