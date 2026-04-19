import pandas as pd
import torch
import unicodedata
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL NAME
# =========================
NER_MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"

# =========================
# LOAD ON IMPORT (SAME AS YOUR WORKING SCRIPT)
# =========================
tokenizer_ner = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)

ner_model.to(device)
ner_model.eval()

id2label = ner_model.config.id2label

ner_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1,
)


# =========================
# ARABIC NORMALIZATION
# =========================
def normalize_arabic(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    return text


# =========================
# NER INFERENCE (SAME AS YOUR WORKING SCRIPT)
# =========================
def predict_ner(df, text_col="text_clean", batch_size=16):

    all_entities = []
    texts = df[text_col].fillna("").astype(str).tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = [normalize_arabic(t) for t in texts[i:i+batch_size]]

        results = ner_pipeline(batch_texts)

        if isinstance(results, dict):
            results = [results]

        for res in results:
            entities = []
            for item in res:
                label = item.get("entity_group") or item.get("entity")
                if label is None:
                    continue
                if label.startswith("B-") or label.startswith("I-"):
                    label = label[2:]
                entities.append((item["word"].strip(), label))
            all_entities.append(entities)

    return all_entities


# =========================
# POST-PROCESSING (SAME AS YOUR SCRIPT)
# =========================
def extract_entities_arabic(entity_pairs):

    label_map = {
        "PER": "names",
        "PERS": "names",
        "LOC": "location",
        "ORG": "organization",
        "DATE": "dates",
        "MISC": "misc"
    }

    fragments = {v: [] for v in label_map.values()}

    current_tokens = []
    current_label = None

    def flush():
        nonlocal current_tokens, current_label
        if current_tokens and current_label in label_map:
            text = " ".join(current_tokens).strip()
            if text:
                fragments[label_map[current_label]].append(text)
        current_tokens = []
        current_label = None

    for token, label in entity_pairs:
        token = token.replace("##", "").strip()
        if not token:
            continue

        if label not in label_map:
            flush()
            continue

        if current_label == label:
            current_tokens.append(token)
        else:
            flush()
            current_tokens = [token]
            current_label = label

    flush()

    return {k: "; ".join(v) for k, v in fragments.items()}


# =========================
# LOAD NER MODEL
# =========================
def load_ner_model():
    return tokenizer_ner, ner_model, id2label


# =========================
# APPLY NER TO DF
# =========================
def apply_ner_to_df(df, text_col="text_clean", tokenizer=None, model=None, id2label=None):
    entities_list = predict_ner(df, text_col)
    df["ner_extracted"] = entities_list

    extracted = [extract_entities_arabic(ent) for ent in entities_list]
    df_entities = pd.DataFrame(extracted, index=df.index)

    df["names"] = df_entities.get("names", "")
    df["location"] = df_entities.get("location", "")
    df["dates"] = df_entities.get("dates", "")
    df["organization"] = df_entities.get("organization", "")
    df["misc"] = df_entities.get("misc", "")

    return df