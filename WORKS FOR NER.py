import pandas as pd
import torch
import unicodedata
from transformers import AutoTokenizer, AutoModelForTokenClassification

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("data/annotated_final.xlsx")
df["text_clean"] = df["text_clean"].fillna("").astype(str)

# If you have a label column to filter by, keep it; otherwise skip
df = df[df["label"] == 1].sample(n=50, random_state=42).reset_index(drop=True)

# Arabic normalization utility
def normalize_arabic(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    return text

df["text_clean"] = df["text_clean"].apply(normalize_arabic)

# =========================
# LOAD MODEL
# =========================
NER_MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"

tokenizer_ner = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)

ner_model.to(device)
ner_model.eval()

id2label = ner_model.config.id2label

# =========================
# PREDICT NER (span-based, keeps full token sequences for entities)
# =========================
def predict_ner(df, text_column="text_clean", batch_size=16):
    all_entities = []

    for i in range(0, len(df), batch_size):
        batch_texts = df[text_column].iloc[i:i+batch_size]

        encoding = tokenizer_ner(
            batch_texts.tolist(),
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = ner_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

        for j in range(len(batch_texts)):
            word_ids = encoding.word_ids(batch_index=j)
            tokens = tokenizer_ner.convert_ids_to_tokens(input_ids[j])
            labels = [id2label[p.item()] for p in preds[j]]

            # Build entities by contiguous spans (preserve full token sequences)
            entities = []
            current_tokens = []
            current_label = None
            current_word_id = None

            for idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                tok = tokens[idx]
                lab = labels[idx]

                if lab.startswith("B-"):
                    if current_label is not None and current_tokens:
                        ent = tokenizer_ner.convert_tokens_to_string(current_tokens).strip()
                        if ent:
                            entities.append((ent, current_label))
                    current_tokens = [tok]
                    current_label = lab[2:]
                    current_word_id = wid
                elif lab.startswith("I-") and current_label == lab[2:] and wid == current_word_id:
                    current_tokens.append(tok)
                else:
                    if current_label is not None and current_tokens:
                        ent = tokenizer_ner.convert_tokens_to_string(current_tokens).strip()
                        if ent:
                            entities.append((ent, current_label))
                    current_tokens = [tok]
                    current_label = lab[2:] if lab.startswith("I-") or lab.startswith("B-") else None
                    current_word_id = wid if (lab.startswith("B-") or lab.startswith("I-")) else None

            if current_label is not None and current_tokens:
                ent = tokenizer_ner.convert_tokens_to_string(current_tokens).strip()
                if ent:
                    entities.append((ent, current_label))

            all_entities.append(entities)

    return all_entities

# =========================
# POST-PROCESSING: extract per-type lists and normalize
# =========================
def extract_entities_arabic(entity_pairs):
    # entity_pairs is a list of (span_text, label) where label in {'PER','LOC','ORG','DATE','MISC'}
    label_map = {
        "PER": "names",
        "LOC": "location",
        "ORG": "organization",
        "DATE": "dates",
        "MISC": "misc"
    }

    # Collect raw fragments per type
    fragments = {v: [] for v in label_map.values()}

    current_texts = []  # for building multi-token spans
    current_type = None

    def flush():
        nonlocal current_texts, current_type
        if current_texts:
            text = " ".join(current_texts).strip()
            if text:
                fragments[label_map.get(current_type, "misc")].append(text)
        current_texts = []
        current_type = None

    for token, label in entity_pairs:
        token = token.strip()
        if not token:
            continue

        if label is None:
            # treat as non-entity boundary
            if current_type is not None:
                flush()
            continue

        ent_type = label

        if ent_type in ["PER", "LOC", "ORG", "DATE", "MISC"]:
            if current_type == ent_type or current_type is None:
                current_texts.append(token)
                current_type = ent_type
            else:
                flush()
                current_texts = [token]
                current_type = ent_type
        else:
            # unknown tag; flush current if any
            if current_type is not None:
                flush()

    flush()

    # Heuristic to merge multi-part Arabic names better
    def merge_names(names):
        if not names:
            return []
        merged = []
        current = names<a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[0]</a>
        for nxt in names[1:]:
            # If next piece likely continues the name (no punctuation between), join
            if not any(p in nxt for p in ["-", "،", ",", ".", "؛"]):
                current = current + " " + nxt
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
        return merged

    if "names" in fragments:
        fragments["names"] = merge_names(fragments["names"])

    # Join each type into a semicolon-separated string
    return {k: "; ".join(v) for k, v in fragments.items()}


# =========================
# RUN PIPELINE
# =========================
ner_results = predict_ner(df)
df["ner_extracted"] = ner_results

df_entities = df["ner_extracted"].apply(extract_entities_arabic)

df["names"] = df_entities.apply(lambda x: x.get("names", ""))
df["location"] = df_entities.apply(lambda x: x.get("location", ""))
df["dates"] = df_entities.apply(lambda x: x.get("dates", ""))

# =========================
# SAVE
# =========================
df.to_excel("TEST_NER.xlsx", index=False)

print(df[["text_clean", "names", "location", "dates"]].head())
