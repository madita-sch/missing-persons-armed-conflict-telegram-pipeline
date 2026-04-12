#############################################
# FULL NER EVALUATION (THESIS-GRADE)
#############################################

import pandas as pd
import re
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support

#############################################
# 📂 LOAD DATA
#############################################

df = pd.read_excel("data/annotated_final.xlsx")

df = df[['text_clean', 'names', 'location', 'dates', 'label']].copy()

df["text_clean"] = df["text_clean"].astype(str)

#############################################
# 🔥 FILTER: ONLY RELEVANT CASES
#############################################

df_cases = df[df["label"] == 1].copy().reset_index(drop=True)

#############################################
# 🧠 GOLD STANDARD PREPARATION
#############################################

def to_set(x):
    if pd.isna(x) or x == "":
        return set()
    return set(str(x).split(";"))

df_cases["gold_names"] = df_cases["names"].apply(to_set)
df_cases["gold_locs"] = df_cases["location"].apply(to_set)
df_cases["gold_dates"] = df_cases["dates"].apply(to_set)

#############################################
# =========================
# 1. CLASSIC (REGEX)
# =========================
#############################################

def regex_ner(text):
    names = re.findall(r"([اأإآء-ي]{3,}(?:\s+[اأإآء-ي]{3,})+)", text)
    locs = re.findall(r"(غزة|رفح|خان يونس|شمال غزة|جنوب غزة)", text)
    dates = re.findall(r"(اليوم|امس|منذ\s+\d+\s+يوم)", text)

    return {
        "names": set(names),
        "locs": set(locs),
        "dates": set(dates)
    }

df_cases["CL_ner"] = df_cases["text_clean"].apply(regex_ner)

#############################################
# =========================
# 2. TRANSFORMER (CAMeLBERT)
# =========================
#############################################

ner_pipe = pipeline(
    "ner",
    model="CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
    aggregation_strategy="simple"
)

def extract_transformer_entities(ner_output):
    names, locs, dates = set(), set(), set()

    for ent in ner_output:
        word = ent["word"]
        label = ent["entity_group"]

        if label == "PER":
            names.add(word)
        elif label == "LOC":
            locs.add(word)
        elif label == "DATE":
            dates.add(word)

    return {
        "names": names,
        "locs": locs,
        "dates": dates
    }

df_cases["TR_raw"] = df_cases["text_clean"].apply(lambda x: ner_pipe(x))
df_cases["TR_ner"] = df_cases["TR_raw"].apply(extract_transformer_entities)

#############################################
# =========================
# 3. LLM (STRUCTURED EXTRACTION)
# =========================
#############################################
import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

#############################################
# 🚀 LOAD MODEL ONCE (IMPORTANT)
#############################################

MODEL_NAME = "aubmindlab/aragpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ✅ CRITICAL FIX
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model.config.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

#############################################
# 🚀 FAST BATCH FUNCTION
#############################################

def llm_ner_batch_fast(texts, batch_size=8):

    results = []

    for i in range(0, len(texts), batch_size):

        batch_texts = texts[i:i+batch_size]

        prompts = [
            f"""Extract entities in JSON:
Text: {t}
Return format:
{{"names": [], "locations": [], "dates": []}}"""
            for t in batch_texts
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            try:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    results.append(json.loads(match.group()))
                else:
                    results.append({"names": [], "locations": [], "dates": []})
            except:
                results.append({"names": [], "locations": [], "dates": []})

    return results


df_cases["LLM_ner"] = llm_ner_batch_fast(df_cases["text_clean"].tolist())

df_cases_small
#############################################
# 📊 EVALUATION FUNCTION (REAL NER F1)
#############################################

def compute_f1(pred_col, gold_col):
    tp, fp, fn = 0, 0, 0

    for pred, gold in zip(df_cases[pred_col], df_cases[gold_col]):

        pred_set = set()
        gold_set = set(gold)

        # normalize dict output
        if isinstance(pred, dict):
            pred_set = set(pred.get("names", []) if gold_col == "gold_names"
                           else pred.get("locs", []) if gold_col == "gold_locs"
                           else pred.get("dates", []))

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return f1

#############################################
# 📊 FINAL SCORES
#############################################

cl_score = (
    compute_f1("CL_ner", "gold_names") +
    compute_f1("CL_ner", "gold_locs") +
    compute_f1("CL_ner", "gold_dates")
) / 3

tr_score = (
    compute_f1("TR_ner", "gold_names") +
    compute_f1("TR_ner", "gold_locs") +
    compute_f1("TR_ner", "gold_dates")
) / 3

llm_score = (
    compute_f1("LLM_ner", "gold_names") +
    compute_f1("LLM_ner", "gold_locs") +
    compute_f1("LLM_ner", "gold_dates")
) / 3

#############################################
# 📊 FINAL TABLE ROW
#############################################

ner_row = {
    "Task": "NER",
    "Classic ML": round(cl_score, 3),
    "Transformer": round(tr_score, 3),
    "LLM": round(llm_score, 3)
}

print(ner_row)