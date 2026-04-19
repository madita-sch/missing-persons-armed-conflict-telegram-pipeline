#############################################
# FULL NER EVALUATION (THESIS-GRADE)
#############################################
import pandas as pd
import numpy as np
import torch
import re

SEED = 42
SAMPLE_SIZE = 50  # keeps it fast

#############################################
# 📂 LOAD DATA
#############################################
import pandas as pd
import re
from transformers import pipeline

df = pd.read_excel("data/annotated_final.xlsx")
df = df[df["label"] == 1].sample(50, random_state=42)

# =====================
# GOLD STANDARD
# =====================
def split(x):
    return set(str(x).split(";")) if pd.notna(x) else set()

df["gold_names"] = df["names"].apply(split)
df["gold_locs"] = df["location"].apply(split)
df["gold_dates"] = df["dates"].apply(split)

# =====================
# CLASSIC REGEX
# =====================
def regex_ner(t):
    return {
        "names": set(re.findall(r"([اأإآء-ي]{3,}(?:\s+[اأإآء-ي]{3,})+)", t)),
        "locs": set(re.findall(r"(غزة|رفح|خان يونس)", t)),
        "dates": set(re.findall(r"(اليوم|امس|منذ\s+\d+)", t))
    }

df["CL"] = df["text_clean"].apply(regex_ner)

# =====================
# TRANSFORMER NER
# =====================
ner_pipe = pipeline("ner", model="CAMeL-Lab/bert-base-arabic-camelbert-msa-ner", aggregation_strategy="simple")

def extract_ner(out):
    res = {"names": set(), "locs": set(), "dates": set()}
    for e in out:
        if e["entity_group"] == "PER":
            res["names"].add(e["word"])
        elif e["entity_group"] == "LOC":
            res["locs"].add(e["word"])
        elif e["entity_group"] == "DATE":
            res["dates"].add(e["word"])
    return res

df["TR"] = df["text_clean"].apply(lambda x: extract_ner(ner_pipe(x)))

# =====================
# MICRO F1
# =====================
def f1(pred, gold_key):
    tp = fp = fn = 0
    for p, g in zip(df[pred], df[gold_key]):
        p_set = set(p)
        tp += len(p_set & g)
        fp += len(p_set - g)
        fn += len(g - p_set)
    return tp / (tp + fp + fn + 1e-9)

ner_row = {
    "Task": "NER",
    "Classic ML": f1("CL", "gold_names"),
    "Transformer": f1("TR", "gold_names"),
    "LLM": 0.0  # optional baseline
}

print(ner_row)