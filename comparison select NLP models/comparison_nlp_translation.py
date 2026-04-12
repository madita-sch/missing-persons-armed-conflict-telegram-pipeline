#############################################
# TRANSLATION COMPARISON 
#############################################

import pandas as pd
import re
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

#############################################
# 📂 LOAD DATA (ONLY RELEVANT CASES)
#############################################

df = pd.read_excel("data/annotated_final.xlsx")

df = df[['text_clean', 'names', 'location', 'dates', 'label']].copy()
df["text_clean"] = df["text_clean"].astype(str)

# ONLY missing persons cases
df = df[df["label"] == 1].reset_index(drop=True)

#Create smaller subset 

df = df[df["label"] == 1].sample(10, random_state=42).copy()

#############################################
# 🧠 1. CLASSIC TRANSLATION (BASELINE)
#############################################

arabic_to_english = {
    "مفقود": "missing",
    "طفل": "child",
    "طفلة": "girl",
    "امرأة": "woman",
    "رجل": "man",
    "غزة": "Gaza",
    "رفح": "Rafah",
    "خان يونس": "Khan Younis"
}

def classic_translate(text):
    for ar, en in arabic_to_english.items():
        text = text.replace(ar, en)
    return text

df["CL_trans"] = df["text_clean"].apply(classic_translate)

##############################################
# 🤖 TRANSFORMER TRANSLATION (FINAL FIXED)
#############################################

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

#############################################
# 📥 LOAD MODEL
#############################################

model_name = "Helsinki-NLP/opus-mt-ar-en"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

#############################################
# 🚀 TRANSLATION FUNCTION
#############################################

def transformer_translate(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,          # improves quality slightly
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#############################################
# 📊 APPLY ONLY TO MISSING CASES (label = 1)
#############################################

df["TR_trans"] = df["text_clean"].apply(transformer_translate)

#############################################
# 🧠 FAST LLM TRANSLATION (SMALL TEST ONLY)
#############################################
from transformers import pipeline
import torch

llm = pipeline(
    "text-generation",
    model="aubmindlab/aragpt2-medium",
    device=0 if torch.cuda.is_available() else -1
)

# IMPORTANT FIXES
llm.tokenizer.pad_token = llm.tokenizer.eos_token

def llm_translate_batch(texts):

    prompts = [
        f"""Translate Arabic to English:

{text}

English:"""
        for text in texts
    ]

    outputs = llm(
        prompts,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=llm.tokenizer.eos_token_id
    )

    results = []

    for out in outputs:

        # FIX: pipeline can return dict OR list depending on version
        if isinstance(out, list):
            out = out[0]

        text = out.get("generated_text", "")

        if "English:" in text:
            results.append(text.split("English:")[-1].strip())
        else:
            results.append(text.strip())

    return results

df["LLM_trans"] = llm_translate_batch(
    df["text_clean"].tolist()
)


#############################################
# 📊 4. EVALUATION METRICS
#############################################

# Sentence embedding model (semantic similarity)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def semantic_similarity(src, tgt):
    emb1 = embedder.encode(src, convert_to_tensor=True)
    emb2 = embedder.encode(tgt, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

df["CL_sem"] = df.apply(lambda x: semantic_similarity(x["text_clean"], x["CL_trans"]), axis=1)
df["TR_sem"] = df.apply(lambda x: semantic_similarity(x["text_clean"], x["TR_trans"]), axis=1)
df["LLM_sem"] = df.apply(lambda x: semantic_similarity(x["text_clean"], x["LLM_trans"]), axis=1)

#############################################
# 🟡 ENTITY PRESERVATION SCORE
#############################################

def extract_entities(text):
    return set(re.findall(r"[اأإآء-ي]{3,}(?:\s+[اأإآء-ي]{3,})+", text))

def entity_score(src, tgt):
    src_ent = extract_entities(src)
    tgt_ent = extract_entities(tgt)

    if len(src_ent) == 0:
        return 1.0

    return len(src_ent & tgt_ent) / len(src_ent)

df["CL_ent"] = df.apply(lambda x: entity_score(x["text_clean"], x["CL_trans"]), axis=1)
df["TR_ent"] = df.apply(lambda x: entity_score(x["text_clean"], x["TR_trans"]), axis=1)
df["LLM_ent"] = df.apply(lambda x: entity_score(x["text_clean"], x["LLM_trans"]), axis=1)

#############################################
# 📊 5. FINAL SCORE COMBINATION
#############################################

def final_score(sem, ent):
    return 0.7 * sem + 0.3 * ent

cl_score = final_score(df["CL_sem"].mean(), df["CL_ent"].mean())
tr_score = final_score(df["TR_sem"].mean(), df["TR_ent"].mean())
llm_score = final_score(df["LLM_sem"].mean(), df["LLM_ent"].mean())

#############################################
# 📦 FINAL ROW FOR YOUR MASTER TABLE
#############################################

translation_row = {
    "Task": "Translation",
    "Classic ML": round(cl_score, 3),
    "Transformer": round(tr_score, 3),
    "LLM": round(llm_score, 3)
}

print("TRANSLATION RESULT:")
print(translation_row)

