#############################################
# Comparison of Classic ML, Transformer, and LLM for Missing Persons Classification
#############################################
import pandas as pd
import numpy as np
import torch
import re

SEED = 42
SAMPLE_SIZE = 50  # keeps it fast


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

df = pd.read_excel("data/annotated_final.xlsx")[["text_clean", "label"]]
df = df.sample(50, random_state=42)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# =====================
# CLASSIC ML
# =====================
vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = vec.fit_transform(train_df["text_clean"])
X_val = vec.transform(val_df["text_clean"])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_df["label"])
cl_pred = clf.predict(X_val)

# =====================
# TRANSFORMER (AraBERT)
# =====================
model_name = "aubmindlab/bert-base-arabert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class DS(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        t = self.df.iloc[i]["text_clean"]
        y = self.df.iloc[i]["label"]
        enc = tokenizer(t, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {**{k:v.squeeze() for k,v in enc.items()}, "labels": torch.tensor(y)}

train_ds = DS(train_df)
val_ds = DS(val_df)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./tmp",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=10,
        report_to="none"
    ),
    train_dataset=train_ds
)

trainer.train()

def tr_predict(df):
    preds = []
    for t in df["text_clean"]:
        inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = model(**inputs)
        preds.append(torch.argmax(out.logits, dim=1).item())
    return preds

tr_pred = tr_predict(val_df)

# =====================
# LLM BASELINE
# =====================
from transformers import pipeline

llm = pipeline("text-generation", model="aubmindlab/aragpt2-medium")

def llm_pred(text):
    out = llm(f"Is this about missing person? 0 or 1:\n{text}\nAnswer:", max_new_tokens=10)[0]["generated_text"]
    return 1 if "1" in out else 0

llm_pred_vals = val_df["text_clean"].apply(llm_pred)

# =====================
# METRICS
# =====================
def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

classification_row = {
    "Task": "Sequence Classification",
    "Classic ML": f1(val_df["label"], cl_pred),
    "Transformer": f1(val_df["label"], tr_pred),
    "LLM": f1(val_df["label"], llm_pred_vals)
}

print(classification_row)
