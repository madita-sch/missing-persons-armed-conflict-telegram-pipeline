import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "aubmindlab/bert-base-arabertv02"


# =========================================================
# 1. LOAD DATA (SAFE FOR CSV + XLSX)
# =========================================================
def load_data(path):
    df = pd.read_excel(path, engine="openpyxl")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    # encode labels if needed
    if df["label"].dtype == "object":
        df["label"] = df["label"].astype("category").cat.codes

    return Dataset.from_pandas(df.reset_index(drop=True))

# =========================================================
# 2. TRAINING
# =========================================================
def train_model(train_path, output_dir):
    dataset = load_data(train_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)

    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(set(dataset["label"]))
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_dir="./logs",
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


# =========================================================
# 3. PREDICTION
# =========================================================
def predict(model_path, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=1)

    return preds.cpu().numpy()