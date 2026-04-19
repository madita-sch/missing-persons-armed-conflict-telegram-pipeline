import torch
import pandas as pd
import numpy as np
import random
import os

from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
# -----------------------------
# SEED
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# DATASET
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.loc[idx, "text_clean"])
        label = int(self.data.loc[idx, "label"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# -----------------------------
# WEIGHTED TRAINER
# -----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# -----------------------------
# TRAIN FUNCTION (FINAL)
# -----------------------------
def train_model(
    data_path="data/annotated_final.xlsx",
    model_name="aubmindlab/bert-base-arabertv2",
    output_dir="./model_output",
    max_length=128,
    test_size=0.2,
    seed=42,
    epochs=5,
    batch_size=8,
    lr=2e-5,
    sample_size=None
):

    set_seed(seed)

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = pd.read_excel(data_path)
    df = df[["text_clean", "label"]].copy()
    df["label"] = df["label"].astype(int)

    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=seed)

    # -------------------------
    # SPLIT
    # -------------------------
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"]
    )

    # -------------------------
    # TOKENIZER
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = TextDataset(train_df, tokenizer, max_length)
    val_dataset = TextDataset(val_df, tokenizer, max_length)

    num_labels = len(df["label"].unique())

    # -------------------------
    # CLASS WEIGHTS
    # -------------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df["label"]),
        y=df["label"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    print("\nClass weights:", class_weights)

    # -------------------------
    # MODEL
    # -------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # -------------------------
    # TRAINING ARGS
    # -------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,   # MAX = 5
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        logging_steps=50,
        report_to="none"
    )

    # -------------------------
    # TRAINER (WITH EARLY STOPPING)
    # -------------------------
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2  # stop if no improvement after 1 epoch
            )
        ]
    )

    # -------------------------
    # TRAIN
    # -------------------------
    trainer.train()

    # BEST MODEL AUTO-LOADED
    result = trainer.evaluate()

    print("\n🏁 FINAL RESULT:", result)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n🏆 BEST MODEL SAVED AT:", output_dir)

    return output_dir, tokenizer

# -----------------------------
# PREDICT (BATCHED)
# -----------------------------
def predict(model_path, texts, max_length=128):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    return preds.tolist()