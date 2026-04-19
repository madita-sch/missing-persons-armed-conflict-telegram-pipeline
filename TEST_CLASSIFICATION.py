import torch
import pandas as pd
import numpy as np
import random

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
    TrainerCallback
)

# =====================================================
# SEED
# =====================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================
# DATASET
# =====================================================
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

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# =====================================================
# METRICS
# =====================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# =====================================================
# CALLBACK (BEST EPOCH TRACKER)
# =====================================================
class BestEpochCallback(TrainerCallback):
    def __init__(self):
        self.best_f1 = -1
        self.best_epoch = -1

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        f1 = metrics.get("eval_f1_macro")

        if f1 is not None:
            print(f"📊 Epoch {state.epoch:.2f} | F1_macro = {f1}")

            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_epoch = state.epoch


# =====================================================
# WEIGHTED TRAINER
# =====================================================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# =====================================================
# TRAIN ONE MODEL
# =====================================================
def train_one_model(model_name, train_dataset, val_dataset,
                    num_labels, output_dir,
                    class_weights=None, weighted=False):

    print("\n" + "="*60)
    print("TRAINING:", "WEIGHTED" if weighted else "NON-WEIGHTED")
    print("="*60)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    callback = BestEpochCallback()

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,

        eval_strategy="epoch",   # ✅ FIXED (NOT eval_strategy)
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        logging_steps=50,
        report_to="none"
    )

    if weighted:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

    trainer.add_callback(callback)

    trainer.train()

    result = trainer.evaluate()

    print("\n🏁 FINAL RESULT:", result)
    print("🏆 BEST EPOCH:", callback.best_epoch)
    print("🏆 BEST F1:", callback.best_f1)

    trainer.save_model(output_dir)

    return result, callback.best_epoch, callback.best_f1


# =====================================================
# PIPELINE
# =====================================================
def run_debug_pipeline(data_path="data/annotated_final.xlsx"):

    set_seed(42)

    df = pd.read_excel(data_path)
    df = df[["text_clean", "label"]].copy()
    df["label"] = df["label"].astype(int)

    sample_size = 200  # or 200 for faster debugging
    df = df.sample(n=min(sample_size, len(df)), random_state=42)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    model_name = "aubmindlab/bert-base-arabertv2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = TextDataset(train_df, tokenizer)
    val_dataset = TextDataset(val_df, tokenizer)

    num_labels = len(df["label"].unique())

    # CLASS WEIGHTS
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df["label"]),
        y=df["label"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    print("\nClass weights:", class_weights)

    # =====================================================
    # NON-WEIGHTED
    # =====================================================
    no_w = train_one_model(
        model_name,
        train_dataset,
        val_dataset,
        num_labels,
        "./debug_no_weights",
        weighted=False
    )

    # =====================================================
    # WEIGHTED
    # =====================================================
    w = train_one_model(
        model_name,
        train_dataset,
        val_dataset,
        num_labels,
        "./debug_weighted",
        class_weights=class_weights,
        weighted=True
    )

    # =====================================================
    # COMPARE
    # =====================================================
    comparison = pd.DataFrame([
        {
            "model": "no_weights",
            "f1": no_w[0]["eval_f1_macro"],
            "best_epoch": no_w[1]
        },
        {
            "model": "with_weights",
            "f1": w[0]["eval_f1_macro"],
            "best_epoch": w[1]
        }
    ])

    print("\n📊 FINAL COMPARISON")
    print(comparison)

    comparison.to_csv("debug_model_comparison.csv", index=False)

    best = comparison.sort_values("f1", ascending=False).iloc[0]["model"]

    print("\n🏆 BEST MODEL:", best)

    return comparison


if __name__ == "__main__":
    run_debug_pipeline()