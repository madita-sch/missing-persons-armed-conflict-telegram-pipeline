import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_excel("data/annotated_final.xlsx")

df = df[["text_clean", "label"]].copy()
df["label"] = df["label"].astype(int)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================================================
# 2. ARA BERT DATASET
# =========================================================
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


# =========================================================
# 3. TRAIN ARA BERT
# =========================================================
model_name = "aubmindlab/bert-base-arabert"

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = TextDataset(train_df, tokenizer)
test_dataset = TextDataset(test_df, tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(df["label"].unique())
)

training_args = TrainingArguments(
    output_dir="./arabert_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
trainer.save_model("./arabert_model")
tokenizer.save_pretrained("./arabert_model")


# =========================================================
# 4. ARA BERT PREDICTIONS
# =========================================================
def predict_arabert(model, tokenizer, texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                str(text),
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(out.logits, dim=1).item()
            preds.append(pred)

    return preds

df_test_sample = test_df.sample(n=10, random_state=42)
arabert_preds = predict_arabert(model, tokenizer, df_test_sample["text_clean"].tolist())


# =========================================================
# 5. LLM FEW-SHOT PREDICTION (PLACEHOLDER)
# =========================================================
# Replace this with OpenAI / API / local LLM call
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)


def llm_fewshot_predict(texts):
    predictions = []

    for text in texts:

        prompt = f"""
You are a classification system.

Task: Decide if the message is about a missing person.

Return ONLY:
missing or not_missing

Examples:
Text: person is missing in Gaza
Label: missing

Text: lost phone in street
Label: not_missing

Now classify:
Text: {text}
Label:
"""

        out = pipe(prompt, max_new_tokens=5, do_sample=False)[0]["generated_text"]

        # simple parsing
        if "missing" in out.lower():
            pred = 1
        else:
            pred = 0

        predictions.append(pred)

    return predictions

llm_preds = llm_fewshot_predict(df_test_sample["text_clean"].tolist())


# =========================================================
# 6. RESULTS DATAFRAME
# =========================================================
df_test_sample = df_test_sample.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
test_df["arabert_pred"] = arabert_preds
results_df = df_test_sample.copy()

results_df["llm_pred"] = llm_preds
results_df["arabert_pred"] = test_df.loc[df_test_sample.index, "arabert_pred"].values



# =========================================================
# 7. EVALUATION
# =========================================================
def evaluate(y_true, y_pred, name):
    print(f"\n===== {name} =====")
    print("F1 macro:", f1_score(y_true, y_pred, average="macro"))
    print("F1 weighted:", f1_score(y_true, y_pred, average="weighted"))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


evaluate(results_df["label"], results_df["arabert_pred"], "AraBERT")
evaluate(results_df["label"], results_df["llm_pred"], "LLM Few-shot")

def get_metrics(y_true, y_pred):
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "accuracy": accuracy_score(y_true, y_pred)
    }

results_summary = pd.DataFrame([
    {"model": "AraBERT", **get_metrics(results_df["label"], results_df["arabert_pred"])},
    {"model": "LLM Few-shot", **get_metrics(results_df["label"], results_df["llm_pred"])},
])
results_summary.to_csv("model_comparison_metrics.csv", index=False)



# =========================================================
# 8. FINAL OUTPUT TABLE
# =========================================================
print(results_df.head())

results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved: model_comparison_results.csv")