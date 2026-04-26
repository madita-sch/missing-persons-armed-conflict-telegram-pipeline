# import libraries
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

# Set up utilities to fix random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Define Dataset class
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text and the lable
        text = str(self.data.loc[idx, "text_clean"])
        label = int(self.data.loc[idx, "label"])

        # Tokenize the text, converting words into input_ids
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Define output format for the Trainer
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to predicted class labels
    preds = np.argmax(logits, axis=1)

    # Compute Accuracy and F1 scores
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# Define Custom Trainer to handle class weights
# To override the default Hugging Face Trainer, subclass with class weights in the loss function.
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
        # Get labels from inputs
        labels = inputs.get("labels")

        # Forward pass through the model
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Apply weighted CrossEntropyLoss 
        # Majority class (no missing message) has lower weight
        # Less frequent class gets higher penalty if misclassified
        loss_fct = CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        # Compute the loss 
        loss = loss_fct(logits, labels)

        # Return loss to trainer
        return (loss, outputs) if return_outputs else loss


# Train function
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
):

    # Set seed for reproducibility
    set_seed(seed)

    # Load the annotated training dataset
    df = pd.read_excel(data_path)
    df = df[["text_clean", "label"]].copy()
    df["label"] = df["label"].astype(int)

    # Train-test split 
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"]
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create datsets for Trainer
    train_dataset = TextDataset(train_df, tokenizer, max_length)
    val_dataset = TextDataset(val_df, tokenizer, max_length)

    # Define number of lables
    num_labels = len(df["label"].unique())

    # Compute class weights to focus more on rare class (missing message) and less on majority class (non-missing)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df["label"]),
        y=df["label"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    print("\nClass weights:", class_weights)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Configuration of training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,

        eval_strategy="epoch", #Evaluation at the end of each epoch
        save_strategy="epoch", #Save model at the end of each epoch

        load_best_model_at_end=True, # Automatically load the best model (based on eval metric)
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        logging_steps=50,
        report_to="none"
    )

    # Build trainer with our custom WeightedTrainer
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
                early_stopping_patience=2  # stop if no improvement for 2 evaluations
            )
        ]
    )

    # Train
    trainer.train()

    # Evaluate the best model on the validation set
    result = trainer.evaluate()


    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Return output
    return output_dir, tokenizer

# Define prediction function to be used in the pipeline
def predict(model_path, texts, max_length=128):

    # Move model to GPU if available for faster classification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    # tokenize the input texts and prepare for model input
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Disable gradients for faster inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Convert logits to predicted class labels (0 or 1)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    # Return labels (1 for potential missing message, 0 for non-missing)
    return preds.tolist()