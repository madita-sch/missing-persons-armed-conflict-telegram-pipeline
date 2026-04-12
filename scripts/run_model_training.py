from src.nlp.classification import train_model

train_model(
    "data/annotated_final.xlsx",
    "models/sequence_classifier"
)