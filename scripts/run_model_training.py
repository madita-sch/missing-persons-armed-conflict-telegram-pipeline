# Import libraries 
from src.nlp.classification import train_model

# Run the training script
if __name__ == "__main__":
    best_model_path, tokenizer = train_model(
        data_path="data/annotated_final.xlsx",
        output_dir="./model_output",
        epochs=5,
        sample_size=None
    )

    print("\n FINAL MODEL PATH:", best_model_path)
    print(type(best_model_path))