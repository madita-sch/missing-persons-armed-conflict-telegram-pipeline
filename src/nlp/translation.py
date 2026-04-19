import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"

# load ONCE (important)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def translate_texts(texts):
    # clean input
    texts = [str(t) if t is not None else "" for t in texts]

    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
        ).to(device)

        with torch.no_grad():
            translated = model.generate(**inputs)
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return results
