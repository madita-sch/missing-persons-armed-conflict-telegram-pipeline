from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"


def load_translation_model():
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    return tokenizer, model


def translate_texts(texts):
    tokenizer, model = load_translation_model()

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)

    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]