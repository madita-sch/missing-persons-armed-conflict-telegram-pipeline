from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"


def load_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def extract_entities(texts):
    ner_pipeline = load_ner_pipeline()

    results = []
    for text in texts:
        entities = ner_pipeline(text)
        results.append(entities)

    return results