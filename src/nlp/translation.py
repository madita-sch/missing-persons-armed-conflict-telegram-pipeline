from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-ar-en"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)