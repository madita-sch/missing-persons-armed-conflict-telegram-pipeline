import re

class RegexNER:
    def __init__(self):
        self.name_pattern = re.compile(r"([اأإآء-ي]{2,}(?:\s+[اأإآء-ي]{2,}){1,3})")
        self.loc_pattern = re.compile(r"(?:في|بمنطقة|شرق|غرب|شمال|جنوب)\s+([اأإآء-ي\s]{2,})")

    def extract(self, text):
        return {
            "names": self.name_pattern.findall(text),
            "locations": self.loc_pattern.findall(text)
        }
    

class TransformerNER:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def predict(self, text):
        # inference only
        return entities