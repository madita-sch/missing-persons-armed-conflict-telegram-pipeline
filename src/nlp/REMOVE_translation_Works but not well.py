import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

MODEL_NAME = "inceptionai/jais-13b-chat"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# source language (approximation for Levantine Arabic)
tokenizer.src_lang = "arb_Arab"

def translate_texts(texts, batch_size=8):
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
            translated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn")
            )

        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        results.extend(decoded)

    return results