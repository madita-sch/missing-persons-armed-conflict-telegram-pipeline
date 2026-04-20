import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

device = model.device


# -------------------------------------------------------
# SAFE TRANSLATION PROMPT (CRITICAL)
# -------------------------------------------------------
def build_prompt(text):
    return f"""
You are a strict translation system.

Translate the following Arabic (Levantine dialect) text into natural English.

RULES:
- Do NOT summarize
- Do NOT add information
- Do NOT remove information
- Do NOT change names
- Do NOT change locations
- Do NOT change phone numbers
- Preserve meaning as closely as possible
- Keep structure similar to original

TEXT:
{text}

OUTPUT:
""".strip()


# -------------------------------------------------------
# TRANSLATION FUNCTION (BATCH SAFE)
# -------------------------------------------------------
def translate_texts(texts, batch_size=4, max_new_tokens=256):
    results = []

    # clean input
    texts = [str(t) if t is not None else "" for t in texts]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        prompts = [build_prompt(t) for t in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,   # VERY IMPORTANT (reduces hallucination)
                do_sample=False
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # post-clean: remove prompt repetition if model echoes it
        cleaned = []
        for d in decoded:
            if "OUTPUT:" in d:
                d = d.split("OUTPUT:")[-1].strip()
            cleaned.append(d)

        results.extend(cleaned)

    return results