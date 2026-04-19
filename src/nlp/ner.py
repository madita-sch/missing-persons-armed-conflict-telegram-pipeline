import re
import unicodedata
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL NAME
# =========================
NER_MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"

# =========================
# LOAD MODEL
# =========================
tokenizer_ner = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
ner_model.to(device)
ner_model.eval()

ner_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1,
)

# =========================
# NORMALIZATION
# =========================
ARABIC_NORMALIZE_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ة": "ه",
    "ى": "ي",
}

DATE_PATTERNS = [
    r"(?:منذ\s+)?[\u0660-\u0669\d]+\s*(?:يوم|أيام|ايام|اسبوع|أسبوع|اسابيع|أسابيع|شهر|اشهر|أشهر|سنة|سنوات)",
    r"(?:خمس|ست|سبع|ثمان|تسع|عشر|اربع|أربع|ثلاث|اثنين|يوم|يومين)\s*(?:يوم|أيام|ايام|اسبوع|أسبوع|اسابيع|أسابيع|شهر|اشهر|أشهر|سنة|سنوات)",
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}",
    r"\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}",
    r"(?:اخر|آخر)\s+مرة[^.،\n]{0,40}",
    r"(?:يوم\s+)?(?:السبت|الاحد|الأحد|الاثنين|الإثنين|الثلاثاء|الاربع|الأربع|الخميس|الجمعة)",
    r"(?:يناير|فبراير|مارس|ابريل|أبريل|مايو|يونيو|يوليو|اغسطس|أغسطس|سبتمبر|اكتوبر|أكتوبر|نوفمبر|ديسمبر|كانون|تشرين|شباط|آذار|نيسان|أيار|حزيران|تموز|آب|أيلول)",
]
DATE_RE = re.compile("|".join(DATE_PATTERNS))

LOCATION_FALLBACK_PATTERNS = [
    r"(?:في|فى|ب(?:ال|ـ)?|بال|عند|من|إلى|الى)\s+((?:ال)?[\u0621-\u064A\u0660-\u0669]{3,}(?:\s+(?:ال)?[\u0621-\u064A\u0660-\u0669]{2,}){0,3})",
    r"(?:منطقة|حي|شارع)\s+((?:ال)?[\u0621-\u064A\u0660-\u0669]{3,}(?:\s+(?:ال)?[\u0621-\u064A\u0660-\u0669]{2,}){0,3})",
]
LOCATION_FALLBACK_RE = [re.compile(p) for p in LOCATION_FALLBACK_PATTERNS]


def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, tgt in ARABIC_NORMALIZE_MAP.items():
        text = text.replace(src, tgt)
    return text


def predict_ner(df, text_col: str = "text_clean", batch_size: int = 16):
    all_entities = []
    texts = [normalize_arabic(str(t)) for t in df[text_col].fillna("")]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_results = ner_pipeline(batch_texts)
        if isinstance(batch_results, dict):
            batch_results = [batch_results]

        for res in batch_results:
            entities = []
            for item in res:
                label = item.get("entity_group") or item.get("entity")
                if not label:
                    continue
                label = label[2:] if label.startswith(("B-", "I-")) else label
                word = item.get("word", "").replace("##", "").strip()
                if word:
                    entities.append((word, label))
            all_entities.append(entities)

    return all_entities


def extract_date(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    normalized = normalize_arabic(text)
    matches = DATE_RE.findall(normalized)
    flat = []
    for m in matches:
        flat.append(m if isinstance(m, str) else next((g for g in m if g), ""))
    flat = [s.strip() for s in flat if s and s.strip()]
    if not flat:
        return None
    seen = set()
    uniq = []
    for s in flat:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return " | ".join(uniq)


def extract_location_fallback(text: str) -> Optional[str]:
    normalized = normalize_arabic(text)
    for pattern in LOCATION_FALLBACK_RE:
        match = pattern.search(normalized)
        if match:
            location = match.group(1).strip()
            tokens = [t for t in location.split() if len(t) >= 2]
            if tokens:
                return " ".join(tokens)
    return None


def extract_entities_arabic(entity_pairs):
    label_map = {
        "PER": "names",
        "PERS": "names",
        "LOC": "location",
        "ORG": "organization",
        "DATE": "dates",
        "MISC": "misc",
    }
    fragments = {v: [] for v in label_map.values()}
    current_tokens = []
    current_label = None

    def flush():
        nonlocal current_tokens, current_label
        if current_tokens and current_label in label_map:
            text = " ".join(current_tokens).strip()
            if text:
                fragments[label_map[current_label]].append(text)
        current_tokens = []
        current_label = None

    for token, label in entity_pairs:
        token = token.replace("##", "").strip()
        if not token:
            continue
        if label not in label_map:
            flush()
            continue
        if current_label == label:
            current_tokens.append(token)
        else:
            flush()
            current_tokens = [token]
            current_label = label
    flush()
    return {k: "; ".join(v) for k, v in fragments.items()}


def load_ner_model():
    return tokenizer_ner, ner_model, ner_pipeline


def apply_ner_to_df(df, text_col: str = "text_clean"):
    entities_list = predict_ner(df, text_col)
    df["ner_extracted"] = entities_list
    extracted = [extract_entities_arabic(e) for e in entities_list]
    df_entities = pd.DataFrame(extracted, index=df.index)
    df["names"] = df_entities.get("names", "")
    df["location"] = df_entities.get("location", "")
    df["dates"] = df_entities.get("dates", "")
    fallback_locations = []
    for row_idx, loc in df["location"].items():
        if not isinstance(loc, str) or not loc.strip():
            fallback_locations.append(extract_location_fallback(df.at[row_idx, text_col]))
        else:
            fallback_locations.append(loc)
    df["location"] = [f if f else l for f, l in zip(fallback_locations, df["location"])]
    df["dates"] = df["dates"].fillna("")
    date_fallback = df[text_col].apply(lambda x: extract_date(str(x)) or "")
    df["dates"] = df["dates"].where(df["dates"].astype(bool), date_fallback)
    return df
