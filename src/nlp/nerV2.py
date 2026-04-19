from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CAMeL Tools imports (lazy + helpful error)
# ---------------------------------------------------------------------------
try:
    from camel_tools.ner import NERecognizer
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.utils.normalize import (
        normalize_alef_ar,
        normalize_alef_maksura_ar,
        normalize_teh_marbuta_ar,
    )
except ImportError:
    sys.stderr.write(
        "ERROR: camel-tools not installed.\n"
        "  pip install camel-tools\n"
        "  camel_data -i ner-arabert\n"
    )
    raise


# ---------------------------------------------------------------------------
# 2. NER — person & location via CAMeL Tools
# ---------------------------------------------------------------------------
def normalize_ar(text: str) -> str:
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    return text


def extract_entities(
    text: str, ner: NERecognizer
) -> Tuple[Optional[str], Optional[str]]:
    """Return (person_name, location) using BIO tags from CAMeL NER."""
    if not isinstance(text, str) or not text.strip():
        return None, None

    tokens = simple_word_tokenize(normalize_ar(text))
    if not tokens:
        return None, None

    tags = ner.predict_sentence(tokens)

    persons: List[str] = []
    locations: List[str] = []
    cur: List[str] = []
    cur_type: Optional[str] = None

    def flush():
        nonlocal cur, cur_type
        if cur and cur_type:
            chunk = " ".join(cur).strip()
            if cur_type == "PER":
                persons.append(chunk)
            elif cur_type == "LOC":
                locations.append(chunk)
        cur, cur_type = [], None

    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            flush()
            cur_type = tag[2:]
            cur = [tok]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur.append(tok)
        else:
            flush()
    flush()

    person = persons[0] if persons else None
    location = locations[0] if locations else None
    return person, location


# ---------------------------------------------------------------------------
# 3. Date extraction — rule-based (NER models rarely tag Arabic dates well)
# ---------------------------------------------------------------------------
DATE_PATTERNS = [
    # "خمس ايام" / "٣٠ يوم" / "30 يوم"
    r"(?:منذ\s+)?[\u0660-\u0669\d]+\s*(?:يوم|أيام|ايام|اسبوع|أسبوع|اسابيع|أسابيع|شهر|اشهر|أشهر|سنة|سنوات)",
    r"(?:خمس|ست|سبع|ثمان|تسع|عشر|اربع|أربع|ثلاث|اثنين|يوم|يومين)\s*(?:يوم|أيام|ايام|اسبوع|أسبوع|اسابيع|أسابيع|شهر|اشهر|أشهر|سنة|سنوات)",
    # explicit dates dd/mm/yyyy or yyyy-mm-dd
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}",
    r"\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}",
    # "اخر مرة ... " phrase
    r"(?:اخر|آخر)\s+مرة[^.،\n]{0,40}",
    # weekday / month names
    r"(?:يوم\s+)?(?:السبت|الاحد|الأحد|الاثنين|الإثنين|الثلاثاء|الاربعاء|الأربعاء|الخميس|الجمعة)",
    r"(?:يناير|فبراير|مارس|ابريل|أبريل|مايو|يونيو|يوليو|اغسطس|أغسطس|سبتمبر|اكتوبر|أكتوبر|نوفمبر|ديسمبر|كانون|تشرين|شباط|آذار|نيسان|أيار|حزيران|تموز|آب|أيلول)",
]
DATE_RE = re.compile("|".join(DATE_PATTERNS))


def extract_date(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    matches = DATE_RE.findall(text)
    # findall with alternations returns tuples sometimes — normalize:
    flat = []
    for m in matches:
        flat.append(m if isinstance(m, str) else next((g for g in m if g), ""))
    flat = [s.strip() for s in flat if s and s.strip()]
    if not flat:
        return None
    # Join unique snippets, preserving order
    seen = set()
    uniq = []
    for s in flat:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return " | ".join(uniq)



# =========================
# LOAD NER MODEL
# =========================
def load_ner_model():
    return NERecognizer.pretrained()


# =========================
# APPLY NER TO DF
# =========================
def apply_ner_to_df(df, ner, text_col="text_clean"):
    persons = []
    locations = []
    dates = []

    for text in df[text_col].fillna("").astype(str):
        p, l = extract_entities(text, ner)
        d = extract_date(text)
        persons.append(p)
        locations.append(l)
        dates.append(d)

    df["names"] = persons
    df["location"] = locations
    df["dates"] = dates

    return df