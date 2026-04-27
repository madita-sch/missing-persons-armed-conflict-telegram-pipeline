# Import libraries 
import re
import json
import time
import unicodedata
from typing import Optional

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load the environment to get Groq API Key
load_dotenv()

# Initialize Groq client and define model
client = Groq()  # picks up GROQ_API_KEY from .env file
MODEL  = "llama-3.3-70b-versatile"

# Create System Prompt for the NER task
SYSTEM_PROMPT = """You are an Arabic NLP specialist analyzing missing persons appeals from Gaza.
These messages are written in Palestinian/Gaza dialect Arabic mixed with MSA.
Extract ONLY the missing person's information — not the sender, not contact persons.
There may be more than one missing person in a single message.

Return JSON only, no explanation, no markdown, no code block:
{
  "missing_names": ["full name 1", "full name 2"],
  "location": "last known location or null",
  "date": "date last seen or null",
  "age": "age if mentioned or null"
}

Rules:
- missing_names is always a list, even if only one person
- Never include the sender's name or names of people providing contact info
- Ignore names that appear after رقمي / للتواصل / اتصل / واتساب
- Location can be a range if person was travelling between two places
- Date can be descriptive if no exact date is given (e.g. 'first month of the war')
- If nothing found, return empty list [] for names and null for others"""

# Normalize Arabic to reduce spelling variation
ARABIC_NORMALIZE_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ة": "ه",
    "ى": "ي",
}

def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, tgt in ARABIC_NORMALIZE_MAP.items():
        text = text.replace(src, tgt)
    return text

# Regex fallback extraction, only used if LLM output is missing/uncertain
# Date detection patterns
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

# Phone number detection patterns
PHONE_RE = re.compile(
    r'(?:\+|00)?(?:970|972|971)?[\s\-]?(?:5[0-9]|0[5-9])[0-9\s\-]{7,12}'
)

# Location detection patterns
LOCATION_FALLBACK_PATTERNS = [
    r"(?:في|فى|ب(?:ال|ـ)?|بال|عند|من|إلى|الى)\s+((?:ال)?[\u0621-\u064A\u0660-\u0669]{3,}(?:\s+(?:ال)?[\u0621-\u064A\u0660-\u0669]{2,}){0,3})",
    r"(?:منطقة|حي|شارع)\s+((?:ال)?[\u0621-\u064A\u0660-\u0669]{3,}(?:\s+(?:ال)?[\u0621-\u064A\u0660-\u0669]{2,}){0,3})",
]
LOCATION_FALLBACK_RE = [re.compile(p) for p in LOCATION_FALLBACK_PATTERNS]

# Define regex functions for fallback extraction
def extract_date_regex(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    normalized = normalize_arabic(text)
    matches = DATE_RE.findall(normalized)
    flat = []
    for m in matches:
        flat.append(m if isinstance(m, str) else next((g for g in m if g), ""))
    flat = [s.strip() for s in flat if s.strip()]
    seen: set = set()
    uniq = []
    for s in flat:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return " | ".join(uniq) if uniq else None


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


def extract_phones(text: str) -> list[str]:
    return [p.strip() for p in PHONE_RE.findall(text or "")]


# Groq NER function 
def call_groq_ner(text: str, retries: int = 3, delay: float = 2.0) -> dict:
    """
    Call Groq with the NER prompt. Returns a dict with keys:
      missing_names (list), location (str|None), date (str|None), age (str|None)
    Falls back to empty result on failure.
    """
    empty = {"missing_names": [], "location": None, "date": None, "age": None}

    if not isinstance(text, str) or not text.strip():
        return empty

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                temperature=0, 
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if model adds them anyway
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)

            return {
                "missing_names": parsed.get("missing_names") or [],
                "location":      parsed.get("location"),
                "date":          parsed.get("date"),
                "age":           parsed.get("age"),
            }

        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(delay)
            continue

        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < retries - 1:
                print(f"Rate limit hit, waiting {delay * (attempt + 1)}s...")
                time.sleep(delay * (attempt + 1))
                continue
            print(f"Groq call failed: {e}")
            return empty

    return empty


# Define batch processing function to run Groq NER on every row
def predict_ner_groq(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    delay_between_calls: float = 0.5,
) -> list[dict]:
    """
    Run Groq NER on every row. Returns a list of dicts (one per row).
    delay_between_calls: seconds between API calls.
    Groq free tier allows ~30 requests/minute for this model.
    """
    results = []
    texts = df[text_col].fillna("").astype(str).tolist()

    for i, text in enumerate(texts):
        print(f"  NER {i+1}/{len(texts)}", end="\r")
        result = call_groq_ner(text)
        results.append(result)
        time.sleep(delay_between_calls)

    print()
    return results


# Define Apply NER function, incl. call to Groq (LLM preferred) and regex fallbacks
def apply_ner_to_df(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    df = df.copy()

    print(f"  Calling Groq NER on {len(df)} rows...")
    ner_results = predict_ner_groq(df, text_col)

    names_list    = []
    location_list = []
    dates_list    = []
    age_list      = []

    for result, idx in zip(ner_results, df.index):
        # Names
        names = result.get("missing_names") or []
        names_list.append("; ".join(names) if names else "")

        # Location: LLM first, regex fallback
        loc = result.get("location")
        if not loc:
            loc = extract_location_fallback(str(df.at[idx, text_col])) or ""
        location_list.append(loc or "")

        # Date: LLM first, regex fallback
        date = result.get("date")
        if not date:
            date = extract_date_regex(str(df.at[idx, text_col])) or ""
        dates_list.append(date or "")

        # Age
        age_list.append(str(result.get("age") or ""))

    df["names"]    = names_list
    df["location"] = location_list
    df["dates"]    = dates_list
    df["age"]      = age_list

    return df