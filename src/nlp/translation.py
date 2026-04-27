# Import libraries
import time
import re
from typing import Optional
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load environment 
load_dotenv()

# Initialize Groq client and load model
client = Groq()
MODEL  = "llama-3.3-70b-versatile"

# Define System prompt for translation task
TRANSLATION_SYSTEM_PROMPT = """You are a professional Arabic to English translator specializing in Palestinian/Gaza dialect.
Translate the given Arabic text to English accurately and naturally.
- Preserve proper nouns (names of people, places) as-is or transliterate them
- Keep the meaning faithful to the original, including emotional tone
- If the input is empty or not Arabic, return an empty string
- Return ONLY the translated text, no explanation, no quotes"""

# Define single text translation
def translate_single(text: str, retries: int = 3, delay: float = 2.0) -> str:
    """Translate one Arabic text to English using Groq."""
    
    # Skip invalid or empty inputs
    if not isinstance(text, str) or not text.strip():
        return ""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                temperature=0.1,    # Low randomness to ensure consistent translation
                max_tokens=512,     # Sufficient for short Telegram messages
            )

            # Extract translated text from response
            return response.choices[0].message.content.strip()

        except Exception as e:
            # Handle API rate limites
            if "rate_limit" in str(e).lower() and attempt < retries - 1:
                print(f"Rate limit hit, waiting {delay * (attempt + 1)}s...")
                time.sleep(delay * (attempt + 1))
                continue

            # If API failure persists, return empty string and log error
            print(f"Translation failed for text: {e}")
            return ""

    return ""


# Define batch translation function
def translate_texts(
    texts: list[str],
    batch_size: int = 1,
    delay_between_calls: float = 0.5,
) -> list[str]:
    """
    Translate a list of Arabic texts to English.
    Returns a list of translated strings in the same order.
    batch_size is kept at 1 — Llama translation quality is better one-by-one
    since batching can cause the model to mix up which text maps to which.
    """
    results = []
    total = len(texts)

    for i, text in enumerate(texts):
        print(f"  Translating {i+1}/{total}", end="\r")
        translated = translate_single(text)
        results.append(translated)
        time.sleep(delay_between_calls)

    print()
    return results


# Apply translation to the selected dataframe columns
def apply_translation_to_df(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    extra_cols: Optional[list[str]] = None,
    delay_between_calls: float = 0.5,
) -> pd.DataFrame:
    """
    Translate text_col and optionally extra columns (names, location, dates).
    Adds _en suffix columns to the dataframe.
    Only translates rows where text_col is non-empty.
    """
    df = df.copy()

    # Combine main text column with optional additional columns
    cols_to_translate = [text_col] + (extra_cols or [])

    for col in cols_to_translate:

        # Skip columns that don't exist
        if col not in df.columns:
            print(f"Column '{col}' not found, skipping")
            continue

        print(f"Translating column: {col}")

        # Prepare text input handling missing or non-string values
        texts = df[col].fillna("").astype(str).tolist()

        # Apply translation 
        translated = translate_texts(texts, delay_between_calls=delay_between_calls)
        
        # Store results in new column with _en suffix
        df[f"{col}_en"] = translated

    return df
