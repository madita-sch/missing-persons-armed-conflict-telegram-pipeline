import unicodedata
import re
import pandas as pd


# -------------------------
# CLEANING
# -------------------------
def normalize_arabic(text):
    text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[-ـ]+", "-", text)
    text = re.sub(r"[آأإ]", "ا", text)
    text = re.sub(r"[^\w\sء-ي]", " ", text)

    return text


def clean_text_column(df, text_col="text"):
    df = df.copy()
    df["text_clean"] = df[text_col].apply(normalize_arabic)
    return df


# -------------------------
# DEDUPLICATION
# -------------------------
def remove_duplicates(df, col="text_clean"):
    df = df.dropna(subset=[col])
    df = df.drop_duplicates(subset=[col])
    return df


# -------------------------
# SPAM FILTER
# -------------------------
def remove_spam(df, col="text_clean"):
    spam_keywords_ar = [
        "استثمار", "ربح", "اشتراك", "فوركس", "اكسب", "مال"
    ]

    pattern = "|".join(spam_keywords_ar)

    df = df[~df[col].str.contains(pattern, na=False)]
    return df


# -------------------------
# PIPELINE WRAPPER
# -------------------------
def preprocess_text_pipeline(df):
    df = clean_text_column(df)
    df = remove_duplicates(df)
    df = remove_spam(df)
    return df