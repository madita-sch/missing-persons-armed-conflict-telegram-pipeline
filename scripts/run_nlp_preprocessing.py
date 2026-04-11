import os
import pandas as pd

from src.preprocessing.telegram_loader import run_telegram_loader
from src.preprocessing.text_preprocessing import normalize_arabic


# -----------------------------
# CONFIG
# -----------------------------
API_ID = 31456951
API_HASH = "ecdfd54cf6be553d2ff005f657648ed8"
CHANNEL = "Gaza20249"

START_DATE = pd.Timestamp("2023-11-07", tz="UTC")
END_DATE = pd.Timestamp("2024-04-01", tz="UTC")   # small test window!

BASE_DIR = "data/nlp"
os.makedirs(BASE_DIR, exist_ok=True)

RAW_PATH = os.path.join(BASE_DIR, "telegram_raw.xlsx")
CLEAN_PATH = os.path.join(BASE_DIR, "telegram_clean.xlsx")
SAMPLE_PATH = os.path.join(BASE_DIR, "telegram_sample.xlsx")


# -----------------------------
# 1. LOAD TELEGRAM DATA
# -----------------------------
print("Fetching Telegram data...")

df = run_telegram_loader(
    API_ID,
    API_HASH,
    CHANNEL,
    START_DATE,
    END_DATE
)

print(f"Loaded {len(df)} messages")

df.to_excel(RAW_PATH, index=False)


# -----------------------------
# 2. CLEAN TEXT
# -----------------------------
print("Cleaning text...")

df = df[df["text"].notna()].copy()
df["text"] = df["text"].astype(str)

df["text_clean"] = df["text"].apply(normalize_arabic)

df = df.dropna(subset=["text_clean"])
df = df.drop_duplicates(subset=["text_clean"])

df.to_excel(CLEAN_PATH, index=False)


# -----------------------------
# 3. SMALL SAMPLE FOR NLP PIPELINE
# -----------------------------
print("Creating small sample...")

if len(df) == 0:
    print("No data found in date range!")
    df_sample = pd.DataFrame()
else:
    df_sample = df.sample(n=min(200, len(df)), random_state=42)

df_sample.to_excel(SAMPLE_PATH, index=False)

print("NLP preprocessing finished successfully.")