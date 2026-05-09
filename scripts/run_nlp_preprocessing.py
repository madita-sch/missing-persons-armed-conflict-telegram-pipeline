import os
import pandas as pd

from src.preprocessing.telegram_loader import run_telegram_loader
from src.preprocessing.text_preprocessing import normalize_arabic
from dotenv import load_dotenv

# Load the environment
load_dotenv()

# Configuration for the Telegram API and data extraction
API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
print(API_ID)
print(API_HASH)
# Define Telegram channel
CHANNEL = "ALMAFKODEN"

# Define date range for data extraction
START_DATE = pd.Timestamp("2024-12-01", tz="UTC")
END_DATE = pd.Timestamp("2024-12-31", tz="UTC") 

# Create output directory
BASE_DIR = "data/text_ALMAFKODEN"
os.makedirs(BASE_DIR, exist_ok=True)

RAW_PATH = os.path.join(BASE_DIR, "telegram_raw.xlsx")
CLEAN_PATH = os.path.join(BASE_DIR, "telegram_clean.xlsx")
SAMPLE_PATH = os.path.join(BASE_DIR, "telegram_sample.xlsx")


# Load Telegram data
df = run_telegram_loader(
    API_ID,
    API_HASH,
    CHANNEL,
    START_DATE,
    END_DATE
)

print(f"Loaded {len(df)} messages")

df.to_excel(RAW_PATH, index=False)


# Clean text data
df = df[df["text"].notna()].copy()
df["text"] = df["text"].astype(str)

df["text_clean"] = df["text"].apply(normalize_arabic)

df = df.dropna(subset=["text_clean"])
df = df.drop_duplicates(subset=["text_clean"])

df.to_excel(CLEAN_PATH, index=False)


# Create small sample for tests
if len(df) == 0:
    print("No data found in date range!")
    df_sample = pd.DataFrame()
else:
    df_sample = df.sample(n=min(200, len(df)), random_state=42)

df_sample.to_excel(SAMPLE_PATH, index=False)