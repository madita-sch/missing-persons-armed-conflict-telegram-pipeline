import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load the environment to get Groq API Key
load_dotenv()

# Get DB connection string from environment
DB_URI = os.getenv("DB_URI")
if DB_URI is None:
    raise ValueError("DB_URI not found in environment variables")
# Connect to PostgreSQL database
engine = create_engine(DB_URI)

# Load the df
df = pd.read_csv("outputs/nlp_results.csv")

# Create cases
cases_df = df[df["cluster_id"] != -1][["cluster_id"]].drop_duplicates()
cases_df.to_sql("cases", engine, if_exists="append", index=False)

# Insert messages
messages_df = df[[
    "id", "date", "text", "text_clean",
    "text_clean_anon", "text_clean_en",
    "views", "forwards", "reactions",
    "is_missing", "cluster_id"
]].copy()

messages_df.rename(columns={
    "id": "message_id",
    "text_clean_en": "text_en"
}, inplace=True)

# Optional: add Telegram link
messages_df["telegram_link"] = messages_df["message_id"].apply(
    lambda x: f"https://t.me/GAZA20249/{x}"
)

messages_df.to_sql("messages", engine, if_exists="append", index=False)

# Split muliple names 
def split_names(x):
    if pd.isna(x):
        return []
    return [n.strip() for n in str(x).split(";") if n.strip()]

rows = []

for _, row in df.iterrows():
    for name in split_names(row["names"]):
        rows.append({
            "name_ar": name,
            "name_en": row.get("names_en"),
            "cluster_id": row["cluster_id"]
        })

names_df = pd.DataFrame(rows)
names_df.drop_duplicates(inplace=True)

names_df.to_sql("names", engine, if_exists="append", index=False)

# Insert locations 
loc_df = df[["location", "location_en", "cluster_id"]].dropna()

loc_df.rename(columns={
    "location": "location_ar"
}, inplace=True)

loc_df.drop_duplicates(inplace=True)

loc_df.to_sql("locations", engine, if_exists="append", index=False)

# Link cases to messages
# First get case_id mapping
cases_db = pd.read_sql("SELECT case_id, cluster_id FROM cases", engine)

merged = df.merge(cases_db, on="cluster_id")

case_messages_df = merged[["case_id", "id"]].rename(
    columns={"id": "message_id"}
)

case_messages_df.to_sql("case_messages", engine, if_exists="append", index=False)

