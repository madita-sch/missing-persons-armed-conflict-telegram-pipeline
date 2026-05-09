# Import libraries
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import numpy as np
from sqlalchemy import text

# Load environment
load_dotenv()

# Create database engine
DB_URI = os.getenv("DB_URI")
if DB_URI is None:
    raise ValueError("DB_URI not found in environment variables")

engine = create_engine(DB_URI)

# Load dataset
df = pd.read_csv("outputs/pred_Gaza20249.csv")

# Connect to the database and populate tables
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE extracted_entities RESTART IDENTITY CASCADE"))
    conn.execute(text("TRUNCATE TABLE messages RESTART IDENTITY CASCADE"))
    conn.execute(text("TRUNCATE TABLE cases RESTART IDENTITY CASCADE"))

# Cases table (cluster_id -> case_id)
agg = (df[df["cluster_id"] != -1]
       .groupby("cluster_id")
       .agg(
           name_ar=("names","first"),
           name_en=("names_en","first"),
           location_ar=("location","first"),
           location_en=("location_en","first"),
           dates_ar=("dates","first"),
           dates_en=("dates_en","first"),
           age=("age","first"),
           message_count=("id","count"),
           first_seen_at=("date","min"),
           last_seen_at=("date","max")
       )
       .reset_index()
       .rename(columns={"cluster_id":"case_id"}))

agg.to_sql("cases", engine, if_exists="append", index=False, method="multi")

# Messages table
messages_df = df[[
    "id", "cluster_id", "date", "text", "text_clean",
    "is_missing", "views", "forwards", "reactions"
]].copy()

messages_df.rename(columns={
    "id": "message_id",
    "cluster_id": "case_id",
    "date": "posted_at",
    "text": "text_raw",
}, inplace=True)

# -1 (unassigned cluster) -> NULL so the FK to cases.case_id is satisfied
messages_df["case_id"] = messages_df["case_id"].replace(-1, np.nan)
messages_df["case_id"] = messages_df["case_id"].astype("Int64")  # nullable int

messages_df["is_missing"] = messages_df["is_missing"].fillna(0).astype(bool)
messages_df["posted_at"]  = pd.to_datetime(messages_df["posted_at"], errors="coerce")

for col in ["views", "forwards", "reactions"]:
    messages_df[col] = pd.to_numeric(messages_df[col], errors="coerce").astype("Int64")

messages_df.to_sql("messages", engine, if_exists="append",
                   index=False, method="multi", chunksize=500)


# Translations table
translations_df = df[[
    "id",
    "text_clean_en",
    "names_en",
    "location_en",
    "dates_en"
]].copy()

translations_df.rename(columns={"id": "message_id"}, inplace=True)

translations_df.to_sql(
    "message_translations",
    engine,
    if_exists="append",
    index=False,
    method="multi"
)

# Pseudonymized text table
anon_df = df[[
    "id",
    "text_clean_anon"
]].copy()

anon_df.rename(columns={"id": "message_id"}, inplace=True)

anon_df.to_sql(
    "message_anonymized",
    engine,
    if_exists="append",
    index=False,
    method="multi"
)

# Extracted entities table
def split_entities(x):
    if pd.isna(x):
        return []
    return [i.strip() for i in str(x).split(";") if i.strip()]

rows = []

for _, row in df.iterrows():
    mid = row["id"]
    case_id = row["cluster_id"] if row["cluster_id"] != -1 else None

    for kind, col_ar, col_en in [
        ("name", "names", "names_en"),
        ("location", "location", "location_en"),
        ("date", "dates", "dates_en"),
        ("age", "age", None),
    ]:
        ars = split_entities(row.get(col_ar))
        ens = split_entities(row.get(col_en)) if col_en else []

        for i, ar in enumerate(ars):
            en = ens[i] if i < len(ens) else None
            rows.append((mid, case_id, kind, ar, en))

        for j in range(len(ars), len(ens)):
            rows.append((mid, case_id, kind, None, ens[j]))

entities_df = pd.DataFrame(rows, columns=[
    "message_id", "case_id", "kind", "value_ar", "value_en"
])

entities_df.to_sql(
    "extracted_entities",
    engine,
    if_exists="append",
    index=False,
    method="multi"
)

print("Database successfully populated")