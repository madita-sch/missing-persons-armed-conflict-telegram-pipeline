# Missing Persons — PostgreSQL Build

Case-centric relational schema for the Telegram missing-persons NLP pipeline.
Target: **PostgreSQL 18.x**.

## Files

| File | Purpose |
|---|---|
| `schema.sql`        | DDL: tables, enums, indexes, view. Drops & recreates everything. |
| `generate_seed.py`  | Reads your CSV and emits `seed.sql` (INSERT statements). |
| `seed.sql`          | Sample seed built from 5 example rows (replace with your own). |

## Data model

```
cases (case_id ⇐ cluster_id)
  ├── messages            (message_id ⇐ id)        many-to-one → cases
  │     ├── message_translations    (1-to-1)
  │     ├── message_anonymized      (1-to-1)
  │     └── extracted_entities      (1-to-many: name/location/date/age)
  └── view: v_case_page   (case + all messages as JSON array)
```

- **`case_id` = `cluster_id`** from the NLP output. Rows where `cluster_id = -1`
  (no cluster) become messages with `case_id IS NULL` and are **not** turned
  into a case row.
- **`message_id` = `id`** from the CSV (first column).
- Each case stores both Arabic and English versions of name, location, dates, age.
- Per-message English text and PII-anonymized text live in their own tables to
  keep `messages` lean.
- `extracted_entities` normalizes multi-value fields (e.g. `"name1; name2"`)
  into one row per entity, keyed to both the message and the case.

## How to use (Windows + PostgreSQL 18.3)

1. **Create the database & schema** (psql):
   ```bash
   psql -U postgres -c "CREATE DATABASE missing_persons;"
   psql -U postgres -d missing_persons -f schema.sql
   ```

2. **Generate `seed.sql` from your CSV**:
   ```bash
   python generate_seed.py ^
     "C:\Users\Lenovo\missing-persons-armed-conflict-telegram-pipeline\outputs\nlp_results - Copy.csv" ^
     seed.sql
   ```

3. **Load it**:
   ```bash
   psql -U postgres -d missing_persons -f seed.sql
   ```

## Querying a case page

```sql
-- Full case page (case header + all messages as JSON)
SELECT * FROM v_case_page WHERE case_id = 2;

-- All cases sorted by activity
SELECT case_id, name_en, message_count, last_seen_at
FROM cases ORDER BY message_count DESC;

-- Every entity attached to a case
SELECT kind, value_ar, value_en
FROM extracted_entities WHERE case_id = 1;

-- Messages without any case (cluster_id = -1)
SELECT message_id, posted_at, text_clean
FROM messages WHERE case_id IS NULL;
```

## Expected CSV columns

The generator expects this header (order does not matter, names do):

```
id, date, text, views, forwards, reactions, text_clean, is_missing,
names, location, dates, age, text_clean_en, names_en, location_en,
dates_en, clean, cluster_id, text_clean_anon
```

Empty cells, `nan`, or `-1` (for `cluster_id`) are handled as NULL.
Multi-value entity fields are split on `;`, `,`, `/`, or newline.

## Re-running

`schema.sql` starts with `DROP TABLE ... CASCADE`, so it is safe to re-run.
After re-running it, regenerate and reload `seed.sql`.
