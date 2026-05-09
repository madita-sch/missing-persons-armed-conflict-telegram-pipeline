#!/usr/bin/env python3
"""
Generate seed.sql from the NLP pipeline CSV.

Usage:
    python generate_seed.py C:/Users/Lenovo/missing-persons-armed-conflict-telegram-pipeline/outputs/nlp_results - Copy.csv path/to/seed.sql

Expected columns (header row):
    id, date, text, views, forwards, reactions, text_clean, is_missing,
    names, location, dates, age, text_clean_en, names_en, location_en,
    dates_en, clean, cluster_id, text_clean_anon
"""
# Import libraries
import csv, sys, re
from pathlib import Path

if len(sys.argv) != 3:
    print(__doc__); sys.exit(1)

src, dst = Path(sys.argv[1]), Path(sys.argv[2])

def q(v):
    """SQL-quote a value. None/empty -> NULL."""
    if v is None: return "NULL"
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"): return "NULL"
    return "'" + s.replace("'", "''") + "'"

def qi(v):
    if v is None: return "NULL"
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"): return "NULL"
    try: return str(int(float(s)))
    except: return "NULL"

def qb(v):
    s = str(v).strip()
    if s in ("1","true","True","TRUE"): return "TRUE"
    if s in ("0","false","False","FALSE"): return "FALSE"
    return "FALSE"

def qts(v):
    """Best-effort timestamp; if blank, NULL."""
    if v is None: return "NULL"
    s = str(v).strip()
    if not s or s.lower() in ("nan","none"): return "NULL"
    return "'" + s.replace("'", "''") + "'::timestamptz"

def split_multi(v):
    """Split a multi-value entity field on ; , / and newline."""
    if v is None: return []
    s = str(v).strip()
    if not s or s.lower() in ("nan","none","-1"): return []
    parts = re.split(r"[;,/\n]+", s)
    return [p.strip() for p in parts if p.strip()]

cases = {}        # cluster_id -> dict of latest non-empty fields
messages = []     # raw rows
entities = []     # (msg_id, case_id, kind, value_ar, value_en)

with src.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            mid = int(float(row["id"]))
        except:
            continue
        try:
            cid = int(float(row.get("cluster_id","")))
        except:
            cid = -1
        case_id = cid if cid >= 0 else None

        if case_id is not None:
            c = cases.setdefault(case_id, {
                "name_ar":None,"name_en":None,"location_ar":None,"location_en":None,
                "dates_ar":None,"dates_en":None,"age":None,
                "first":None,"last":None,"count":0,
            })
            for k_csv,k_db in [("names","name_ar"),("names_en","name_en"),
                               ("location","location_ar"),("location_en","location_en"),
                               ("dates","dates_ar"),("dates_en","dates_en"),
                               ("age","age")]:
                v = (row.get(k_csv) or "").strip()
                if v and v.lower() not in ("nan","none","-1") and not c[k_db]:
                    c[k_db] = v
            d = (row.get("date") or "").strip()
            if d:
                if not c["first"] or d < c["first"]: c["first"] = d
                if not c["last"]  or d > c["last"]:  c["last"]  = d
            c["count"] += 1

        messages.append({
            "id": mid, "case_id": case_id,
            "date": row.get("date",""),
            "text": row.get("text",""),
            "text_clean": row.get("text_clean",""),
            "is_missing": row.get("is_missing","0"),
            "views": row.get("views",""),
            "forwards": row.get("forwards",""),
            "reactions": row.get("reactions",""),
            "text_clean_en": row.get("text_clean_en",""),
            "names_en": row.get("names_en",""),
            "location_en": row.get("location_en",""),
            "dates_en": row.get("dates_en",""),
            "clean": row.get("clean",""),
            "text_clean_anon": row.get("text_clean_anon",""),
        })

        # extracted entities (zip Arabic + English where possible)
        for kind, ar_col, en_col in [
            ("name","names","names_en"),
            ("location","location","location_en"),
            ("date","dates","dates_en"),
            ("age","age",None),
        ]:
            ars = split_multi(row.get(ar_col,""))
            ens = split_multi(row.get(en_col,"")) if en_col else []
            for i,a in enumerate(ars):
                e = ens[i] if i < len(ens) else None
                entities.append((mid, case_id, kind, a, e))
            # English-only leftovers
            for j in range(len(ars), len(ens)):
                entities.append((mid, case_id, kind, None, ens[j]))

with dst.open("w", encoding="utf-8") as out:
    out.write("-- Auto-generated seed data\nBEGIN;\n\n")

    out.write("-- Cases\n")
    for cid,c in sorted(cases.items()):
        out.write(
            f"INSERT INTO cases (case_id,name_ar,name_en,location_ar,location_en,"
            f"dates_ar,dates_en,age,message_count,first_seen_at,last_seen_at) VALUES ("
            f"{cid},{q(c['name_ar'])},{q(c['name_en'])},{q(c['location_ar'])},"
            f"{q(c['location_en'])},{q(c['dates_ar'])},{q(c['dates_en'])},{q(c['age'])},"
            f"{c['count']},{qts(c['first'])},{qts(c['last'])});\n"
        )

    out.write("\n-- Messages\n")
    for m in messages:
        out.write(
            f"INSERT INTO messages (message_id,case_id,posted_at,text_raw,text_clean,"
            f"is_missing,views,forwards,reactions) VALUES ("
            f"{m['id']},"
            f"{m['case_id'] if m['case_id'] is not None else 'NULL'},"
            f"{qts(m['date'])},{q(m['text'])},{q(m['text_clean'])},"
            f"{qb(m['is_missing'])},{qi(m['views'])},{qi(m['forwards'])},{qi(m['reactions'])});\n"
        )

    out.write("\n-- Translations\n")
    for m in messages:
        if any((m['text_clean_en'],m['names_en'],m['location_en'],m['dates_en'])):
            out.write(
                f"INSERT INTO message_translations (message_id,text_clean_en,names_en,location_en,dates_en) "
                f"VALUES ({m['id']},{q(m['text_clean_en'])},{q(m['names_en'])},"
                f"{q(m['location_en'])},{q(m['dates_en'])});\n"
            )

    out.write("\n-- Anonymized\n")
    for m in messages:
        if m['clean'] or m['text_clean_anon']:
            out.write(
                f"INSERT INTO message_anonymized (message_id,clean,text_clean_anon) "
                f"VALUES ({m['id']},{q(m['clean'])},{q(m['text_clean_anon'])});\n"
            )

    out.write("\n-- Extracted entities\n")
    for mid,cid,kind,ar,en in entities:
        out.write(
            f"INSERT INTO extracted_entities (message_id,case_id,kind,value_ar,value_en) VALUES ("
            f"{mid},{cid if cid is not None else 'NULL'},'{kind}',{q(ar)},{q(en)});\n"
        )

    out.write("\nCOMMIT;\n")

print(f"Wrote {dst}")
print(f"  cases:    {len(cases)}")
print(f"  messages: {len(messages)}")
print(f"  entities: {len(entities)}")
