"""
=============================================================
  LEAKAGE DETECTION SCRIPT
  Train: annotated_final  (columns: text_clean, label)
  Test : test_Gaza20249   (columns: text_clean, is_missing)
=============================================================
Run:  python leakage_detection.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# 0. LOAD YOUR DATASETS
#    Replace the two lines below with however
#    you actually load annotated_final and
#    test_Gaza20249 (read_csv, from memory, etc.)
# ─────────────────────────────────────────────
train = pd.read_excel("data/annotated_final.xlsx")
test  = pd.read_csv("data/gold_Gaza20249.csv")

# For now we assume they're already in memory:
train = annotated_final.copy()
test  = test_Gaza20249.copy()

TEXT_COL_TRAIN = "text_clean"
TEXT_COL_TEST  = "text_clean"
LABEL_TRAIN    = "label"
LABEL_TEST     = "is_missing"

SIMILARITY_THRESHOLD = 0.92   # tune if needed

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
DIVIDER = "\n" + "═" * 60

def normalize(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()

def section(title: str):
    print(f"{DIVIDER}\n  {title}\n{'─'*60}")


# ─────────────────────────────────────────────
# PREP
# ─────────────────────────────────────────────
train["norm"] = normalize(train[TEXT_COL_TRAIN])
test["norm"]  = normalize(test[TEXT_COL_TEST])

train["split"] = "train"
test["split"]  = "test"

# unified corpus for intra-corpus checks
combined = pd.concat(
    [train[["norm", LABEL_TRAIN, "split"]].rename(columns={LABEL_TRAIN: "label"}),
     test[["norm", LABEL_TEST,  "split"]].rename(columns={LABEL_TEST:  "label"})],
    ignore_index=True
)


# ════════════════════════════════════════════
# CHECK 1 — Exact duplicates within train set
# ════════════════════════════════════════════
section("CHECK 1 · Exact duplicates inside TRAIN")

train_dups = train["norm"].value_counts()
train_dups = train_dups[train_dups > 1]
print(f"Unique texts that appear >1× in train : {len(train_dups)}")
print(f"Total duplicated rows in train        : {train_dups.sum()}")
if len(train_dups):
    print("\nTop 5 most-repeated texts:")
    for txt, cnt in train_dups.head(5).items():
        print(f"  [{cnt}×] {txt[:120]}")


# ════════════════════════════════════════════
# CHECK 2 — Exact train/test overlap  ← KEY
# ════════════════════════════════════════════
section("CHECK 2 · Exact train ∩ test overlap")

train_set = set(train["norm"])
test_set  = set(test["norm"])
overlap   = train_set & test_set

print(f"Exact overlapping texts : {len(overlap)}")
if overlap:
    print("\n⚠️  LEAKAGE CONFIRMED — sample of overlapping texts:")
    for t in list(overlap)[:10]:
        print(f"  · {t[:120]}")
else:
    print("✓  No exact duplicates between train and test.")


# ════════════════════════════════════════════
# CHECK 3 — Near-duplicates (TF-IDF cosine)
# ════════════════════════════════════════════
section("CHECK 3 · Near-duplicates via TF-IDF cosine similarity")

all_texts = train["norm"].tolist() + test["norm"].tolist()
n_train   = len(train)

vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), sublinear_tf=True)
X = vectorizer.fit_transform(all_texts)

X_train = X[:n_train]
X_test  = X[n_train:]

# compute cross-split similarity (train rows × test rows)
sim_matrix = cosine_similarity(X_train, X_test)   # shape: (n_train, n_test)

# find pairs above threshold
row_idx, col_idx = np.where(sim_matrix > SIMILARITY_THRESHOLD)
near_dup_pairs = list(zip(row_idx, col_idx,
                          sim_matrix[row_idx, col_idx]))

print(f"Near-duplicate pairs (threshold={SIMILARITY_THRESHOLD}) : {len(near_dup_pairs)}")

if near_dup_pairs:
    print(f"\n⚠️  HIGH SIMILARITY PAIRS — probable leakage:")
    top_pairs = sorted(near_dup_pairs, key=lambda x: -x[2])[:10]
    for ti, tei, score in top_pairs:
        print(f"\n  Similarity: {score:.4f}")
        print(f"  TRAIN [{ti}]: {all_texts[ti][:120]}")
        print(f"  TEST  [{n_train+tei}]: {all_texts[n_train+tei][:120]}")
else:
    print("✓  No near-duplicates above threshold found.")


# ════════════════════════════════════════════
# CHECK 4 — Telegram repost chains
#   Heuristic: same first 80 chars (prefix)
#   after stripping forwarding markers
# ════════════════════════════════════════════
section("CHECK 4 · Telegram repost chain fingerprint")

def telegram_prefix(s: str, length: int = 80) -> str:
    """Strip common Telegram forward markers, return first `length` chars."""
    s = s.lower().strip()
    for marker in ["forwarded from", "фwd:", "fwd:", ">>", "►"]:
        if s.startswith(marker):
            s = s[len(marker):].strip()
    return s[:length]

train["prefix"] = train["norm"].apply(telegram_prefix)
test["prefix"]  = test["norm"].apply(telegram_prefix)

train_prefixes = set(train["prefix"])
test_prefixes  = set(test["prefix"])
repost_overlap = train_prefixes & test_prefixes

# remove very short prefixes (noise)
repost_overlap = {p for p in repost_overlap if len(p) > 20}

print(f"Shared Telegram repost-chain prefixes : {len(repost_overlap)}")
if repost_overlap:
    print("\n⚠️  Likely repost chains in BOTH splits:")
    for p in list(repost_overlap)[:10]:
        print(f"  · {p[:100]}")
else:
    print("✓  No repost-chain overlaps detected.")


# ════════════════════════════════════════════
# CHECK 5 — Label + text pattern leakage
#   (duplicates that also carry the same label)
# ════════════════════════════════════════════
section("CHECK 5 · Label consistency among train duplicates")

dup_mask  = train["norm"].duplicated(keep=False)
dup_train = train[dup_mask].copy()

if len(dup_train):
    label_variance = (
        dup_train.groupby("norm")[LABEL_TRAIN]
        .nunique()
        .value_counts()
        .rename("count")
    )
    print("Duplicate text groups by # distinct labels:")
    print(label_variance.to_string())

    inconsistent = dup_train.groupby("norm")[LABEL_TRAIN].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    print(f"\nTexts with CONFLICTING labels : {len(inconsistent)}")
    if len(inconsistent):
        print("⚠️  Dataset inconsistency — same text, different labels:")
        for txt in inconsistent.index[:5]:
            sub = dup_train[dup_train["norm"] == txt][[TEXT_COL_TRAIN, LABEL_TRAIN]]
            print(sub.to_string(index=False))
            print()
else:
    print("No duplicates in train — skipping label consistency check.")


# ════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════
section("SUMMARY")

flags = {
    "Exact train/test overlap"       : len(overlap) > 0,
    "Near-duplicate pairs (≥0.92)"   : len(near_dup_pairs) > 0,
    "Telegram repost chain overlap"  : len(repost_overlap) > 0,
    "Conflicting labels (train dups)": len(dup_train) > 0 and
                                       len(train.groupby("norm")[LABEL_TRAIN]
                                               .nunique()[lambda x: x > 1]) > 0,
}

for check, flagged in flags.items():
    status = "⚠️  FLAGGED" if flagged else "✓  CLEAN"
    print(f"  {status:<14}  {check}")

if any(flags.values()):
    print("\n🔴  ACTION REQUIRED: leakage or inconsistency detected.")
    print("    Re-split your data ensuring all versions/reposts of")
    print("    a message land entirely in train OR test, not both.")
else:
    print("\n🟢  No leakage evidence found across all five checks.")
    print("    Your model metrics are likely genuine.")

print(DIVIDER)


# How many test rows are contaminated?
contaminated_idx = test[test["norm"].isin(set(train["norm"]))].index
print(f"Contaminated test rows : {len(contaminated_idx)}")
print(f"Total test rows        : {len(test)}")
print(f"Contamination rate     : {len(contaminated_idx)/len(test)*100:.1f}%")
print(f"\nContaminated test indices: {sorted(contaminated_idx.tolist())}")



# View contaminated rows with all their actual content
contaminated_rows = test.iloc[contaminated_idx].copy()

print(f"{'IDX':<6} {'MSG_ID':<15} {'TEXT (first 80 chars)':<80} {'LABEL'}")
print("─" * 120)

for pos, row in contaminated_rows.iterrows():
    # adjust column name if your ID column is named differently
    msg_id = row.get("id", row.get("message_id", row.get("msg_id", "NO_ID_COL")))
    label  = row.get("is_missing", "—")
    text   = str(row["text_clean"])[:80]
    print(f"{pos:<6} {str(msg_id):<15} {text:<80} {label}")