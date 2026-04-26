import re
import pandas as pd

# =========================================================
# NORMALIZATION
# =========================================================
def normalize(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# PHONE ANONYMIZATION
# =========================================================
PHONE_PATTERN = re.compile(r"(?:(?:\+?\d{1,3})?[\s\-]?)?(?:\d[\s\-]?){7,15}")

def anonymize_phones(text):
    if not isinstance(text, str):
        return text

    counter = {"i": 1}

    def repl(_):
        pid = f"PHONE_{counter['i']:03d}"
        counter["i"] += 1
        return pid

    return PHONE_PATTERN.sub(repl, text)


# =========================================================
# SPLIT NAMES
# =========================================================
def split_names(names):
    if not isinstance(names, str):
        return []

    names = normalize(names)
    parts = re.split(r"[;،,/]| و ", names)

    return [p.strip() for p in parts if len(p.strip().split()) >= 2]


# =========================================================
# MAIN ANONYMIZATION (CLUSTER-BASED)
# =========================================================
def anonymize_dataframe(df, text_col="text_clean", names_col="names", cluster_col="cluster_id"):

    df = df.copy()

    # -------------------------------------------------
    # 1. ASSIGN PERSON IDS BY CLUSTER
    # -------------------------------------------------
    cluster_to_pid = {}
    pid_counter = 1

    for cluster in df[cluster_col].fillna(-1).unique():
        if cluster == -1:
            continue  # optional: keep unclustered separate or ignore

        cluster_to_pid[cluster] = f"PERSON_{pid_counter:03d}"
        pid_counter += 1

    # -------------------------------------------------
    # 2. BUILD NAME → CLUSTER MAP
    # -------------------------------------------------
    name_to_cluster = {}

    for _, row in df.iterrows():
        cluster = row.get(cluster_col, -1)
        names = split_names(row.get(names_col, ""))

        for n in names:
            name_to_cluster[n] = cluster

    # -------------------------------------------------
    # 3. REPLACE IN TEXT
    # -------------------------------------------------
    anon_texts = []

    for _, row in df.iterrows():
        text = normalize(row.get(text_col, ""))

        names = split_names(row.get(names_col, ""))

        # replace longest first
        names = sorted(names, key=len, reverse=True)

        for name in names:
            cluster = name_to_cluster.get(name, -1)

            if cluster == -1:
                continue

            pid = cluster_to_pid.get(cluster, "PERSON_UNKNOWN")

            pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
            text = re.sub(pattern, pid, text)

        text = anonymize_phones(text)
        anon_texts.append(text)

    df[text_col + "_anon"] = anon_texts

    # -------------------------------------------------
    # 4. EXPORT MAPPING
    # -------------------------------------------------
    mapping_df = pd.DataFrame([
        {"cluster_id": k, "person_id": v}
        for k, v in cluster_to_pid.items()
    ])

    return df, mapping_df

#TEST
df_test = pd.read_excel("outputs/test_Cluster_output.xlsx")

df_test_anon, name_map = anonymize_dataframe(
    df_test,
    text_col="text_clean",
    names_col="names",
    cluster_col="cluster_id"
)

with pd.ExcelWriter("outputs/nlp_withanon.xlsx", engine="openpyxl") as writer:
    df_test_anon.to_excel(writer, sheet_name="all_predictions", index=False)
    name_map.to_excel(writer, sheet_name="name_mapping", index=False)


df_test_anon[["text_clean", "text_clean_anon"]].head()
with pd.ExcelWriter("outputs/nlp_withanon.xlsx", engine="openpyxl") as writer:
    df_test.to_excel(writer, sheet_name="all_predictions", index=False)
