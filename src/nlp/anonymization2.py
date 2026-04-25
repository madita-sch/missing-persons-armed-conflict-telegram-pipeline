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
# CANONICAL NAME (ORDER-INDEPENDENT)
# =========================================================
def canonical_name(name):
    name = normalize(name)
    parts = name.split()
    parts = sorted(parts)
    return " ".join(parts)


# =========================================================
# PHONE ANONYMIZATION
# =========================================================
PHONE_PATTERN = re.compile(r"""
    (?:
        \+?\d{1,3}[\s\-]?
    )?
    (?:\d[\s\-]?){7,15}
""", re.VERBOSE)


def anonymize_phones(text):
    if not isinstance(text, str):
        return text

    counter = {"i": 1}

    def repl(match):
        pid = f"PHONE_{counter['i']:03d}"
        counter["i"] += 1
        return pid

    return PHONE_PATTERN.sub(repl, text)


# =========================================================
# ENTITY REGISTRY (GLOBAL CONSISTENCY)
# =========================================================
class EntityRegistry:
    def __init__(self):
        self.name_to_id = {}
        self.counter = 1

    def get_person(self, name):
        key = canonical_name(name)

        if key in self.name_to_id:
            return self.name_to_id[key]

        pid = f"PERSON_{self.counter:03d}"
        self.counter += 1
        self.name_to_id[key] = pid
        return pid


# =========================================================
# SPLIT NAMES SAFELY
# =========================================================
def split_names(names):
    if not isinstance(names, str):
        return []

    names = normalize(names)
    parts = re.split(r"[;،,/]| و ", names)

    return [
        p.strip()
        for p in parts
        if len(p.strip().split()) >= 2
    ]


# =========================================================
# REPLACE NAMES IN TEXT
# =========================================================
def replace_names(text, names, registry):
    if not isinstance(text, str):
        return text

    text = normalize(text)
    candidates = split_names(names)

    # longest first to avoid partial overwrite
    candidates = sorted(candidates, key=len, reverse=True)

    for name in candidates:
        pid = registry.get_person(name)

        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        text = re.sub(pattern, pid, text)

    return text


# =========================================================
# MAIN FUNCTION
# =========================================================
def anonymize_dataframe(df, text_col="text_clean", names_col="names"):

    registry = EntityRegistry()
    output_texts = []

    for _, row in df.iterrows():
        text = row.get(text_col, "")
        names = row.get(names_col, "")

        text = replace_names(text, names, registry)
        text = anonymize_phones(text)

        output_texts.append(text)

    df[text_col + "_anon"] = output_texts

    # export mapping
    mapping_df = pd.DataFrame([
        {"name": k, "person_id": v}
        for k, v in registry.name_to_id.items()
    ])

    return df, mapping_df


#TEST
df_test = pd.read_excel("outputs/nlp_withanon.xlsx")

df_test_anon, name_map, phone_map = anonymize_dataframe(
    df_test,
    text_col="text_clean",
    names_col="names"
)

df_test_anon[["text_clean", "text_clean_anon"]].head()
with pd.ExcelWriter("outputs/nlp_withanon.xlsx", engine="openpyxl") as writer:
    df_test.to_excel(writer, sheet_name="all_predictions", index=False)
