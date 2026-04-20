import pandas as pd
import re

# =========================================================
# BUILD ANONYMIZATION MAP
# =========================================================
def build_anonymization_map(entity_ids):
    """
    Converts entity_ids into anonymous PERSON_XXX labels.
    This is a privacy layer (NOT entity resolution).
    """
    unique_entities = sorted(set([e for e in entity_ids if e]))

    anon_map = {}

    for i, ent in enumerate(unique_entities):
        anon_map[ent] = f"PERSON_{i+1:03d}"

    return anon_map


# =========================================================
# ANONYMIZE TEXT
# =========================================================
import re

def anonymize_text(text, name_to_anon):
    """
    Replace raw names in text with PERSON_xxx labels.
    Works on actual surface names, not entity_id.
    """

    if not isinstance(text, str):
        return text

    for name, anon in name_to_anon.items():
        if not name:
            continue

        pattern = re.escape(name)

        # replace all occurrences
        text = re.sub(pattern, anon, text)

    return text

# =========================================================
# APPLY TO DATAFRAME
# =========================================================
def anonymize_dataframe(df, entity_col="entity_id", text_col="text_clean", names_col="names"):

    entity_ids = df[entity_col].fillna("").astype(str).tolist()

    # build PERSON mapping
    anon_map = build_anonymization_map(entity_ids)

    # reverse: entity_id → PERSON_xxx
    df["anon_id"] = df[entity_col].map(anon_map)

    # -----------------------------------------------------
    # CRITICAL FIX: build NAME → PERSON mapping
    # -----------------------------------------------------
    name_to_anon = {}

    for name, eid in zip(df[names_col], df[entity_col]):
        if isinstance(name, str) and eid in anon_map:
            name_to_anon[name] = anon_map[eid]

    # apply to text
    df[text_col + "_anon"] = df[text_col].apply(
        lambda x: anonymize_text(x, name_to_anon)
    )

    # apply to names column
    df[names_col + "_anon"] = df[entity_col].map(anon_map)

    return df, anon_map