import re

# =========================================================
# NORMALIZATION
# =========================================================
def normalize(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# GLOBAL REGISTRY (IMPORTANT)
# =========================================================
class EntityRegistry:
    def __init__(self):
        self.name_to_person = {}
        self.person_counter = 1

    def get_person(self, name):
        name = normalize(name)

        if name in self.name_to_person:
            return self.name_to_person[name]

        pid = f"PERSON_{self.person_counter:03d}"
        self.person_counter += 1
        self.name_to_person[name] = pid
        return pid


# =========================================================
# MATCH NAME INSIDE TEXT SAFELY
# =========================================================
def replace_names_in_text(text, names, registry):
    if not isinstance(text, str):
        return text

    text = normalize(text)

    if not isinstance(names, str):
        return text

    # split multi-names safely
    candidates = re.split(r"[;،,]", names)

    # longest first = prevents partial overwrites
    candidates = sorted(
        [normalize(n) for n in candidates if len(n.split()) >= 2],
        key=len,
        reverse=True
    )

    for name in candidates:
        if not name:
            continue

        person = registry.get_person(name)

        # strict word-boundary match (prevents false matches)
        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        text = re.sub(pattern, person, text)

    return text


# =========================================================
# MAIN FUNCTION
# =========================================================
def anonymize_dataframe(df, text_col="text_clean", names_col="names"):

    registry = EntityRegistry()

    anonymized_texts = []

    for _, row in df.iterrows():
        text = row[text_col]
        names = row.get(names_col, "")

        new_text = replace_names_in_text(text, names, registry)
        anonymized_texts.append(new_text)

    df[text_col + "_anon"] = anonymized_texts

    # export mapping (VERY IMPORTANT for traceability)
    mapping_df = pd.DataFrame([
        {"name": k, "person_id": v}
        for k, v in registry.name_to_person.items()
    ])

    return df, mapping_df