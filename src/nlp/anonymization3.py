import re
import pandas as pd
from collections import defaultdict


# =========================================================
# NORMALIZATION
# =========================================================
def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical(name):
    return " ".join(sorted(normalize(name).split()))


# =========================================================
# ENTITY REGISTRY
# =========================================================
class EntityRegistry:
    def __init__(self):
        self.map = {}   # canonical_key -> PERSON_XXX
        self.counter = 1

    def get(self, name):
        key = canonical(name)
        if key not in self.map:
            self.map[key] = f"PERSON_{self.counter:03d}"
            self.counter += 1
        return self.map[key]

    def register_alias(self, name, person_id):
        self.map[canonical(name)] = person_id


# =========================================================
# SPLIT MULTI-NAMES
# =========================================================
def split_names(names):
    if not isinstance(names, str):
        return []
    parts = re.split(r"[;،,/]| و ", normalize(names))
    return [p.strip() for p in parts if len(p.strip().split()) >= 2]


# =========================================================
# NAME INDEX
# Inverted index: token -> list of (name_tokens_tuple, person_id)
# Enables O(1) candidate lookup instead of scanning all names
# =========================================================
class NameIndex:
    def __init__(self):
        # token -> set of entry indices
        self.token_to_entries = defaultdict(set)
        # list of (tokens_tuple, canonical_key, length, person_id)
        self.entries = []

    def add(self, name_tokens, person_id):
        canon = " ".join(sorted(name_tokens))
        # Deduplicate: don't add same canonical twice
        for i, (_, ck, _, _) in enumerate(self.entries):
            if ck == canon:
                return i
        idx = len(self.entries)
        self.entries.append((tuple(name_tokens), canon, len(name_tokens), person_id))
        for tok in name_tokens:
            self.token_to_entries[tok].add(idx)
        return idx

    def candidates_for_token(self, token):
        """Return entries that contain this token."""
        return [self.entries[i] for i in self.token_to_entries.get(token, [])]

    def is_empty(self):
        return len(self.entries) == 0


# =========================================================
# BUILD NAME INDEX FROM REGISTRY
# =========================================================
def build_index(registry):
    idx = NameIndex()
    for canon_key, pid in registry.map.items():
        tokens = canon_key.split()  # already sorted/canonical
        if len(tokens) >= 2:
            idx.add(tokens, pid)
    return idx


# =========================================================
# REPLACE NAMES — O(words × avg_candidates × max_name_len)
# =========================================================
def replace_names_in_text(words, name_index, fuzzy_threshold=0.75):
    if not words or name_index.is_empty():
        return words

    n = len(words)
    output = []
    i = 0

    while i < n:
        # Get candidate name entries that share a token with words[i]
        candidates = name_index.candidates_for_token(words[i])

        if not candidates:
            output.append(words[i])
            i += 1
            continue

        # Sort candidates longest-first for greedy longest match
        candidates = sorted(candidates, key=lambda e: -e[2])
        max_win = candidates[0][2] + 1  # longest candidate length + 1 buffer

        matched = False
        for window in range(min(max_win, n - i), 1, -1):
            chunk = words[i:i + window]
            chunk_set = set(chunk)
            chunk_canon = " ".join(sorted(chunk))

            for name_tokens, name_canon, name_len, pid in candidates:
                if abs(window - name_len) > 2:
                    continue

                # Exact canonical match
                if chunk_canon == name_canon:
                    output.append(pid)
                    i += window
                    matched = True
                    break

                # Fuzzy: fraction of name tokens present in chunk
                name_set = set(name_tokens)
                overlap = len(name_set & chunk_set) / len(name_set)
                if overlap >= fuzzy_threshold:
                    output.append(pid)
                    i += window
                    matched = True
                    break

            if matched:
                break

        if not matched:
            output.append(words[i])
            i += 1

    return output


# =========================================================
# PHONE ANONYMIZATION
# =========================================================
PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-]{7,}\d")

_phone_counter = [1]

def anonymize_phones(text):
    def repl(_):
        pid = f"PHONE_{_phone_counter[0]:03d}"
        _phone_counter[0] += 1
        return pid
    return PHONE_PATTERN.sub(repl, text)


# =========================================================
# MAIN
# =========================================================
def anonymize_dataframe(df, text_col="text_clean", names_col="names",
                        cluster_col=None, fuzzy_threshold=0.75):

    registry = EntityRegistry()

    # ----------------------------------------------------------
    # STEP 1: One pass — build registry from all clusters/rows
    # ----------------------------------------------------------
    if cluster_col and cluster_col in df.columns:
        for cluster_id, group in df.groupby(cluster_col):
            all_names = []
            for names in group[names_col].dropna():
                all_names.extend(split_names(names))

            if not all_names:
                continue

            rep_id = registry.get(all_names[0])

            for n in all_names:
                registry.register_alias(n, rep_id)

                # Register all sub-spans of length >= 2
                tokens = normalize(n).split()
                for start in range(len(tokens)):
                    for end in range(start + 2, len(tokens) + 1):
                        sub = " ".join(tokens[start:end])
                        sub_key = canonical(sub)
                        if sub_key not in registry.map:
                            registry.map[sub_key] = rep_id
    else:
        # No cluster col: just register all names row by row
        for names in df[names_col].dropna():
            for n in split_names(names):
                registry.get(n)
                tokens = normalize(n).split()
                for start in range(len(tokens)):
                    for end in range(start + 2, len(tokens) + 1):
                        sub = " ".join(tokens[start:end])
                        sub_key = canonical(sub)
                        if sub_key not in registry.map:
                            registry.get(sub)

    # ----------------------------------------------------------
    # STEP 2: Build inverted index ONCE (not per row)
    # ----------------------------------------------------------
    name_index = build_index(registry)

    # ----------------------------------------------------------
    # STEP 3: Anonymize each row using shared index
    # ----------------------------------------------------------
    outputs = []
    _phone_counter[0] = 1  # reset phone counter per dataframe

    for _, row in df.iterrows():
        text = row.get(text_col, "")
        if not isinstance(text, str):
            outputs.append(text)
            continue

        words = normalize(text).split()
        replaced = replace_names_in_text(words, name_index, fuzzy_threshold)
        anon_text = anonymize_phones(" ".join(replaced))
        outputs.append(anon_text)

    df = df.copy()
    df[text_col + "_anon"] = outputs

    mapping = pd.DataFrame([
        {"canonical_name": k, "person_id": v}
        for k, v in registry.map.items()
    ])

    return df, mapping


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    df_test = pd.read_excel("outputs/test_Cluster_output.xlsx")

    df_test_anon, name_map = anonymize_dataframe(
        df_test,
        text_col="text_clean",
        names_col="names",
        cluster_col="cluster_id",
        fuzzy_threshold=0.75
    )

    with pd.ExcelWriter("outputs/nlp_withanon.xlsx", engine="openpyxl") as writer:
        df_test_anon.to_excel(writer, sheet_name="all_predictions", index=False)
        name_map.to_excel(writer, sheet_name="name_mapping", index=False)