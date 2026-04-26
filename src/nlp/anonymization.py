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
        self.map = {}
        self.counter = 1

    def get(self, name):
        key = canonical(name)
        if key not in self.map:
            self.map[key] = f"PERSON_{self.counter:03d}"
            self.counter += 1
        return self.map[key]

    def register_alias(self, name, person_id):
        key = canonical(name)
        if key not in self.map:
            self.map[key] = person_id


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
# =========================================================
class NameIndex:
    def __init__(self):
        self.token_to_entries = defaultdict(set)
        self.entries = []
        self._canon_seen = {}

    def add(self, name_tokens, person_id):
        if len(name_tokens) < 2:
            return
        canon = " ".join(sorted(name_tokens))
        if canon in self._canon_seen:
            return
        idx = len(self.entries)
        self.entries.append((tuple(name_tokens), canon, len(name_tokens), person_id))
        self._canon_seen[canon] = idx
        for tok in set(name_tokens):
            self.token_to_entries[tok].add(idx)

    def candidates_for_token(self, token):
        return [self.entries[i] for i in self.token_to_entries.get(token, [])]

    def is_empty(self):
        return len(self.entries) == 0


def build_index(registry):
    idx = NameIndex()
    for canon_key, pid in registry.map.items():
        tokens = canon_key.split()
        if len(tokens) >= 2:
            idx.add(tokens, pid)
    return idx


# =========================================================
# CONTIGUOUS SPAN MATCH
#
# Finds the shortest contiguous span in `words` starting at
# position `start` that contains all name_tokens in order.
#
# Returns (coverage, end_index) where end_index is exclusive.
# coverage = matched_tokens / len(name_tokens)
#
# Max span = name_len + SLACK extra words (handles titles like
# "الحاج", "ابو", "ابن" inserted between name parts).
# =========================================================
SLACK = 3  # allow up to N extra words between name tokens

def find_name_span(name_tokens, words, start):
    """
    Try to match name_tokens as an in-order subsequence starting
    at words[start], within a window of len(name_tokens) + SLACK.

    Returns (coverage, span_end) or (0, 0) if no match.
    """
    n = len(name_tokens)
    max_end = min(start + n + SLACK, len(words))

    ni = 0  # index into name_tokens
    last_matched = start - 1

    for ci in range(start, max_end):
        if ni < n and words[ci] == name_tokens[ni]:
            ni += 1
            last_matched = ci

    coverage = ni / n
    span_end = last_matched + 1 if ni > 0 else 0
    return coverage, span_end


# =========================================================
# REPLACE NAMES IN TEXT
# =========================================================
def replace_names_in_text(words, name_index, fuzzy_threshold=0.80):
    if not words or name_index.is_empty():
        return words

    n = len(words)
    output = []
    i = 0

    while i < n:
        candidates = name_index.candidates_for_token(words[i])

        if not candidates:
            output.append(words[i])
            i += 1
            continue

        # Longest name first — prefer most specific
        candidates = sorted(candidates, key=lambda e: -e[2])

        best_pid = None
        best_score = 0.0
        best_end = i

        for name_tokens, name_canon, name_len, pid in candidates:
            # Fast path: exact canonical match in tight window
            tight_end = min(i + name_len + 1, n)
            chunk = words[i:tight_end]
            if " ".join(sorted(chunk[:name_len])) == name_canon:
                best_pid = pid
                best_score = 1.0
                best_end = i + name_len
                break

            # Subsequence match with slack
            score, span_end = find_name_span(list(name_tokens), words, i)

            if score >= fuzzy_threshold and score > best_score:
                # Guard: don't let a 2-token name consume a huge span
                # unless coverage is very high
                span_len = span_end - i
                if span_len > name_len + SLACK:
                    continue
                best_score = score
                best_pid = pid
                best_end = span_end

        if best_pid is not None:
            output.append(best_pid)
            i = best_end
        else:
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
                        cluster_col=None, fuzzy_threshold=0.80):

    registry = EntityRegistry()

    if cluster_col and cluster_col in df.columns:
        for cluster_id, group in df.groupby(cluster_col):
            all_names = []
            for names_val in group[names_col].dropna():
                all_names.extend(split_names(names_val))
            if not all_names:
                continue

            rep_id = registry.get(all_names[0])
            for n in all_names[1:]:
                registry.register_alias(n, rep_id)
    else:
        for names_val in df[names_col].dropna():
            for n in split_names(names_val):
                registry.get(n)

    name_index = build_index(registry)

    _phone_counter[0] = 1
    outputs = []

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
