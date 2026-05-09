# Import libraries
import re
import pandas as pd
from collections import defaultdict

# Initialize slack as tolerance window, allowing fuzzy matching of len(name_tokens) + SLACK
SLACK = 2 

# Normalization of arabic text
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


# Define a canonical form for a name 
# to enable variation in name order and match across messages
def canonical(name):
    return " ".join(sorted(normalize(name).split()))


# Create EntityRegistry to assign unique IDs to names and handle aliases
class EntityRegistry:
    def __init__(self):
        self.map = {}       # canonical_name -> person_id
        self.counter = 1    # incremental ID generator

    def get(self, name):
        key = canonical(name)

        # Assign new ID only if not seen before; otherwise return existing ID
        if key not in self.map:
            self.map[key] = f"PERSON_{self.counter:03d}"
            self.counter += 1
        return self.map[key]

    def register_alias(self, name, person_id):
        # Map different name variations to the same person_id
        key = canonical(name)
        if key not in self.map:
            self.map[key] = person_id


# Split multiple names in a string using common delimiters and Arabic and
def split_names(names):
    if not isinstance(names, str):
        return []
    parts = re.split(r"[;،,/]| و ", normalize(names))
    return [p.strip() for p in parts if len(p.strip().split()) >= 2]


# Define NameIndex to match names in text using token-based lookup and fuzzy matching
class NameIndex:
    def __init__(self):
        self.token_to_entries = defaultdict(set)
        self.entries = []
        self._canon_seen = {}

    def add(self, name_tokens, person_id):
        # Ignore single-token names
        if len(name_tokens) < 2:
            return
        canon = " ".join(sorted(name_tokens))
        # Avoid duplicate entries for the same canonical name
        if canon in self._canon_seen:
            return
        idx = len(self.entries)
        self.entries.append((tuple(name_tokens), canon, len(name_tokens), person_id))
        self._canon_seen[canon] = idx
        # Map each token to this entry (for fast lookup)
        for tok in set(name_tokens):
            self.token_to_entries[tok].add(idx)

    def candidates_for_token(self, token):
        # Retrieve possible name matches for a token    
        return [self.entries[i] for i in self.token_to_entries.get(token, [])]

    def is_empty(self):
        return len(self.entries) == 0

# Build index from registry for efficient name matching in text
def build_index(registry):
    idx = NameIndex()
    for canon_key, pid in registry.map.items():
        tokens = canon_key.split()
        # Only index names with 2 or more tokens to reduce false positives
        if len(tokens) >= 2:
            idx.add(tokens, pid)
    return idx


# Fuzzy name matching (span detection)
# to match names even when tokens slightly separated or extra tokens appear
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


# Replace best match with pseudonyms in text using the name index and fuzzy matching
def replace_names_in_text(words, name_index, fuzzy_threshold=0.80):
    if not words or name_index.is_empty():
        return words

    n = len(words)
    output = []
    i = 0

    while i < n:
        candidates = name_index.candidates_for_token(words[i])

        # No match --> keep original token and move on
        if not candidates:
            output.append(words[i])
            i += 1
            continue

        # Prefer longer (more speficic) names by sorting candidates by length
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

            # Fuzzy subsequence match with slack
            score, span_end = find_name_span(list(name_tokens), words, i)

            if score >= fuzzy_threshold and score > best_score:
                span_len = span_end - i
                # Prevent overly long incorrect matches 
                if span_len > name_len + SLACK:
                    continue
                best_score = score
                best_pid = pid
                best_end = span_end

        # Replace of keep original token
        if best_pid is not None:
            output.append(best_pid)
            i = best_end
        else:
            output.append(words[i])
            i += 1

    return output


# Phone number Pseudynomization
# Define phone pattern
PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-]{7,}\d")
_phone_counter = [1]

# Define function to replace phone numbers with pseudonyms
def pseudonymize_phones(text):
    def repl(_):
        pid = f"PHONE_{_phone_counter[0]:03d}"
        _phone_counter[0] += 1
        return pid
    return PHONE_PATTERN.sub(repl, text)


# Create main pseudonymization function 
def pseudonymize_dataframe(df, text_col="text_clean", names_col="names",
                        cluster_col=None, fuzzy_threshold=0.80):

    registry = EntityRegistry()

    # Build Entity Registry from names column, using cluster_col to group aliases if provided
    if cluster_col and cluster_col in df.columns:
        for cluster_id, group in df.groupby(cluster_col):
            all_names = []
            for names_val in group[names_col].dropna():
                all_names.extend(split_names(names_val))
            if not all_names:
                continue

            # First name defines the cluster identity
            rep_id = registry.get(all_names[0])

            # Map all aliases to the same ID
            for n in all_names[1:]:
                registry.register_alias(n, rep_id)
    else:
        # No clustering, then treat each name independently
        for names_val in df[names_col].dropna():
            for n in split_names(names_val):
                registry.get(n)

    # Build name index for matching in text
    name_index = build_index(registry)

    # Reset phone counter for consistent pseudonymization across runs
    _phone_counter[0] = 1
    outputs = []

    # Process text row by row 
    for _, row in df.iterrows():
        text = row.get(text_col, "")
        if not isinstance(text, str):
            outputs.append(text)
            continue

        # Normalize and tokenize
        words = normalize(text).split()
        # Replace names with pseudonyms
        replaced = replace_names_in_text(words, name_index, fuzzy_threshold)
        # Replace phone numbers with pseudonyms
        anon_text = pseudonymize_phones(" ".join(replaced))
        outputs.append(anon_text)

    # Save results
    df = df.copy()
    df[text_col + "_anon"] = outputs

    # Mapping table for reference (canonical name to person_id)
    mapping = pd.DataFrame([
        {"canonical_name": k, "person_id": v}
        for k, v in registry.map.items()
    ])

    return df, mapping
