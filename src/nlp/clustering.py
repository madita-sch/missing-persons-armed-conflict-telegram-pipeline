# Import libraries
import re
import networkx as nx
from difflib import SequenceMatcher

#Normalization (standardize strings to reduce variation)
def normalize(name):
    name = str(name).lower().strip()

    # Remove common Arabic prefixes
    name = name.replace("ابو ", " ")
    name = name.replace("أبو ", " ")

    # Remove punctuation and special characters
    name = re.sub(r"[^\w\s]", "", name)

    # Normalize whitespace 
    name = re.sub(r"\s+", " ", name)

    return name.strip()


# Tokenization (split name into a set of tokens to prepare for similarity measure)
def tokens(name):
    return set(name.split())


# Define Similarity components
# Jaccard: measures the overlap between two sets of tokens to detect shared name components
def jaccard(a, b):
    A, B = tokens(a), tokens(b)
    if not A or not B:
        return 0
    return len(A & B) / len(A | B)

# Sørensen–Dice coefficient: similar to Jaccard but gives more weight to shared tokens to identify partial matches
def dice(a, b):
    A, B = tokens(a), tokens(b)
    if not A or not B:
        return 0
    return (2 * len(A & B)) / (len(A) + len(B))

# Character-level similarity using edit distance ratio
# Compares the sequences of characters in the names to capture minor spelling variations and typos
def edit_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Rule-based boost: to similarity if at least 2 tokens shared
def rule_boost(a, b):
    A, B = tokens(a), tokens(b)
    return 0.2 if len(A & B) >= 2 else 0.0


# Similarity function combining in a single weighted score: 
# Token-based and character-level similarity and rule-based boost
def similarity(a, b):
    return (
        0.3 * jaccard(a, b) +
        0.3 * dice(a, b) +
        0.3 * edit_sim(a, b) +
        0.1 * rule_boost(a, b)
    )


# Graph Construction capturing relationships between names based on similarity scores
# Nodes represent individual names, edges represent similarity above a threshold, 
def build_graph(df, threshold=0.60, text_col="clean"):

    G = nx.Graph()

    # Add each record as a node in the graph
    for i in df.index:
        G.add_node(i)

    # Compute pairwise similarity between records, connect those exceeding threshold of 0.60
    for i in df.index:
        for j in df.index:
            if j <= i:
                continue

            sim = similarity(df.loc[i, text_col], df.loc[j, text_col])

            if sim >= threshold:
                G.add_edge(i, j)

    return G


# Cluster extraction: each connected component becomes on cluster_id
def extract_clusters(G):

    clusters = {}
    cluster_id = 0

    for component in nx.connected_components(G):
        for node in component:
            clusters[node] = cluster_id
        cluster_id += 1

    return clusters