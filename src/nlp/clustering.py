# =====================================================
# clustering.py (SRC MODULE - NO EXECUTION)
# =====================================================

import re
import networkx as nx
from difflib import SequenceMatcher

# -----------------------------------------------------
# NORMALIZATION
# -----------------------------------------------------
def normalize(name):
    name = str(name).lower().strip()

    name = name.replace("ابو ", " ")
    name = name.replace("أبو ", " ")

    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)

    return name.strip()


# -----------------------------------------------------
# TOKENIZATION
# -----------------------------------------------------
def tokens(name):
    return set(name.split())


# -----------------------------------------------------
# SIMILARITY COMPONENTS
# -----------------------------------------------------
def jaccard(a, b):
    A, B = tokens(a), tokens(b)
    if not A or not B:
        return 0
    return len(A & B) / len(A | B)


def dice(a, b):
    A, B = tokens(a), tokens(b)
    if not A or not B:
        return 0
    return (2 * len(A & B)) / (len(A) + len(B))


def edit_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


def rule_boost(a, b):
    A, B = tokens(a), tokens(b)
    return 0.2 if len(A & B) >= 2 else 0.0


# -----------------------------------------------------
# FINAL SIMILARITY FUNCTION
# -----------------------------------------------------
def similarity(a, b):
    return (
        0.3 * jaccard(a, b) +
        0.3 * dice(a, b) +
        0.3 * edit_sim(a, b) +
        0.1 * rule_boost(a, b)
    )


# -----------------------------------------------------
# GRAPH CONSTRUCTION
# -----------------------------------------------------
def build_graph(df, threshold=0.60, text_col="clean"):

    G = nx.Graph()

    for i in df.index:
        G.add_node(i)

    for i in df.index:
        for j in df.index:
            if j <= i:
                continue

            sim = similarity(df.loc[i, text_col], df.loc[j, text_col])

            if sim >= threshold:
                G.add_edge(i, j)

    return G


# -----------------------------------------------------
# CLUSTER EXTRACTION
# -----------------------------------------------------
def extract_clusters(G):

    clusters = {}
    cluster_id = 0

    for component in nx.connected_components(G):
        for node in component:
            clusters[node] = cluster_id
        cluster_id += 1

    return clusters