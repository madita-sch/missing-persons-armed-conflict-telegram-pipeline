from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np


# Load model once (IMPORTANT for performance)
_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def cluster_texts(texts, eps=0.35, min_samples=2):
    """
    Semantic clustering using embeddings + DBSCAN.
    Returns cluster labels for each text.
    """

    if len(texts) == 0:
        return []

    # ---------------------------------------------------------
    # 1. EMBEDDINGS
    # ---------------------------------------------------------
    embeddings = _model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    # ---------------------------------------------------------
    # 2. DBSCAN CLUSTERING
    # ---------------------------------------------------------
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine"
    )

    labels = clustering.fit_predict(embeddings)

    return labels.tolist()