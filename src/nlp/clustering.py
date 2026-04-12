import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_data(path):
    df = pd.read_csv(path)
    return df["text"].tolist()


def embed_texts(texts):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def cluster_texts(texts, n_clusters=5):
    embeddings = embed_texts(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    return labels


def save_clusters(texts, labels, output_path):
    df = pd.DataFrame({
        "text": texts,
        "cluster": labels
    })
    df.to_csv(output_path, index=False)