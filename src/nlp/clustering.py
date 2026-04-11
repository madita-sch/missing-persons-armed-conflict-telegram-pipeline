from sklearn.cluster import DBSCAN, KMeans

class TextClusterer:
    def __init__(self, method="dbscan"):
        self.method = method

    def cluster(self, X):
        if self.method == "dbscan":
            return DBSCAN(eps=0.6, metric="cosine").fit_predict(X)
        else:
            return KMeans(n_clusters=5).fit_predict(X)