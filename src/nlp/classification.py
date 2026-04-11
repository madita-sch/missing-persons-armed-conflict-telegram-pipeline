class ClassicalClassifier:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model

    def train(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)