class EntityMatcher:
    def cosine_similarity_matrix(self, embeddings):
        return cosine_similarity(embeddings)

    def find_pairs(self, sim_matrix, threshold=0.8):
        pairs = []
        for i in range(len(sim_matrix)):
            for j in range(i+1, len(sim_matrix)):
                if sim_matrix[i, j] > threshold:
                    pairs.append((i, j))
        return pairs