import numpy as np
import pandas as pd

from src.interfaces import EMBEDDINGS, EMBEDDINGS_BLOB, RELEVANT_MIN_STARS, load_embeddings


class ContentBased:
    def __init__(self, embeddings_path=EMBEDDINGS, name="content_chunked"):
        self.matrix, self.business_ids = load_embeddings(embeddings_path)
        self.index = {business_id: i for i, business_id in enumerate(self.business_ids)}
        self.name = name
        self.taste = {}

    def fit(self, train: pd.DataFrame) -> None:
        self.taste = {}
        for user_id, group in train.groupby("user_id"):
            liked = group[group["stars"] >= RELEVANT_MIN_STARS]
            pool = liked if not liked.empty else group

            idxs = [self.index[b] for b in pool["business_id"] if b in self.index]
            if not idxs:
                continue

            vec = self.matrix[idxs].mean(axis=0)
            norm = np.linalg.norm(vec)
            if norm > 0:
                self.taste[user_id] = vec / norm

    def recommend(self, user_ids, k, exclude):
        result = {user_id: [] for user_id in user_ids}

        known_users = [user_id for user_id in user_ids if user_id in self.taste]
        if not known_users:
            return result

        taste_matrix = np.stack([self.taste[user_id] for user_id in known_users])
        sims = taste_matrix @ self.matrix.T

        for row, user_id in enumerate(known_users):
            row_sims = sims[row].copy()

            excluded_idxs = [
                self.index[b] for b in exclude.get(user_id, ()) if b in self.index
            ]
            if excluded_idxs:
                row_sims[excluded_idxs] = -np.inf

            n = min(k, len(row_sims))
            top = np.argpartition(-row_sims, n - 1)[:n] if n > 0 else np.array([], dtype=int)
            top = top[np.argsort(-row_sims[top])]

            result[user_id] = [
                self.business_ids[i] for i in top if np.isfinite(row_sims[i])
            ]

        return result

    def score(self, user_id, business_ids):
        if user_id not in self.taste:
            return [float("nan")] * len(business_ids)

        taste = self.taste[user_id]
        return [
            float(taste @ self.matrix[self.index[b]]) if b in self.index else float("nan")
            for b in business_ids
        ]


class ContentBasedBlob(ContentBased):
    def __init__(self, embeddings_path=EMBEDDINGS_BLOB, name="content_blob"):
        super().__init__(embeddings_path=embeddings_path, name=name)
