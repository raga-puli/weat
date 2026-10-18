import random
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import skew

from src.interfaces import (
    EMBEDDINGS,
    EMBEDDINGS_BLOB,
    RELEVANT_MIN_STARS,
    SEED,
    load_embeddings,
)


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
        sims = self._score_matrix(taste_matrix)

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

    def taste_vector_from(self, business_ids: list[str]) -> np.ndarray | None:
        """L2-normalized mean embedding of an arbitrary history (e.g. a
        member's pre-window likes), skipping unknown IDs. None if no ID is
        known or the mean is the zero vector.
        """
        idxs = [self.index[b] for b in business_ids if b in self.index]
        if not idxs:
            return None
        vec = self.matrix[idxs].mean(axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    def scores_for(self, taste: np.ndarray) -> np.ndarray:
        """The per-user score of every restaurant, in business_ids.txt order:
        here the cosine with `taste` (rows of self.matrix are unit-norm).
        Subclasses that change scoring override this (and _score_matrix)."""
        return self.matrix @ taste

    def _score_matrix(self, taste_matrix: np.ndarray) -> np.ndarray:
        """scores_for for a batch of users: (n_users, n_restaurants)."""
        return taste_matrix @ self.matrix.T


class ContentBasedBlob(ContentBased):
    def __init__(self, embeddings_path=EMBEDDINGS_BLOB, name="content_blob"):
        super().__init__(embeddings_path=embeddings_path, name=name)


class ContentBasedCentered(ContentBased):
    """ContentBased over mean-centered, re-normalized embeddings.

    Subtracting the catalog-wide mean vector before L2-normalizing
    pulls apart restaurants that were only close together because
    they all sat near the centroid of the embedding space (the
    "hub" restaurants with thin, generic category tags).
    """

    def __init__(self, embeddings_path=EMBEDDINGS, name="content_centered"):
        super().__init__(embeddings_path=embeddings_path, name=name)
        centered = self.matrix - self.matrix.mean(axis=0)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = centered / norms


class ContentBasedDebiased(ContentBasedCentered):
    """ContentBasedCentered with each restaurant's mean score subtracted.

    After fit, hub[r] is the mean over all fitted users u of cos(u, r), and
    every score is cos(u, r) - hub[r]. That is column-centering of the
    user-by-restaurant score matrix: a hubness correction that stops
    restaurants close to almost everyone's taste from topping every list.
    It was fixed in advance and has no parameters. Because cosine is linear
    in u, hub = matrix @ (mean taste vector), a single matrix-vector product.
    Before fit, hub is zero and scores are plain cosines.
    """

    def __init__(self, embeddings_path=EMBEDDINGS, name="content_debiased"):
        super().__init__(embeddings_path=embeddings_path, name=name)
        self.hub = np.zeros(len(self.business_ids), dtype=self.matrix.dtype)

    def fit(self, train: pd.DataFrame) -> None:
        super().fit(train)
        if self.taste:
            mean_taste = np.stack(list(self.taste.values())).mean(axis=0)
            self.hub = self.matrix @ mean_taste
        else:
            self.hub = np.zeros(len(self.business_ids), dtype=self.matrix.dtype)

    def scores_for(self, taste: np.ndarray) -> np.ndarray:
        return self.matrix @ taste - self.hub

    def _score_matrix(self, taste_matrix: np.ndarray) -> np.ndarray:
        return taste_matrix @ self.matrix.T - self.hub

    def score(self, user_id, business_ids):
        if user_id not in self.taste:
            return [float("nan")] * len(business_ids)
        scores = self.scores_for(self.taste[user_id])
        return [float(scores[self.index[b]]) if b in self.index else float("nan") for b in business_ids]


def hubness_diagnostics(recs, catalog_business_ids, seed=SEED, n_pairs=500):
    """Diagnose how much a recs dict[user_id, list[business_id]]
    collapses onto a small set of "hub" businesses.

    Returns occurrence_skewness (skew of the per-catalog-item
    occurrence count, zero counts included), the 10 most frequently
    recommended items (with counts and share of users), the number
    of unique items recommended, and the mean top-10 overlap between
    `n_pairs` seeded random user pairs (Jaccard-style: shared items /
    the larger of the two list sizes).
    """
    n_users = len(recs)

    counts = Counter()
    for items in recs.values():
        counts.update(items)

    occurrence = np.array([counts.get(b, 0) for b in catalog_business_ids])
    occurrence_skewness = float(skew(occurrence)) if len(occurrence) else float("nan")

    top10_most_frequent = [
        {
            "business_id": business_id,
            "count": count,
            "share_of_users": count / n_users if n_users else 0.0,
        }
        for business_id, count in counts.most_common(10)
    ]

    n_unique_items_recommended = int((occurrence > 0).sum())

    rng = random.Random(seed)
    user_ids = list(recs.keys())
    overlaps = []
    if len(user_ids) >= 2:
        for _ in range(n_pairs):
            u, v = rng.sample(user_ids, 2)
            a, b = set(recs[u]), set(recs[v])
            denom = max(len(a), len(b), 1)
            overlaps.append(len(a & b) / denom)
    mean_top10_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    return {
        "n_users": n_users,
        "occurrence_skewness": occurrence_skewness,
        "top10_most_frequent": top10_most_frequent,
        "n_unique_items_recommended": n_unique_items_recommended,
        "mean_top10_overlap": mean_top10_overlap,
        "n_pairs_sampled": len(overlaps),
    }
