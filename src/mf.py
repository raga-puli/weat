"""Implicit-feedback matrix factorization (ALS) recommender.

Every train review is an interaction. Confidence follows Hu, Koren & Volinsky
(2008): c_ui = 1 + alpha * [stars >= RELEVANT_MIN_STARS], with preference
p_ui = 1 for every reviewed (user, restaurant) pair and 0 otherwise. A user
who reviewed a restaurant more than once gets one entry, liked if any review
was. All train users are used, including light users, who still help learn
item factors.

Backend: implicit.als.AlternatingLeastSquares (CPU) when importable; otherwise
an exact numpy implementation of the same objective. Either way the fit prints
which one ran.

    python -m src.eval --model src.mf:ALS      # 64 factors (default)
    python -m src.eval --model src.mf:ALS32    # 32 factors
    python -m src.eval --model src.mf:ALS128   # 128 factors
"""

import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.interfaces import RELEVANT_MIN_STARS, SEED

ALPHA = 10.0
REGULARIZATION = 0.05
ITERATIONS = 20


def confidence_matrix(train: pd.DataFrame, alpha: float = ALPHA):
    """(C, user_ids, business_ids): C is a users x restaurants CSR matrix of
    confidences 1 + alpha * liked, one entry per reviewed pair."""
    pairs = train.groupby(["user_id", "business_id"], sort=True)["stars"].max().reset_index()
    user_ids = pairs["user_id"].drop_duplicates().tolist()
    business_ids = sorted(pairs["business_id"].unique())
    rows = pd.Index(user_ids).get_indexer(pairs["user_id"])
    cols = pd.Index(business_ids).get_indexer(pairs["business_id"])
    conf = 1.0 + alpha * (pairs["stars"].to_numpy() >= RELEVANT_MIN_STARS)
    C = csr_matrix((conf.astype(np.float32), (rows, cols)), shape=(len(user_ids), len(business_ids)))
    return C, user_ids, business_ids


def _als_half_step(C: csr_matrix, Y: np.ndarray, regularization: float) -> np.ndarray:
    """Exact solve of x_u = (YtY + Yt(C_u - I)Y + reg I)^-1 Yt C_u p_u for every row u."""
    n_factors = Y.shape[1]
    YtY = Y.T @ Y
    reg = regularization * np.eye(n_factors)
    X = np.zeros((C.shape[0], n_factors), dtype=np.float64)
    for u in range(C.shape[0]):
        start, end = C.indptr[u], C.indptr[u + 1]
        if start == end:
            continue
        idx, conf = C.indices[start:end], C.data[start:end].astype(np.float64)
        Yi = Y[idx]
        A = YtY + (Yi.T * (conf - 1.0)) @ Yi + reg
        X[u] = np.linalg.solve(A, Yi.T @ conf)  # p_ui = 1 on observed entries
    return X


def als_numpy(C: csr_matrix, factors: int, regularization: float, iterations: int, seed: int = SEED):
    """Implicit-feedback ALS (Hu, Koren & Volinsky 2008) with exact solves."""
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.01, size=(C.shape[0], factors))
    Y = rng.normal(scale=0.01, size=(C.shape[1], factors))
    Ct = C.T.tocsr()
    for _ in range(iterations):
        X = _als_half_step(C, Y, regularization)
        Y = _als_half_step(Ct, X, regularization)
    return X.astype(np.float32), Y.astype(np.float32)


class ALS:
    """Recommender-protocol wrapper around implicit-feedback ALS."""

    def __init__(self, factors: int = 64, alpha: float = ALPHA, regularization: float = REGULARIZATION,
                 iterations: int = ITERATIONS, seed: int = SEED, force_numpy: bool = False):
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.force_numpy = force_numpy
        self.name = f"als_f{factors}"
        self.backend = None
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}
        self.business_ids: list[str] = []
        self.user_factors = np.zeros((0, factors), dtype=np.float32)
        self.item_factors = np.zeros((0, factors), dtype=np.float32)

    def fit(self, train: pd.DataFrame) -> None:
        C, user_ids, business_ids = confidence_matrix(train, self.alpha)
        self.user_index = {u: i for i, u in enumerate(user_ids)}
        self.item_index = {b: i for i, b in enumerate(business_ids)}
        self.business_ids = business_ids
        t0 = time.perf_counter()
        self.user_factors, self.item_factors, self.backend = self._factorize(C)
        print(f"ALS backend: {self.backend} (factors={self.factors}, alpha={self.alpha:g}, "
              f"regularization={self.regularization:g}, iterations={self.iterations}); "
              f"{C.shape[0]} users x {C.shape[1]} restaurants, {C.nnz} interactions, "
              f"fit {time.perf_counter() - t0:.1f}s")

    def _factorize(self, C: csr_matrix):
        if not self.force_numpy:
            try:
                from implicit.als import AlternatingLeastSquares
            except ImportError:
                pass
            else:
                from contextlib import nullcontext

                try:
                    from threadpoolctl import threadpool_limits

                    # implicit runs its own threads; one BLAS thread each avoids oversubscription
                    # (implicit warns at construction otherwise)
                    blas_limit = threadpool_limits(1, "blas")
                except ImportError:
                    blas_limit = nullcontext()
                with blas_limit:
                    model = AlternatingLeastSquares(
                        factors=self.factors, regularization=self.regularization, iterations=self.iterations,
                        random_state=self.seed, use_gpu=False,
                    )
                    model.fit(C, show_progress=False)
                return (np.asarray(model.user_factors, dtype=np.float32),
                        np.asarray(model.item_factors, dtype=np.float32), "implicit")
        X, Y = als_numpy(C, self.factors, self.regularization, self.iterations, self.seed)
        return X, Y, "numpy"

    def recommend(self, user_ids, k, exclude):
        result = {user_id: [] for user_id in user_ids}
        known = [u for u in user_ids if u in self.user_index]
        if not known or k <= 0:
            return result
        scores = self.user_factors[[self.user_index[u] for u in known]] @ self.item_factors.T
        for row, user_id in enumerate(known):
            row_scores = scores[row].copy()
            excluded = [self.item_index[b] for b in exclude.get(user_id, ()) if b in self.item_index]
            if excluded:
                row_scores[excluded] = -np.inf
            n = min(k, len(row_scores))
            top = np.argpartition(-row_scores, n - 1)[:n]
            top = top[np.argsort(-row_scores[top], kind="stable")]
            result[user_id] = [self.business_ids[i] for i in top if np.isfinite(row_scores[i])]
        return result

    def score(self, user_id, business_ids):
        if user_id not in self.user_index:
            return [float("nan")] * len(business_ids)
        u = self.user_factors[self.user_index[user_id]]
        return [float(u @ self.item_factors[self.item_index[b]]) if b in self.item_index else float("nan")
                for b in business_ids]


class ALS32(ALS):
    def __init__(self, **kwargs):
        super().__init__(factors=32, **kwargs)


class ALS128(ALS):
    def __init__(self, **kwargs):
        super().__init__(factors=128, **kwargs)
