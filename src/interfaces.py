from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

DATA = Path("data/processed")
RESTAURANTS = DATA / "restaurants.parquet"
REVIEWS = DATA / "reviews.parquet"
USERS = DATA / "users.parquet"
TRAIN_REVIEWS = DATA / "train_reviews.parquet"
VAL_REVIEWS = DATA / "val_reviews.parquet"
TEST_REVIEWS = DATA / "test_reviews.parquet"
EMBEDDINGS = DATA / "embeddings.npy"
EMBEDDINGS_BLOB = DATA / "embeddings_blob.npy"
BUSINESS_IDS = DATA / "business_ids.txt"
RESULTS_MD = Path("results.md")

RELEVANT_MIN_STARS = 4
SEED = 42


class Recommender(Protocol):
    """A restaurant recommender.

    `train` is TRAIN_REVIEWS (all reviews except val/test holdouts).
    `exclude` maps user_id -> business_ids to never recommend (their
    train-reviewed places).
    `score` returns higher-is-better predicted affinity.
    """

    name: str

    def fit(self, train: pd.DataFrame) -> None: ...

    def recommend(
        self, user_ids: list[str], k: int, exclude: dict[str, set[str]]
    ) -> dict[str, list[str]]: ...

    def score(self, user_id: str, business_ids: list[str]) -> list[float]: ...


def load_embeddings(path: Path = EMBEDDINGS) -> tuple[np.ndarray, list[str]]:
    matrix = np.load(path)
    with open(BUSINESS_IDS) as f:
        business_ids = [line.strip() for line in f]
    assert len(matrix) == len(business_ids)
    return matrix, business_ids
