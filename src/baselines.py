import hashlib
import math
import random

import pandas as pd

from src import interfaces


def _seeded_rng(user_id: str) -> random.Random:
    digest = hashlib.sha256(f"{interfaces.SEED}:{user_id}".encode()).hexdigest()
    return random.Random(int(digest, 16))


class Random:
    name = "random"

    def fit(self, train: pd.DataFrame) -> None:
        restaurants = pd.read_parquet(interfaces.RESTAURANTS)
        self.catalog = restaurants["business_id"].tolist()

    def recommend(self, user_ids, k, exclude):
        recs = {}
        for user_id in user_ids:
            excluded = exclude.get(user_id, set())
            candidates = [b for b in self.catalog if b not in excluded]
            rng = _seeded_rng(user_id)
            recs[user_id] = rng.sample(candidates, min(k, len(candidates)))
        return recs

    def score(self, user_id, business_ids):
        return [_seeded_rng(f"{user_id}:{b}").random() for b in business_ids]


class Popularity:
    name = "popularity"

    def fit(self, train: pd.DataFrame) -> None:
        self.counts = train.groupby("business_id").size().to_dict()

    def recommend(self, user_ids, k, exclude):
        ranked_ids = [
            business_id
            for business_id, _ in sorted(
                self.counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ]
        recs = {}
        for user_id in user_ids:
            excluded = exclude.get(user_id, set())
            candidates = [b for b in ranked_ids if b not in excluded]
            recs[user_id] = candidates[:k]
        return recs

    def score(self, user_id, business_ids):
        return [math.log1p(self.counts.get(bid, 0)) for bid in business_ids]
