"""Group recommendation endpoints (Person A).

Scoring runs in numpy through src/aggregate.py (not the SQL path in
src/db_groups.py), because only the numpy path supports per-member
normalization. The model is ContentBasedDebiased: ContentBasedCentered (the
README's headline configuration) plus the hubness correction described there.
Taste vectors are each user's whole train history, for users with 5+ reviews
(users.parquet); other user_ids get a 422. The hub vector is the mean over those
served users, not over all train users as in src.eval / group_eval.

Heavy resources come from lru_cache getters used as dependencies, so tests can
override them and nothing loads at import time. Closed restaurants are excluded
unless include_closed=true (CLAUDE.md: user-facing paths exclude them by default).
"""

from functools import lru_cache
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from src import interfaces
from src.aggregate import LAMBDA, STRATEGIES as AGGREGATORS, fairness_repeated, normalize
from src.groups import GROUPS
from src import recommend as recommend_lib  # not `recommend`: that's the POST /recommend handler below
from src.recommend import ContentBasedCentered, ContentBasedDebiased

router = APIRouter(prefix="/group", tags=["group"])

Strategy = Literal["average", "least_misery", "fairness", "fairness_repeated"]
Norm = Literal["raw", "zscore", "rank"]
PROFILE_SIZE = 5
MIN_MEMBERS, MAX_MEMBERS = 2, 6
# The served model. All scoring goes through model.scores_for (the per-user
# score function), so switching to e.g. ContentBasedDebiased is this one line.
RECOMMENDER = ContentBasedDebiased


# --- dependencies --------------------------------------------------------------


@lru_cache(maxsize=1)
def _serving_train() -> pd.DataFrame:
    """recommend_lib.serving_train(), read once per process."""
    return recommend_lib.serving_train()


@lru_cache(maxsize=1)
def get_recommender() -> ContentBasedCentered:
    """The served model, restored from data/processed/taste_cache_<name>.npz
    when the cache's fingerprint matches; otherwise fit (cold path: logged as a
    warning) and the cache written. Identical state either way. Build the cache
    ahead of time with `python -m src.recommend build-cache --model <class>`."""
    model = RECOMMENDER()
    recommend_lib.load_or_fit(model, train_fn=_serving_train)
    return model


@lru_cache(maxsize=1)
def get_restaurants() -> pd.DataFrame:
    """business_id -> name, categories, price_level, latitude, longitude, is_open."""
    from src.db import _as_dict, parse_price_level

    r = pd.read_parquet(interfaces.RESTAURANTS)
    if "is_open" not in r.columns:
        raise RuntimeError("restaurants.parquet has no is_open column; rerun python -m src.prep --stage restaurants")
    price = [parse_price_level((_as_dict(a) or {}).get("RestaurantsPriceRange2")) for a in r["attributes"]]
    return pd.DataFrame(
        {
            "name": r["name"].to_numpy(),
            "categories": r["categories"].to_numpy(),
            "price_level": price,
            "latitude": r["latitude"].to_numpy(),
            "longitude": r["longitude"].to_numpy(),
            "is_open": r["is_open"].astype(bool).to_numpy(),
        },
        index=pd.Index(r["business_id"], name="business_id"),
    )


@lru_cache(maxsize=1)
def get_user_reviews() -> dict[str, pd.DataFrame]:
    """user_id -> that user's train reviews (business_id, stars), for served users."""
    train = _serving_train().sort_values("user_id", kind="stable")
    users = train["user_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    reviews = train[["business_id", "stars"]].reset_index(drop=True)
    bounds = np.r_[starts, len(train)]
    return {users[s]: reviews.iloc[s:e] for s, e in zip(bounds[:-1], bounds[1:])}


@lru_cache(maxsize=1)
def get_val_users() -> tuple[str, ...]:
    return tuple(sorted(set(pd.read_parquet(interfaces.VAL_REVIEWS, columns=["user_id"])["user_id"])))


@lru_cache(maxsize=1)
def get_val_groups() -> pd.DataFrame:
    groups = pd.read_parquet(GROUPS)
    return groups[groups["split"] == "val"].reset_index(drop=True)


# --- request / response models ----------------------------------------------------


class GroupParams(BaseModel):
    members: list[str] = Field(min_length=MIN_MEMBERS, max_length=MAX_MEMBERS)
    norm: Norm = "rank"
    lam: float = Field(LAMBDA, ge=0)
    k: int = Field(10, ge=1, le=50)
    exclude_visited: bool = True
    include_closed: bool = False

    @field_validator("members")
    @classmethod
    def _distinct(cls, members: list[str]) -> list[str]:
        duplicates = sorted({m for m in members if members.count(m) > 1})
        if duplicates:
            raise ValueError(f"members must be distinct; repeated: {duplicates}")
        return members


class RecommendRequest(GroupParams):
    strategy: Strategy = "least_misery"


class CompareRequest(GroupParams):
    strategies: list[Strategy] = Field(min_length=2, max_length=2)


# --- core -------------------------------------------------------------------------


def _profiles(model, members: list[str]) -> np.ndarray:
    known = [m for m in members if m in model.taste]
    missing = [m for m in members if m not in model.taste]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"unknown user_ids or users without a taste profile: {missing}",
        )
    return np.stack([model.taste[m] for m in known])


def _cohesion(tastes: np.ndarray) -> float:
    pairs = list(combinations(range(len(tastes)), 2))
    return float(np.mean([tastes[i] @ tastes[j] for i, j in pairs]))


def _excluded(model, restaurants, user_reviews, members, exclude_visited, include_closed) -> np.ndarray:
    excluded = np.zeros(len(model.business_ids), dtype=bool)
    if exclude_visited:
        for m in members:
            reviews = user_reviews.get(m)
            if reviews is not None:
                excluded[[model.index[b] for b in reviews["business_id"] if b in model.index]] = True
    if not include_closed:
        is_open = restaurants["is_open"].reindex(model.business_ids)
        excluded |= ~is_open.fillna(False).to_numpy(dtype=bool)
    return excluded


def _clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _price(value) -> int | None:
    """1-4, or None. The table column is float when some levels are missing."""
    value = _clean(value)
    return None if value is None or pd.isna(value) else int(value)


def recommend_group(params: GroupParams, strategy: str, model, restaurants, user_reviews) -> dict:
    members = list(params.members)
    tastes = _profiles(model, members)
    excluded = _excluded(model, restaurants, user_reviews, members,
                         params.exclude_visited, params.include_closed)
    if excluded.all():
        raise HTTPException(status_code=422, detail="no candidate restaurants left after exclusions")

    scores = np.stack([model.scores_for(t) for t in tastes])
    S = normalize(scores, excluded, params.norm)
    history = None
    if strategy == "fairness_repeated":
        picks, hist, weights = fairness_repeated(scores, excluded, params.k, norm=params.norm,
                                                 return_weights=True)
        aggregate_scores = [float(weights[t] @ S[:, p]) for t, p in enumerate(picks)]
        history = [
            {
                "dinner": t + 1,
                "business_id": model.business_ids[p],
                "member_scores": dict(zip(members, map(float, hist[t]))),
                "member_weights": dict(zip(members, map(float, weights[t]))),
            }
            for t, p in enumerate(picks)
        ]
    else:
        kwargs = {"lam": params.lam} if strategy == "fairness" else {}
        picks = AGGREGATORS[strategy](scores, excluded, params.k, norm=params.norm, **kwargs)
        per = S[:, picks]
        aggregate_scores = {
            "average": per.mean(axis=0),
            "least_misery": per.min(axis=0),
            "fairness": per.mean(axis=0) - params.lam * (per.max(axis=0) - per.min(axis=0)),
        }[strategy].tolist()

    restaurants_out = []
    for p, agg in zip(picks, aggregate_scores):
        business_id = model.business_ids[p]
        info = restaurants.loc[business_id] if business_id in restaurants.index else None
        restaurants_out.append(
            {
                "business_id": business_id,
                "name": None if info is None else info["name"],
                "categories": None if info is None else _clean(info["categories"]),
                "price_level": None if info is None else _price(info["price_level"]),
                "lat": None if info is None else _clean(info["latitude"]),
                "lon": None if info is None else _clean(info["longitude"]),
                "score": float(agg),
                "member_scores": dict(zip(members, map(float, S[:, p]))),
            }
        )

    member_means = S[:, picks].mean(axis=1) if len(picks) else np.full(len(members), np.nan)
    result = {
        "strategy": strategy,
        "norm": params.norm,
        "params": {"lam": params.lam, "k": params.k, "exclude_visited": params.exclude_visited,
                   "include_closed": params.include_closed},
        "members": members,
        "cohesion": _cohesion(tastes),
        "n_candidates": int((~excluded).sum()),
        "restaurants": restaurants_out,
        "summary": {
            "note": f"PREDICTED from taste profiles ({params.norm}-normalized member scores), not observed satisfaction",
            "group_utility_predicted": float(member_means.mean()),
            "worst_member_predicted": float(member_means.min()),
            "fairness_gap_predicted": float(member_means.max() - member_means.min()),
            "member_means_predicted": dict(zip(members, map(float, member_means))),
        },
    }
    if history is not None:
        result["history"] = history
    return result


# --- endpoints ----------------------------------------------------------------------


@router.get("/users/sample")
def sample_users(
    n: int = Query(4, ge=MIN_MEMBERS, le=MAX_MEMBERS),
    cohesion: Literal["any", "high", "low"] = "any",
    seed: int = interfaces.SEED,
    model=Depends(get_recommender),
    restaurants=Depends(get_restaurants),
    user_reviews=Depends(get_user_reviews),
    val_users=Depends(get_val_users),
    val_groups=Depends(get_val_groups),
) -> dict:
    """n seeded-random val users, each with a 5-restaurant "profile" (their
    highest-rated train restaurants), since Yelp users are anonymous. For
    high/low cohesion, the users come from one existing val group in that half
    of the cohesion distribution (median split over val groups). Cohesion here
    uses whole-history taste profiles; group_eval uses pre-window profiles, so
    its buckets can differ."""
    rng = np.random.default_rng(seed)
    group_id = None
    if cohesion == "any":
        pool = [u for u in val_users if u in model.taste]
        if len(pool) < n:
            raise HTTPException(status_code=422, detail=f"only {len(pool)} val users have taste profiles")
        users = sorted(rng.choice(pool, size=n, replace=False).tolist())
    else:
        groups = val_groups[val_groups["members"].map(len) >= n]
        groups = groups[groups["members"].map(lambda ms: all(m in model.taste for m in ms))]
        if groups.empty:
            raise HTTPException(status_code=422, detail=f"no val group has {n} or more members")
        values = groups["members"].map(lambda ms: _cohesion(np.stack([model.taste[m] for m in ms])))
        median = float(values.median())
        bucket = groups[values > median] if cohesion == "high" else groups[values <= median]
        row = bucket.iloc[int(rng.integers(len(bucket)))]
        group_id = row["group_id"]
        users = sorted(rng.choice(list(row["members"]), size=n, replace=False).tolist())

    out = []
    for u in users:
        reviews = user_reviews.get(u, pd.DataFrame(columns=["business_id", "stars"]))
        top = reviews.sort_values(["stars", "business_id"], ascending=[False, True]).head(PROFILE_SIZE)
        out.append(
            {
                "user_id": u,
                "profile": [
                    {"business_id": b,
                     "name": restaurants["name"].get(b),
                     "stars": float(s)}
                    for b, s in zip(top["business_id"], top["stars"])
                ],
            }
        )
    response = {"cohesion_bucket": cohesion, "seed": seed, "users": out}
    if group_id is not None:
        response["group_id"] = group_id
        response["cohesion"] = _cohesion(np.stack([model.taste[u] for u in users]))
    return response


@router.post("/recommend")
def recommend(
    request: RecommendRequest,
    model=Depends(get_recommender),
    restaurants=Depends(get_restaurants),
    user_reviews=Depends(get_user_reviews),
) -> dict:
    return recommend_group(request, request.strategy, model, restaurants, user_reviews)


@router.post("/compare")
def compare(
    request: CompareRequest,
    model=Depends(get_recommender),
    restaurants=Depends(get_restaurants),
    user_reviews=Depends(get_user_reviews),
) -> dict:
    results = [recommend_group(request, s, model, restaurants, user_reviews) for s in request.strategies]
    a, b = ({r["business_id"] for r in res["restaurants"]} for res in results)
    return {"strategies": list(request.strategies), "results": results, "overlap": len(a & b)}
