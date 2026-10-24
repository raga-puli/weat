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
from src.aggregate import LAMBDA, TAU, STRATEGIES as AGGREGATORS, fairness_repeated, normalize
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


def data_source() -> str:
    """DATA_SOURCE from the environment (.env): "files" (default; parquet/npy under
    data/processed) or "db" (everything from DATABASE_URL, as filled by src.deploy_load)."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    source = os.getenv("DATA_SOURCE", "files").strip().lower()
    if source not in ("files", "db"):
        raise RuntimeError(f"DATA_SOURCE must be 'files' or 'db', not {source!r}")
    return source


@lru_cache(maxsize=1)
def _db_tables() -> dict[str, pd.DataFrame]:
    """Every table the group API needs, read once from DATABASE_URL."""
    from src.db import get_conn

    queries = {
        # byte order, the same row order as business_ids.txt
        "restaurants": 'SELECT business_id, name, latitude, longitude, price_level, categories, is_open, '
                       'embedding FROM restaurants ORDER BY business_id COLLATE "C"',
        "user_taste": "SELECT user_id, embedding FROM user_taste ORDER BY user_id",
        "model_arrays": "SELECT name, model, n, vals FROM model_arrays",
        "user_profiles": "SELECT user_id, rank, business_id, stars FROM user_profiles ORDER BY user_id, rank",
        "user_visited": "SELECT user_id, business_id FROM user_visited ORDER BY user_id, business_id",
        "sample_groups": "SELECT group_id, members, cohesion, cohesion_bucket FROM sample_groups ORDER BY group_id",
    }
    out = {}
    with get_conn(connect_timeout=15) as conn, conn.cursor() as cur:
        for name, sql in queries.items():
            cur.execute(sql)
            out[name] = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    return out


def _as_array(value) -> np.ndarray:
    """A pgvector value as float32: Vector objects (pgvector >= 0.4) or plain lists/arrays."""
    return np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value, dtype=np.float32)


def model_from_tables(tables: dict[str, pd.DataFrame]):
    """The served model rebuilt from DB tables: raw embeddings -> RECOMMENDER's
    transform (centering), then the stored taste vectors and hub."""
    r = tables["restaurants"]
    matrix = np.stack([_as_array(e) for e in r["embedding"]])
    model = RECOMMENDER(matrix=matrix, business_ids=r["business_id"].tolist())
    model.taste = {u: _as_array(e)
                   for u, e in zip(tables["user_taste"]["user_id"], tables["user_taste"]["embedding"])}
    if hasattr(model, "hub"):
        arrays = tables["model_arrays"].set_index("name")
        if "hub" not in arrays.index or arrays.loc["hub", "model"] != model.name:
            raise RuntimeError(f"model_arrays has no hub for {model.name}; rerun python -m src.deploy_load")
        hub = np.asarray(arrays.loc["hub", "vals"], dtype=np.float32)
        if hub.shape != model.hub.shape:
            raise RuntimeError(f"stored hub has {hub.shape[0]} entries, catalog has {model.hub.shape[0]}")
        model.hub = hub
    return model


@lru_cache(maxsize=1)
def _serving_train() -> pd.DataFrame:
    """recommend_lib.serving_train(), read once per process."""
    return recommend_lib.serving_train()


@lru_cache(maxsize=1)
def get_recommender() -> ContentBasedCentered:
    """The served model. files: restored from data/processed/taste_cache_<name>.npz
    when the cache's fingerprint matches; otherwise fit (cold path: logged as a
    warning) and the cache written. Identical state either way. Build the cache
    ahead of time with `python -m src.recommend build-cache --model <class>`.
    db: rebuilt from the database (taste vectors for the demo users only)."""
    if data_source() == "db":
        return model_from_tables(_db_tables())
    model = RECOMMENDER()
    recommend_lib.load_or_fit(model, train_fn=_serving_train)
    return model


def restaurants_from_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    r = tables["restaurants"]
    return pd.DataFrame(
        {
            "name": r["name"].to_numpy(),
            "categories": [None if c is None else ", ".join(c) for c in r["categories"]],
            "price_level": [None if p is None or pd.isna(p) else int(p) for p in r["price_level"]],
            "latitude": r["latitude"].astype(float).to_numpy(),
            "longitude": r["longitude"].astype(float).to_numpy(),
            "is_open": r["is_open"].fillna(False).astype(bool).to_numpy(),
        },
        index=pd.Index(r["business_id"], name="business_id"),
    )


def user_reviews_from_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """user_id -> (business_id, stars): the top-5 profile rows with their stars,
    then the rest of the visited set with stars unknown (NaN sorts after every
    star rating, so profiles and exclusions match files mode)."""
    profiles = tables["user_profiles"]
    visited = tables["user_visited"]
    out = {}
    for user_id, v in visited.groupby("user_id", sort=False):
        p = profiles[profiles["user_id"] == user_id].sort_values("rank")
        rest = sorted(set(v["business_id"]) - set(p["business_id"]))
        out[user_id] = pd.DataFrame({
            "business_id": list(p["business_id"]) + rest,
            "stars": list(p["stars"].astype(float)) + [np.nan] * len(rest),
        })
    return out


@lru_cache(maxsize=1)
def get_restaurants() -> pd.DataFrame:
    """business_id -> name, categories, price_level, latitude, longitude, is_open."""
    if data_source() == "db":
        return restaurants_from_tables(_db_tables())
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
    if data_source() == "db":
        return user_reviews_from_tables(_db_tables())
    train = _serving_train().sort_values("user_id", kind="stable")
    users = train["user_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    reviews = train[["business_id", "stars"]].reset_index(drop=True)
    bounds = np.r_[starts, len(train)]
    return {users[s]: reviews.iloc[s:e] for s, e in zip(bounds[:-1], bounds[1:])}


@lru_cache(maxsize=1)
def get_val_users() -> tuple[str, ...]:
    """Users /users/sample may draw (cohesion=any). db: the demo users."""
    if data_source() == "db":
        return tuple(sorted(_db_tables()["user_taste"]["user_id"]))
    return tuple(sorted(set(pd.read_parquet(interfaces.VAL_REVIEWS, columns=["user_id"])["user_id"])))


@lru_cache(maxsize=1)
def get_val_groups() -> pd.DataFrame:
    """Groups /users/sample draws from for cohesion=high/low. db: the precomputed
    sample_groups, which carry their own cohesion_bucket."""
    if data_source() == "db":
        g = _db_tables()["sample_groups"]
        return g.assign(members=g["members"].map(list)).reset_index(drop=True)
    groups = pd.read_parquet(GROUPS)
    return groups[groups["split"] == "val"].reset_index(drop=True)


# --- request / response models ----------------------------------------------------


class GroupParams(BaseModel):
    members: list[str] = Field(min_length=MIN_MEMBERS, max_length=MAX_MEMBERS)
    norm: Norm = "rank"
    lam: float = Field(LAMBDA, ge=0)
    tau: float = Field(
        TAU, gt=0,
        description="fairness_repeated temperature (member weights = softmax(-cumulative score / tau)). "
                    "The API default, 0.1, is the pre-registered evaluation setting. The Group Dinner page "
                    "defaults to 0.3 instead, because 0.3 performed better per dinner on validation "
                    "(README: τ sensitivity).",
    )
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
                                                 tau=params.tau, return_weights=True)
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
        "params": {"lam": params.lam, "tau": params.tau, "k": params.k, "exclude_visited": params.exclude_visited,
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
        if "cohesion_bucket" in groups.columns:  # db: sample_groups were bucketed at load time
            bucket = groups[groups["cohesion_bucket"] == cohesion]
        else:
            values = groups["members"].map(lambda ms: _cohesion(np.stack([model.taste[m] for m in ms])))
            median = float(values.median())
            bucket = groups[values > median] if cohesion == "high" else groups[values <= median]
        if bucket.empty:
            raise HTTPException(status_code=422, detail=f"no {cohesion}-cohesion group has {n} or more members")
        row = bucket.iloc[int(rng.integers(len(bucket)))]
        group_id = row["group_id"]
        users = sorted(rng.choice(list(row["members"]), size=n, replace=False).tolist())

    response = {"cohesion_bucket": cohesion, "seed": seed,
                "users": [_profile(u, restaurants, user_reviews) for u in users]}
    if group_id is not None:
        response["group_id"] = group_id
        response["cohesion"] = _cohesion(np.stack([model.taste[u] for u in users]))
    return response


def _profile(user_id: str, restaurants, user_reviews) -> dict:
    """A user's PROFILE_SIZE highest-rated train restaurants (ties by business_id)."""
    reviews = user_reviews.get(user_id, pd.DataFrame(columns=["business_id", "stars"]))
    top = reviews.sort_values(["stars", "business_id"], ascending=[False, True]).head(PROFILE_SIZE)
    return {
        "user_id": user_id,
        "profile": [
            {"business_id": b, "name": restaurants["name"].get(b), "stars": float(s)}
            for b, s in zip(top["business_id"], top["stars"])
        ],
    }


@router.get("/users/profiles")
def user_profiles(
    user_ids: list[str] = Query(..., min_length=1, max_length=MAX_MEMBERS),
    model=Depends(get_recommender),
    restaurants=Depends(get_restaurants),
    user_reviews=Depends(get_user_reviews),
) -> dict:
    """Profiles ("top-rated places") for typed-in user_ids; 422 naming any
    unknown user or user without a taste profile."""
    _profiles(model, list(user_ids))
    return {"users": [_profile(u, restaurants, user_reviews) for u in user_ids]}


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
