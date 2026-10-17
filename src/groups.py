import hashlib
from collections import Counter

import numpy as np
import pandas as pd

from src.interfaces import (
    DATA,
    RELEVANT_MIN_STARS,
    RESTAURANTS,
    SEED,
    TRAIN_REVIEWS,
    VAL_REVIEWS,
)

GROUPS = DATA / "groups.parquet"

WINDOW_DAYS = 30
GROUP_SIZES = (3, 4)
MAX_GROUPS = 12000
MAX_PER_TARGET = 20
# Keeps heavy reviewers from dominating (and correlating) the evaluation.
MAX_GROUPS_PER_USER = 10
VAL_FRACTION = 0.7
# A user may join a window only with this many 4+ star train reviews dated
# strictly before the window's start (group_eval's min-history rule).
MIN_PRIOR_LIKES = 3

# Known gap (embedding leakage): each member's own review of the target is
# still in TRAIN_REVIEWS, so it can be in the restaurant's embedding text.
# group_eval keeps it out of member profiles and candidates (time cut at
# window_start) and measures the embedding side with its leak diagnostic.
# TODO: fix the embedding side (re-encode targets without members' reviews).


def make_group_id(members: list[str], target_business_id: str) -> str:
    key = "|".join(sorted(members)) + "@" + target_business_id
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def assign_split(group_id: str, seed: int = SEED, val_fraction: float = VAL_FRACTION) -> str:
    """Deterministic val/test assignment from (seed, group_id); independent
    of row order and of Python's salted hash().
    """
    h = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16)
    return "val" if h / 2**32 < val_fraction else "test"


def _nth_like_dates(liked, n):
    """user_id -> date of their n-th liked review; users with fewer than n
    likes are absent."""
    nth = liked.sort_values(["user_id", "date"]).groupby("user_id", sort=False).nth(n - 1)
    return dict(zip(nth["user_id"], pd.to_datetime(nth["date"]).to_numpy("datetime64[ns]")))


def _candidate_groups(train, member_ids, rng, window_days, sizes, min_stars, min_prior_likes):
    """One random group per (anchor review, size): the anchor plus size-1
    other users whose liked review of the same restaurant falls within
    window_days after the anchor's. Enumerating every 3-4 subset of a
    window would explode for popular places, so each anchor is sampled.

    Only users with >= min_prior_likes liked reviews dated strictly before
    the window's start (the anchor's date) may join, anchor included. That
    is: their min_prior_likes-th like is earlier than the window start.
    Those prior likes never include the target, because every user's
    earliest like of the target is at or after the window start.
    """
    liked_all = train[(train["stars"] >= min_stars) & train["user_id"].isin(member_ids)]
    nth_like = _nth_like_dates(liked_all, min_prior_likes)
    never = np.datetime64("2262-01-01", "ns")

    # One review per (user, restaurant) so a user can't fill two slots.
    liked = liked_all.sort_values(["business_id", "date", "user_id"]).drop_duplicates(
        subset=["business_id", "user_id"], keep="first"
    )
    window = np.timedelta64(window_days, "D")

    candidates = []
    for business_id, rows in liked.groupby("business_id", sort=True):
        if len(rows) < min(sizes):
            continue
        users = rows["user_id"].to_numpy()
        dates = pd.to_datetime(rows["date"]).to_numpy("datetime64[ns]")
        ready = np.array([nth_like.get(u, never) for u in users], dtype="datetime64[ns]")
        ends = np.searchsorted(dates, dates + window, side="right")
        for i, end in enumerate(ends):
            start = dates[i]
            if not ready[i] < start:
                continue
            others = i + 1 + np.flatnonzero(ready[i + 1 : end] < start)
            for size in sizes:
                if len(others) < size - 1:
                    continue
                picks = rng.choice(others, size - 1, replace=False)
                members = tuple(sorted([users[i], *users[picks]]))
                candidates.append((business_id, members, start))
    return candidates


def build_groups(
    train: pd.DataFrame,
    member_ids: set[str],
    seed: int = SEED,
    max_groups: int = MAX_GROUPS,
    max_per_target: int = MAX_PER_TARGET,
    window_days: int = WINDOW_DAYS,
    sizes: tuple[int, ...] = GROUP_SIZES,
    min_stars: float = RELEVANT_MIN_STARS,
    min_prior_likes: int = MIN_PRIOR_LIKES,
    max_per_user: int = MAX_GROUPS_PER_USER,
) -> pd.DataFrame:
    """Synthetic dinner groups: 3-4 member_ids who all liked the same
    restaurant (the held-out target) within one window_days window, each
    with >= min_prior_likes likes before the window. Deduplicated by member
    set, at most max_per_target groups per target, at most max_per_user
    groups per user, at most max_groups overall.
    """
    rng = np.random.default_rng(seed)
    candidates = _candidate_groups(
        train, member_ids, rng, window_days, sizes, min_stars, min_prior_likes
    )

    seen = set()
    per_target = Counter()
    per_user = Counter()
    rows = []
    # Visiting candidates in a seeded random order makes the dedup and the
    # caps an unbiased sample rather than favoring early restaurants/dates.
    for idx in rng.permutation(len(candidates)):
        target, members, window_start = candidates[idx]
        if (
            members in seen
            or per_target[target] >= max_per_target
            or any(per_user[u] >= max_per_user for u in members)
        ):
            continue
        seen.add(members)
        per_target[target] += 1
        per_user.update(members)
        rows.append(
            {
                "group_id": make_group_id(list(members), target),
                "members": list(members),
                "target_business_id": target,
                "window_start": pd.Timestamp(window_start),
                "size": len(members),
            }
        )
        if len(rows) == max_groups:
            break

    groups = pd.DataFrame(
        rows, columns=["group_id", "members", "target_business_id", "window_start", "size"]
    )
    groups["split"] = [assign_split(g, seed) for g in groups["group_id"]]
    return groups


if __name__ == "__main__":
    train = pd.read_parquet(TRAIN_REVIEWS, columns=["user_id", "business_id", "stars", "date"])
    val_users = set(pd.read_parquet(VAL_REVIEWS, columns=["user_id"])["user_id"])

    groups = build_groups(train, val_users)
    groups.to_parquet(GROUPS, index=False)

    print(f"groups: {len(groups)}")
    print(pd.crosstab(groups["size"], groups["split"], margins=True).to_string())
    print(f"distinct targets: {groups['target_business_id'].nunique()}")

    restaurants = pd.read_parquet(RESTAURANTS, columns=["business_id", "name", "review_count"])
    top = (
        groups["target_business_id"]
        .value_counts()
        .head(10)
        .rename_axis("business_id")
        .reset_index(name="n_groups")
        .merge(restaurants, on="business_id", how="left")
    )
    print(top[["name", "n_groups", "review_count"]].to_string(index=False))
