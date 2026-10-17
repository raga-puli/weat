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
MAX_GROUPS = 3000
MAX_PER_TARGET = 20
VAL_FRACTION = 0.7

# Known gap (embedding leakage): each member's own review of the target is
# still in TRAIN_REVIEWS, so it feeds that member's taste profile (and the
# restaurant's embedding text), and the target sits in the member's
# train-reviewed `exclude` set.
# TODO: when building member profiles for group eval, drop each member's
# reviews of the group's target and don't exclude the target for them.


def make_group_id(members: list[str], target_business_id: str) -> str:
    key = "|".join(sorted(members)) + "@" + target_business_id
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def assign_split(group_id: str, seed: int = SEED, val_fraction: float = VAL_FRACTION) -> str:
    """Deterministic val/test assignment from (seed, group_id); independent
    of row order and of Python's salted hash().
    """
    h = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16)
    return "val" if h / 2**32 < val_fraction else "test"


def _candidate_groups(train, member_ids, rng, window_days, sizes, min_stars):
    """One random group per (anchor review, size): the anchor plus size-1
    other users whose liked review of the same restaurant falls within
    window_days after the anchor's. Enumerating every 3-4 subset of a
    window would explode for popular places, so each anchor is sampled.
    """
    liked = train[(train["stars"] >= min_stars) & train["user_id"].isin(member_ids)]
    # One review per (user, restaurant) so a user can't fill two slots.
    liked = liked.sort_values(["business_id", "date", "user_id"]).drop_duplicates(
        subset=["business_id", "user_id"], keep="first"
    )
    window = np.timedelta64(window_days, "D")

    candidates = []
    for business_id, rows in liked.groupby("business_id", sort=True):
        if len(rows) < min(sizes):
            continue
        users = rows["user_id"].to_numpy()
        dates = pd.to_datetime(rows["date"]).to_numpy("datetime64[ns]")
        ends = np.searchsorted(dates, dates + window, side="right")
        for i, end in enumerate(ends):
            n_others = end - i - 1
            for size in sizes:
                if n_others < size - 1:
                    continue
                picks = i + 1 + rng.choice(n_others, size - 1, replace=False)
                members = tuple(sorted([users[i], *users[picks]]))
                candidates.append((business_id, members, dates[i]))
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
) -> pd.DataFrame:
    """Synthetic dinner groups: 3-4 member_ids who all liked the same
    restaurant (the held-out target) within one window_days window.
    Deduplicated by member set, at most max_per_target groups per target,
    at most max_groups overall.
    """
    rng = np.random.default_rng(seed)
    candidates = _candidate_groups(train, member_ids, rng, window_days, sizes, min_stars)

    seen = set()
    per_target = Counter()
    rows = []
    # Visiting candidates in a seeded random order makes the dedup and the
    # caps an unbiased sample rather than favoring early restaurants/dates.
    for idx in rng.permutation(len(candidates)):
        target, members, window_start = candidates[idx]
        if members in seen or per_target[target] >= max_per_target:
            continue
        seen.add(members)
        per_target[target] += 1
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
