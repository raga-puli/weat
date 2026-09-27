from dataclasses import dataclass

import pandas as pd

from src.interfaces import (
    RELEVANT_MIN_STARS,
    REVIEWS,
    TEST_REVIEWS,
    TRAIN_REVIEWS,
    USERS,
    VAL_REVIEWS,
)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users_split: int
    repeat_reviews_removed: int
    users_with_repeat_reviews: int


def split_reviews(reviews: pd.DataFrame, users: pd.DataFrame) -> Split:
    """Time-based per-user val/test split, deduped so a held-out
    restaurant never also appears (via an earlier OR a later repeat
    review) in that user's training data.
    """
    dedup = reviews.sort_values(["date", "review_id"]).drop_duplicates(
        subset=["user_id", "business_id"], keep="first"
    )

    dedup_counts = dedup["user_id"].value_counts()
    eligible = set(users["user_id"]) & set(dedup_counts[dedup_counts >= 5].index)

    candidates = dedup[dedup["user_id"].isin(eligible)].sort_values(
        ["user_id", "date", "review_id"]
    )
    test_rows = candidates.groupby("user_id", sort=False).tail(1)
    remaining = candidates.drop(test_rows.index)
    val_rows = remaining.groupby("user_id", sort=False).tail(1)

    val_ids = set(val_rows["review_id"])
    test_ids = set(test_rows["review_id"])
    held_out = val_ids | test_ids

    # Exclude every review by a split user of their held-out (val or
    # test) business -- not just the two held-out review_ids -- so a
    # restaurant they visited more than once can't leak into train via
    # a duplicate review at some other date.
    exclude_pairs = set(zip(val_rows["user_id"], val_rows["business_id"]))
    exclude_pairs |= set(zip(test_rows["user_id"], test_rows["business_id"]))

    review_pairs = pd.MultiIndex.from_arrays([reviews["user_id"], reviews["business_id"]])
    exclude_index = pd.MultiIndex.from_tuples(
        exclude_pairs, names=["user_id", "business_id"]
    )
    excluded_mask = review_pairs.isin(exclude_index)

    old_train_mask = ~reviews["review_id"].isin(held_out)
    repeat_removed_mask = old_train_mask & excluded_mask
    repeat_reviews_removed = int(repeat_removed_mask.sum())
    users_with_repeat_reviews = int(reviews.loc[repeat_removed_mask, "user_id"].nunique())

    train = reviews[~excluded_mask]
    val = reviews[reviews["review_id"].isin(val_ids)]
    test = reviews[reviews["review_id"].isin(test_ids)]

    return Split(
        train=train,
        val=val,
        test=test,
        n_users_split=len(eligible),
        repeat_reviews_removed=repeat_reviews_removed,
        users_with_repeat_reviews=users_with_repeat_reviews,
    )


if __name__ == "__main__":
    reviews = pd.read_parquet(REVIEWS)
    users = pd.read_parquet(USERS)

    result = split_reviews(reviews, users)

    result.train.to_parquet(TRAIN_REVIEWS, index=False)
    result.val.to_parquet(VAL_REVIEWS, index=False)
    result.test.to_parquet(TEST_REVIEWS, index=False)

    val_relevant = (result.val["stars"] >= RELEVANT_MIN_STARS).mean()
    test_relevant = (result.test["stars"] >= RELEVANT_MIN_STARS).mean()

    print(f"users split: {result.n_users_split}")
    print(
        f"train: {len(result.train)}  val: {len(result.val)}  test: {len(result.test)}"
    )
    print(f"val relevant fraction (stars >= {RELEVANT_MIN_STARS}): {val_relevant:.4f}")
    print(f"test relevant fraction (stars >= {RELEVANT_MIN_STARS}): {test_relevant:.4f}")
    print(
        f"removed {result.repeat_reviews_removed} additional train reviews "
        f"(repeat visits to a held-out business) across "
        f"{result.users_with_repeat_reviews} users"
    )
