import pandas as pd

from src.split import split_reviews


def _review(review_id, user_id, business_id, day, stars=4):
    return {
        "review_id": review_id,
        "user_id": user_id,
        "business_id": business_id,
        "stars": stars,
        "text": "some review text",
        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
    }


def _build_reviews():
    rows = []

    # u1: exactly 5 reviews, 5 distinct businesses -> boundary-eligible.
    for i, day in enumerate([1, 2, 3, 4, 5]):
        rows.append(_review(f"u1-r{i}", "u1", f"u1-b{i}", day))

    # u2: 6 raw reviews but a repeat visit to b0 (days 1 and 2) -> 5
    # distinct businesses after dedup -> eligible, and exercises the
    # dedup-keeps-earliest rule.
    rows.append(_review("u2-r0a", "u2", "u2-b0", 1))
    rows.append(_review("u2-r0b", "u2", "u2-b0", 2))
    for i, day in enumerate([3, 4, 5, 6]):
        rows.append(_review(f"u2-r{i+1}", "u2", f"u2-b{i+1}", day))

    # u3: 5 raw reviews (qualifies on raw count) but two are repeat
    # visits to the same business -> only 4 distinct businesses after
    # dedup -> must NOT be split, everything stays in train.
    rows.append(_review("u3-r0a", "u3", "u3-b0", 1))
    rows.append(_review("u3-r0b", "u3", "u3-b0", 2))
    for i, day in enumerate([3, 4, 5]):
        rows.append(_review(f"u3-r{i+1}", "u3", f"u3-b{i+1}", day))

    # u4: only 3 reviews, below the raw threshold -> never appears in
    # the users table -> must never be split.
    for i, day in enumerate([1, 2, 3]):
        rows.append(_review(f"u4-r{i}", "u4", f"u4-b{i}", day))

    # u5: 5 distinct businesses (b1..b5), so b5 becomes their test
    # business (day 5) and b4 their val business (day 4). u5 then
    # revisits b5 again later (day 10) -- a duplicate review of the
    # held-out business that the old review_id-only exclusion would
    # have left sitting in train.
    for i, day in enumerate([1, 2, 3, 4, 5]):
        rows.append(_review(f"u5-r{i+1}", "u5", f"u5-b{i+1}", day))
    rows.append(_review("u5-r5b", "u5", "u5-b5", 10))

    return pd.DataFrame(rows)


def _build_users():
    # Mirrors users.parquet: only users with 5+ raw reviews are
    # present at all (u4 is excluded here, same as prep.py's HAVING).
    return pd.DataFrame({"user_id": ["u1", "u2", "u3", "u5"]})


def test_no_review_id_in_more_than_one_split():
    result = split_reviews(_build_reviews(), _build_users())

    train_ids = set(result.train["review_id"])
    val_ids = set(result.val["review_id"])
    test_ids = set(result.test["review_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_each_eligible_user_has_exactly_one_val_and_test_row():
    result = split_reviews(_build_reviews(), _build_users())

    val_counts = result.val["user_id"].value_counts()
    test_counts = result.test["user_id"].value_counts()

    assert set(val_counts.index) == {"u1", "u2", "u5"}
    assert set(test_counts.index) == {"u1", "u2", "u5"}
    assert (val_counts == 1).all()
    assert (test_counts == 1).all()
    assert result.n_users_split == 3

    # u3 qualifies on raw count but drops below 5 after dedup, and
    # must not be split.
    assert "u3" not in val_counts.index
    assert "u3" not in test_counts.index


def test_test_date_after_val_date_after_max_train_date_per_user():
    result = split_reviews(_build_reviews(), _build_users())

    for user_id in ["u1", "u2", "u5"]:
        val_date = result.val.loc[
            result.val["user_id"] == user_id, "date"
        ].iloc[0]
        test_date = result.test.loc[
            result.test["user_id"] == user_id, "date"
        ].iloc[0]
        max_train_date = result.train.loc[
            result.train["user_id"] == user_id, "date"
        ].max()

        assert test_date >= val_date >= max_train_date


def test_repeat_review_of_held_out_business_is_fully_removed_from_train():
    result = split_reviews(_build_reviews(), _build_users())

    # u5's test business is b5 (day 5); they reviewed it again on day
    # 10, and that duplicate must not leak into train either.
    assert not (
        (result.train["user_id"] == "u5") & (result.train["business_id"] == "u5-b5")
    ).any()
    assert "u5-r5b" not in set(result.train["review_id"])
    assert (result.train["user_id"] == "u5").sum() == 3  # only b1, b2, b3 remain

    assert result.repeat_reviews_removed == 1
    assert result.users_with_repeat_reviews == 1


def test_users_with_fewer_than_5_reviews_appear_only_in_train():
    result = split_reviews(_build_reviews(), _build_users())

    for user_id in ["u3", "u4"]:
        assert (result.train["user_id"] == user_id).sum() == (
            (_build_reviews()["user_id"] == user_id).sum()
        )
        assert user_id not in set(result.val["user_id"])
        assert user_id not in set(result.test["user_id"])
