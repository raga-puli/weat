import pandas as pd

from src.groups import MIN_PRIOR_LIKES, assign_split, build_groups


def _review(user_id, business_id, day, stars=5):
    return {
        "user_id": user_id,
        "business_id": business_id,
        "stars": stars,
        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
    }


def _history(user, n=3):
    # n early likes, 100 days apart. Users' first likes (h0, h1, h2) can't
    # form groups themselves: nobody has 3 likes before day -100.
    return [_review(user, f"h{j}", -300 + 100 * j) for j in range(n)]


def _build_train():
    rows = []
    for i in range(1, 10):
        # u6 has only two prior likes -> never eligible.
        rows += _history(f"u{i}", n=2 if i == 6 else 3)

    # b1: u1-u6 all like it within days 0-20 -> many possible groups.
    for i, day in enumerate([0, 3, 7, 12, 18, 20]):
        rows.append(_review(f"u{i+1}", "b1", day))
    # u1 reviewed b1 twice; must still fill only one slot.
    rows.append(_review("u1", "b1", 5))
    # u7 likes b1 but 100+ days after everyone else -> never grouped there.
    rows.append(_review("u7", "b1", 130))
    # u8 is in the window but rated it 3 stars -> never a member.
    rows.append(_review("u8", "b1", 10, stars=3))
    # u9 is in the window and liked it, but isn't a candidate member.
    rows.append(_review("u9", "b1", 10))

    # b2: three users exactly 30 days apart edge-to-edge -> one group;
    # b3: same but 31 days -> no group.
    for user, day in [("u1", 0), ("u2", 15), ("u3", 30)]:
        rows.append(_review(user, "b2", day))
    for user, day in [("u1", 0), ("u2", 15), ("u3", 31)]:
        rows.append(_review(user, "b3", day))

    return pd.DataFrame(rows)


MEMBERS = {f"u{i}" for i in range(1, 8)} | {"u8"}


def _liked(train, user, business=None):
    rows = train[(train["user_id"] == user) & (train["stars"] >= 4)]
    if business is not None:
        rows = rows[rows["business_id"] == business]
    return rows


def test_groups_are_valid():
    train = _build_train()
    groups = build_groups(train, MEMBERS)

    assert len(groups) > 0
    for g in groups.itertuples():
        # no member appears twice in a group
        assert len(set(g.members)) == len(g.members) == g.size
        assert g.size in (3, 4)
        assert set(g.members) <= MEMBERS - {"u6", "u8"}
        end = g.window_start + pd.Timedelta(days=30)
        for user in g.members:
            # liked the target within [window_start, +30 days]
            dates = _liked(train, user, g.target_business_id)["date"]
            assert any(g.window_start <= d <= end for d in dates)
            # >= 3 likes strictly before the window, not counting the target
            prior = _liked(train, user)
            prior = prior[
                (prior["date"] < g.window_start)
                & (prior["business_id"] != g.target_business_id)
            ]
            assert len(prior) >= MIN_PRIOR_LIKES


def test_history_businesses_never_become_targets():
    groups = build_groups(_build_train(), MEMBERS)
    assert not groups["target_business_id"].str.startswith("h").any()


def test_member_sets_are_distinct():
    groups = build_groups(_build_train(), MEMBERS)
    member_sets = [frozenset(m) for m in groups["members"]]
    assert len(member_sets) == len(set(member_sets))
    assert groups["group_id"].is_unique


def test_window_boundary():
    train = _build_train()
    groups = build_groups(train, MEMBERS)
    assert "b3" not in set(groups["target_business_id"])
    # b2's only valid group can lose its member set to a b1 duplicate;
    # check it without b1.
    b2 = build_groups(train[train["business_id"] != "b1"], MEMBERS)
    assert list(b2.query("target_business_id == 'b2'")["members"]) == [["u1", "u2", "u3"]]


def test_prior_likes_must_be_strictly_before_window_start():
    def train_with_third_like_on(day):
        rows = []
        for user in ["a", "b", "c"]:
            rows += _history(user, n=2)
        rows += [_review("a", "pa", -10), _review("b", "pb", -10), _review("c", "pc", day)]
        # window starts at day 50 (a's like of bx)
        rows += [_review("a", "bx", 50), _review("b", "bx", 55), _review("c", "bx", 60)]
        return pd.DataFrame(rows)

    members = {"a", "b", "c"}
    same_day = build_groups(train_with_third_like_on(50), members)
    assert "bx" not in set(same_day["target_business_id"])
    day_before = build_groups(train_with_third_like_on(49), members)
    assert list(day_before.query("target_business_id == 'bx'")["members"]) == [["a", "b", "c"]]


def test_caps():
    train = _build_train()
    assert len(build_groups(train, MEMBERS, max_per_target=2).query("target_business_id == 'b1'")) == 2
    assert len(build_groups(train, MEMBERS, max_groups=3)) == 3


def test_per_user_cap():
    train = _build_train()
    uncapped = build_groups(train, MEMBERS)
    assert uncapped["members"].explode().value_counts().max() > 2

    capped = build_groups(train, MEMBERS, max_per_user=2)
    assert len(capped) > 0
    assert capped["members"].explode().value_counts().max() <= 2


def test_split_is_deterministic():
    train = _build_train()
    a = build_groups(train, MEMBERS)
    b = build_groups(train, MEMBERS)
    pd.testing.assert_frame_equal(a, b)

    ids = [f"g{i}" for i in range(2000)]
    splits = [assign_split(g) for g in ids]
    assert splits == [assign_split(g) for g in ids]
    assert set(splits) == {"val", "test"}
    assert 0.65 < splits.count("val") / len(splits) < 0.75
