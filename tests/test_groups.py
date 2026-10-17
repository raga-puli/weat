import pandas as pd

from src.groups import assign_split, build_groups


def _review(user_id, business_id, day, stars=5):
    return {
        "user_id": user_id,
        "business_id": business_id,
        "stars": stars,
        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
    }


def _build_train():
    rows = []

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


def _liked_dates(train, user, business):
    rows = train[
        (train["user_id"] == user) & (train["business_id"] == business) & (train["stars"] >= 4)
    ]
    return list(rows["date"])


def test_groups_are_valid():
    train = _build_train()
    groups = build_groups(train, MEMBERS)

    assert len(groups) > 0
    for g in groups.itertuples():
        # no member appears twice in a group
        assert len(set(g.members)) == len(g.members) == g.size
        assert g.size in (3, 4)
        assert set(g.members) <= MEMBERS - {"u8"}
        # every member liked the target within [window_start, +30 days]
        end = g.window_start + pd.Timedelta(days=30)
        for user in g.members:
            dates = _liked_dates(train, user, g.target_business_id)
            assert any(g.window_start <= d <= end for d in dates)


def test_member_sets_are_distinct():
    groups = build_groups(_build_train(), MEMBERS)
    member_sets = [frozenset(m) for m in groups["members"]]
    assert len(member_sets) == len(set(member_sets))
    assert groups["group_id"].is_unique


def test_window_boundary():
    groups = build_groups(_build_train(), MEMBERS)
    targets = set(groups["target_business_id"])
    assert "b3" not in targets
    # b2's only valid group can lose its member set to a b1 duplicate;
    # check it in isolation.
    b2 = build_groups(_build_train().query("business_id == 'b2'"), MEMBERS)
    assert list(b2["members"]) == [["u1", "u2", "u3"]]


def test_caps():
    train = _build_train()
    assert len(build_groups(train, MEMBERS, max_per_target=2).query("target_business_id == 'b1'")) == 2
    assert len(build_groups(train, MEMBERS, max_groups=3)) == 3


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
