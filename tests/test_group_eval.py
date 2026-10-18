import numpy as np
import pandas as pd
import pytest

import src.recommend as recommend
from src.group_eval import (
    ALL_STRATEGIES,
    GROUPS_RESULTS,
    PreWindowPopularity,
    _list_metrics,
    add_buckets,
    bootstrap_vs,
    build_parser,
    cluster_bootstrap_ci,
    evaluate_groups,
    leak_by_popularity,
    lift_cost_table,
    list_overlap,
    log_results,
    modeled_diffs_by_cohesion,
    paired_tests,
    selected_strategies,
    strategy_config,
    summarize,
    target_leak_flags,
    tercile_table,
)
from src.recommend import ContentBased, ContentBasedCentered


# --- embed.py: --limit isolation and guard-before-save ---------------------
# (Lives here because tests/test_embed.py isn't assigned to Person A in week 2.)


@pytest.fixture
def fake_embed(tmp_path, monkeypatch):
    import src.embed as embed

    ids = [f"b{i}" for i in range(4)]
    reviews = pd.DataFrame(
        [
            {"review_id": f"{b}-{s}", "business_id": b, "stars": s, "text": "x" * 300}
            for b in ids
            for s in (5, 2)
        ]
    )
    restaurants = pd.DataFrame({"name": ids, "categories": "Food"}, index=pd.Index(ids, name="business_id"))

    class FakeModel:
        def __init__(self, name):
            pass

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3), dtype="float32")

    monkeypatch.setattr(embed, "_load_eligible_reviews", lambda: (reviews, restaurants, list(ids)))
    monkeypatch.setattr(embed, "SentenceTransformer", FakeModel)
    for name in ("EMBEDDINGS", "EMBEDDINGS_BLOB", "BUSINESS_IDS"):
        suffix = ".txt" if name == "BUSINESS_IDS" else ".npy"
        monkeypatch.setattr(embed, name, tmp_path / (name.lower() + suffix))
    return embed, ids


@pytest.mark.parametrize("mode", ["chunked", "blob"])
def test_limit_run_never_touches_real_outputs(fake_embed, mode):
    embed, ids = fake_embed
    real = [embed.EMBEDDINGS, embed.EMBEDDINGS_BLOB, embed.BUSINESS_IDS]
    run = embed.run_chunked if mode == "chunked" else embed.run_blob

    run(limit=2)

    assert not any(p.exists() for p in real)
    emb_path, ids_path = embed._output_paths(
        embed.EMBEDDINGS if mode == "chunked" else embed.EMBEDDINGS_BLOB, 2
    )
    assert emb_path.name.endswith("_limit2.npy") and ids_path.name.endswith("_limit2.txt")
    assert np.load(emb_path).shape == (2, 3)
    assert ids_path.read_text().split() == ids[:2]


def test_guard_fails_before_any_npy_is_saved(fake_embed):
    embed, ids = fake_embed
    embed.BUSINESS_IDS.write_text("\n".join(reversed(ids)) + "\n")

    with pytest.raises(ValueError, match="ordering would change"):
        embed.run_chunked()

    assert not embed.EMBEDDINGS.exists()
    assert embed.BUSINESS_IDS.read_text().split() == list(reversed(ids))


def test_full_run_writes_real_outputs(fake_embed):
    embed, ids = fake_embed
    embed.run_chunked()
    assert np.load(embed.EMBEDDINGS).shape == (4, 3)
    assert embed.BUSINESS_IDS.read_text().split() == ids


# --- ContentBased.taste_vector_from / scores_for --------------------------
# (Lives here because tests/test_recommend.py isn't assigned to Person A in week 2.)

TINY_IDS = ["r0", "r1", "r2", "r3"]
TINY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.6, 0.8, 0.0],
        [0.0, 0.6, 0.8],
    ],
    dtype="float32",
)


@pytest.fixture
def tiny_embeddings(monkeypatch):
    monkeypatch.setattr(recommend, "load_embeddings", lambda path: (TINY.copy(), list(TINY_IDS)))


def test_taste_vector_skips_unknown_ids(tiny_embeddings):
    model = ContentBased()
    np.testing.assert_allclose(
        model.taste_vector_from(["r0", "nope", "r1"]), model.taste_vector_from(["r0", "r1"])
    )


def test_taste_vector_none_when_nothing_known(tiny_embeddings):
    model = ContentBased()
    assert model.taste_vector_from([]) is None
    assert model.taste_vector_from(["nope", "also-nope"]) is None


def test_taste_vector_is_unit_norm_mean(tiny_embeddings):
    model = ContentBased()
    taste = model.taste_vector_from(["r0", "r1"])
    np.testing.assert_allclose(np.linalg.norm(taste), 1.0, rtol=1e-6)
    np.testing.assert_allclose(taste, np.array([1, 1, 0]) / np.sqrt(2), rtol=1e-6)


def test_centered_model_uses_centered_matrix(tiny_embeddings):
    model = ContentBasedCentered()
    centered = TINY - TINY.mean(axis=0)
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    np.testing.assert_allclose(model.matrix, centered, rtol=1e-6)

    expected = centered[[0, 2]].mean(axis=0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(model.taste_vector_from(["r0", "r2"]), expected, rtol=1e-5)
    # and differs from the uncentered model's vector
    assert not np.allclose(model.taste_vector_from(["r0", "r2"]), ContentBased().taste_vector_from(["r0", "r2"]))


def test_scores_for_follows_business_ids_order(tiny_embeddings):
    for model in (ContentBased(), ContentBasedCentered()):
        taste = model.taste_vector_from(["r3"])
        scores = model.scores_for(taste)
        assert scores.shape == (len(TINY_IDS),)
        for i, b in enumerate(model.business_ids):
            assert b == TINY_IDS[i]
            np.testing.assert_allclose(scores[i], model.matrix[model.index[b]] @ taste, rtol=1e-6)
        assert int(np.argmax(scores)) == TINY_IDS.index("r3")


T0 = pd.Timestamp("2020-01-01")
WINDOW_START = T0 + pd.Timedelta(days=100)
BUSINESSES = ["bt", "b0", "b1", "b2", "b3", "b4", "b5", "bpop_before", "bpop_after"]


def _day(d):
    return T0 + pd.Timedelta(days=d)


def _review(user_id, business_id, day, stars=5):
    return {"user_id": user_id, "business_id": business_id, "stars": stars, "date": _day(day)}


class RecordingModel(ContentBased):
    """ContentBased over a tiny in-memory matrix; records every history
    passed to taste_vector_from."""

    def __init__(self):
        rng = np.random.default_rng(0)
        m = rng.normal(size=(len(BUSINESSES), 8))
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.business_ids = list(BUSINESSES)
        self.index = {b: i for i, b in enumerate(self.business_ids)}
        self.name = "recording"
        self.profiles = []

    def taste_vector_from(self, business_ids):
        self.profiles.append(list(business_ids))
        return super().taste_vector_from(business_ids)


def _build_train():
    rows = []
    for u in ["u1", "u2", "u3"]:
        # three pre-window likes each (enough history)
        rows += [_review(u, "b0", 10), _review(u, "b1", 20), _review(u, "b2", 30)]
        # liked the target inside the window (how groups.py found them)
        rows.append(_review(u, "bt", 105))
    # u1: a 5-star review exactly at window_start and one after it -> neither
    # may enter u1's profile or be excluded from candidates.
    rows.append(_review("u1", "b4", 100))
    rows.append(_review("u1", "b5", 150))
    # u1 also disliked b3 before the window -> not in profile, but excluded.
    rows.append(_review("u1", "b3", 40, stars=2))

    # u4: only two pre-window likes -> any group with u4 is dropped.
    rows += [_review("u4", "b0", 10), _review("u4", "b1", 20), _review("u4", "bt", 101)]
    # u5: enough likes, but rated the target 2 stars before the window.
    rows += [_review("u5", b, d) for b, d in [("b0", 10), ("b1", 20), ("b2", 30)]]
    rows += [_review("u5", "bt", 50, stars=2), _review("u5", "bt", 102)]

    # Non-members: bpop_before has 2 pre-window reviews, bpop_after has 5
    # post-window ones; the target has 1 before and 4 after.
    rows += [_review(f"x{i}", "bpop_before", 60 + i) for i in range(2)]
    rows += [_review(f"y{i}", "bpop_after", 120 + i) for i in range(5)]
    rows += [_review("z0", "bt", 70)] + [_review(f"z{i}", "bt", 110 + i) for i in range(1, 5)]
    return pd.DataFrame(rows)


def _group(group_id, members):
    return {
        "group_id": group_id,
        "members": members,
        "target_business_id": "bt",
        "window_start": WINDOW_START,
        "size": len(members),
        "split": "val",
    }


def _run(groups, train=None, model=None):
    model = model or RecordingModel()
    train = _build_train() if train is None else train
    results, drops = evaluate_groups(pd.DataFrame(groups), train, model)
    return results, drops, model


def test_only_strictly_pre_window_likes_enter_profiles():
    results, drops, model = _run([_group("g_ok", ["u1", "u2", "u3"])])
    assert results["group_id"].unique().tolist() == ["g_ok"]
    assert len(model.profiles) == 3
    for profile in model.profiles:
        assert sorted(profile) == ["b0", "b1", "b2"]  # no b4 (at T), b5 (after), b3 (2 stars), bt


def test_target_excluded_from_profile_but_kept_as_candidate():
    results, _, model = _run([_group("g_ok", ["u1", "u2", "u3"])])
    assert all("bt" not in p for p in model.profiles)
    # Excluded: only the pre-window reviewed b0, b1, b2, b3. bt, b4 and b5 stay.
    assert (results["n_candidates"] == len(BUSINESSES) - 4).all()

    # If the target's embedding equals the members' shared taste, the
    # content strategies must rank it first -- so it really is a candidate.
    model = RecordingModel()
    taste = model.matrix[[model.index[b] for b in ["b0", "b1", "b2"]]].mean(axis=0)
    model.matrix[model.index["bt"]] = taste / np.linalg.norm(taste)
    results, _, _ = _run([_group("g_ok", ["u1", "u2", "u3"])], model=model)
    content = results[
        results["strategy"].isin(["average", "least_misery", "fairness", "fairness_repeated"])
    ]
    assert (content["hit"] == 1).all()
    assert (content["ndcg"] == 1).all()


def test_groups_failing_filters_are_dropped():
    results, drops, _ = _run(
        [
            _group("g_ok", ["u1", "u2", "u3"]),
            _group("g_short", ["u1", "u2", "u4"]),
            _group("g_seen", ["u1", "u2", "u5"]),
        ]
    )
    assert set(results["group_id"]) == {"g_ok"}
    assert drops["too_few_prior_likes"] == 1
    assert drops["target_in_prior_history"] == 1


def test_popularity_uses_only_pre_window_counts():
    train = _build_train()
    pop = PreWindowPopularity(train, BUSINESSES)
    counts = dict(zip(BUSINESSES, pop.counts(WINDOW_START)))
    assert counts["bpop_after"] == 0
    assert counts["bpop_before"] == 2
    # bt: z0 (day 70) and u5's 2-star review (day 50); in-window ones excluded.
    assert counts["bt"] == 2

    results, _, _ = _run([_group("g_ok", ["u1", "u2", "u3"])], train=train)
    assert (results["target_prior_reviews"] == 2).all()


def test_popularity_strategy_ranks_by_pre_window_counts():
    # Pre-window, bt and bpop_before tie at 2 and bt comes first in catalog
    # order; bpop_after (5 reviews, all post-window) must not outrank it.
    results, _, _ = _run([_group("g_ok", ["u1", "u2", "u3"])])
    pop = results[results["strategy"] == "popularity"].iloc[0]
    assert pop["hit"] == 1 and pop["ndcg"] == 1


def test_member_percentile_metrics_are_consistent():
    results, _, _ = _run([_group("g_ok", ["u1", "u2", "u3"])])
    for r in results.itertuples():
        assert 0 <= r.worst_member <= r.group_score <= 1
        assert r.fairness_gap >= 0
    assert set(results["strategy"]) == {
        "random", "popularity", "average", "least_misery", "fairness", "fairness_repeated"
    }


@pytest.mark.parametrize("bad", ["u4", "u5"])
def test_no_survivors_returns_empty(bad):
    results, drops, _ = _run([_group("g", ["u1", "u2", bad])])
    assert results.empty
    assert sum(drops.values()) >= 1


# --- paired statistics, overlap, buckets, leak flags ----------------------


def _fake_results(hits, lists=None, targets=None):
    """hits: {strategy: [0/1 per group]}; targets: target per group
    (default: every group its own target)."""
    rows = []
    for s, hs in hits.items():
        for g, h in enumerate(hs):
            rows.append(
                {
                    "group_id": f"g{g}",
                    "target_business_id": targets[g] if targets else f"t{g}",
                    "strategy": s,
                    "hit": float(h),
                    "ndcg": float(h),
                    "top_k": (lists or {}).get(s, [["x"]] * len(hs))[g],
                }
            )
    return pd.DataFrame(rows)


def test_paired_tests_counts_discordant_pairs():
    res = _fake_results({"average": [1, 1, 1, 0, 0], "least_misery": [1, 0, 0, 0, 1]})
    row = paired_tests(res).iloc[0]
    assert (row["a"], row["b"]) == ("average", "least_misery")
    assert (row["a_hit_b_miss"], row["a_miss_b_hit"]) == (2, 1)
    assert row["p_mcnemar_indep"] == pytest.approx(1.0)


def test_paired_tests_no_discordance_gives_p_one():
    res = _fake_results({"average": [1, 0, 0], "fairness": [1, 0, 0]})
    assert paired_tests(res).iloc[0]["p_mcnemar_indep"] == 1.0


def test_bootstrap_vs_average_is_paired_and_seeded():
    res = _fake_results({"average": [1] * 30 + [0] * 70, "random": [0] * 100})
    a, b = bootstrap_vs(res), bootstrap_vs(res)
    pd.testing.assert_frame_equal(a, b)
    row = a.loc["random"]
    assert row["hit_diff_vs_average"] == pytest.approx(-0.3)
    assert row["ci_hi"] < 0 and row["ci_excludes_0"]


def test_cluster_bootstrap_keeps_targets_together():
    # One target -> every resample is the whole data -> zero-width CI.
    lo, hi = cluster_bootstrap_ci(np.array([1, 0, 0, 1.0]), np.array(["t"] * 4))
    assert lo == hi == pytest.approx(0.5)

    # Two targets, each internally homogeneous: resample means can only be
    # 0, 0.5 or 1 (whole clusters), never e.g. 0.25 as with group resampling.
    values = np.array([1.0] * 10 + [0.0] * 10)
    clusters = np.array(["a"] * 10 + ["b"] * 10)
    lo, hi = cluster_bootstrap_ci(values, clusters)
    assert lo in (0.0, 0.5, 1.0) and hi in (0.0, 0.5, 1.0)
    assert (lo, hi) == (0.0, 1.0)


def test_cluster_ci_wider_than_iid_when_targets_correlate():
    # 20 targets x 10 groups, hits perfectly correlated within target.
    targets = [f"t{i // 10}" for i in range(200)]
    hits = [1 if i // 10 < 4 else 0 for i in range(200)]
    clustered = bootstrap_vs(
        _fake_results({"average": [0] * 200, "popularity": hits}, targets=targets)
    ).loc["popularity"]
    iid = bootstrap_vs(_fake_results({"average": [0] * 200, "popularity": hits})).loc["popularity"]
    assert clustered["ci_hi"] - clustered["ci_lo"] > 2 * (iid["ci_hi"] - iid["ci_lo"])


def test_list_overlap_jaccard():
    res = _fake_results(
        {"average": [0, 0], "fairness": [0, 0]},
        lists={"average": [["a", "b"], ["a", "b"]], "fairness": [["a", "b"], ["c", "d"]]},
    )
    overlap = list_overlap(res)
    assert overlap.loc["average", "fairness"] == pytest.approx(0.5)
    assert overlap.loc["average", "average"] == 1.0


def test_bucket_tables_print_low_mid_high():
    rows = []
    for g in range(9):
        for s in ["random", "average"]:
            rows.append({"group_id": f"g{g}", "strategy": s, "cohesion": g / 10,
                         "target_prior_reviews": 100 - g, "hit": 0.0, "ndcg": 0.0,
                         "group_score": 0.5, "worst_member": 0.5, "fairness_gap": 0.0})
    table = summarize(add_buckets(pd.DataFrame(rows)), ["popularity_bucket", "strategy"])
    assert list(table.index) == [(b, s) for b in ["low", "mid", "high"] for s in ["random", "average"]]
    assert (table["n"] == 3).all()


def test_target_leak_flags():
    groups = pd.DataFrame(
        [
            {"group_id": "leaky", "members": ["u1", "u2"], "target_business_id": "t"},
            {"group_id": "clean", "members": ["u3", "u4"], "target_business_id": "t"},
            {"group_id": "no_reviews", "members": ["u5"], "target_business_id": "s"},
        ]
    )
    train = pd.DataFrame(
        [
            {"review_id": "r1", "user_id": "u1", "business_id": "t"},
            {"review_id": "r2", "user_id": "u2", "business_id": "t"},
            {"review_id": "r3", "user_id": "u3", "business_id": "t"},
            {"review_id": "r9", "user_id": "u4", "business_id": "other"},
        ]
    )
    embedded = {"t": {"r2", "r_someone_else"}, "other": {"r9"}}
    flags = target_leak_flags(groups, train, embedded)
    assert flags.to_dict() == {"leaky": True, "clean": False, "no_reviews": False}


def _bucketed_results():
    rows = []
    for g in range(12):
        for s, hit, score in [("average", g % 2, 0.9), ("least_misery", 0, 0.8), ("fairness", 0, 0.85)]:
            rows.append({"group_id": f"g{g}", "target_business_id": f"t{g % 4}", "strategy": s,
                         "hit": float(hit), "ndcg": float(hit), "group_score": score,
                         "worst_member": score - 0.1, "fairness_gap": 0.1,
                         "popularity_bucket": "low" if g < 6 else "mid",
                         "cohesion_bucket": "low" if g % 3 else "high",
                         "target_leak": g % 4 == 0})
    return pd.DataFrame(rows)


def test_leak_by_popularity_splits_bucket_and_flag():
    table = leak_by_popularity(_bucketed_results())
    assert set(table.index.get_level_values("popularity_bucket")) == {"low", "mid"}
    low_flagged = table.loc[("low", "average", True)]
    # low bucket = g0..g5; flagged (g % 4 == 0) = g0, g4, both even -> no average hits
    assert low_flagged["n"] == 2 and low_flagged["hit@10"] == 0.0
    assert table.loc[("low", "average", False)]["n"] == 4


def test_modeled_diffs_by_cohesion():
    table = modeled_diffs_by_cohesion(_bucketed_results())
    row = table.loc[("high", "group_score", "least_misery - average")]
    assert row["value"] == pytest.approx(-0.1)
    assert row["ci_excludes_0"]
    assert table.loc[("low", "fairness_gap", "fairness - average")]["value"] == pytest.approx(0.0)


def test_repeated_diagnostics():
    from src.aggregate import fairness_repeated, normalize
    from src.group_eval import _repeated_diagnostics

    # Two like-minded members and one opposite: after dinner 1 the opposite
    # member is lowest and, at tau=0.1, effectively dictates dinner 2.
    rng = np.random.default_rng(6)
    base = rng.normal(size=50)
    scores = np.stack([base, base + 0.01 * rng.normal(size=50), -base])
    excluded = np.zeros(50, dtype=bool)
    picks, history, weights = fairness_repeated(scores, excluded, 5, return_weights=True)
    d = _repeated_diagnostics(picks, history, weights, normalize(scores, excluded), excluded)
    assert len(d["fr_max_weight"]) == 5
    assert d["fr_max_weight"][0] == pytest.approx(1 / 3)
    assert d["fr_top_is_lowest"][0] is None  # all totals tied at dinner 1
    untied = [x for x in d["fr_top_is_lowest"][1:] if x is not None]
    assert untied and all(untied)  # softmax(-cum/tau): true by construction
    assert d["fr_max_weight"][1] > 0.9 and d["fr_dictator"][1]


# --- CLI: strategy selection, --log, secondary metrics, regression ----------
RICH_T0 = pd.Timestamp("2020-01-01")
RICH_WINDOW = RICH_T0 + pd.Timedelta(days=100)
RICH_IDS = ["t"] + [f"r{i:02d}" for i in range(39)]


class RichModel(ContentBased):
    def __init__(self):
        m = np.random.default_rng(7).normal(size=(len(RICH_IDS), 8))
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.business_ids = list(RICH_IDS)
        self.index = {b: i for i, b in enumerate(self.business_ids)}
        self.name = "rich"


def _rich_review(user, business, day, stars=5):
    return {"user_id": user, "business_id": business, "stars": stars,
            "date": RICH_T0 + pd.Timedelta(days=day)}


def rich_train():
    likes = {
        "u1": ["r00", "r01", "r02", "r03"],
        "u2": ["r04", "r05", "r06", "r07"],
        "u3": ["r00", "r08", "r09", "r10"],
        "u4": ["r11", "r12", "r13", "r14"],
    }
    rows = []
    for user, businesses in likes.items():
        rows += [_rich_review(user, b, 10 + 7 * j) for j, b in enumerate(businesses)]
        rows.append(_rich_review(user, "t", 105))  # in-window like of the target
        rows.append(_rich_review(user, "r30", 150))  # post-window: never used
    rows.append(_rich_review("u2", "r15", 50, stars=2))  # pre-window dislike: excluded only
    # non-members, for popularity: r20 popular before the window, r21 only after
    rows += [_rich_review(f"x{i}", "r20", 60 + i) for i in range(3)]
    rows += [_rich_review(f"y{i}", "r21", 120 + i) for i in range(6)]
    return pd.DataFrame(rows)


def rich_groups():
    return pd.DataFrame(
        [
            {"group_id": "gA", "members": ["u1", "u2", "u3"], "target_business_id": "t",
             "window_start": RICH_WINDOW, "size": 3, "split": "val"},
            {"group_id": "gB", "members": ["u2", "u3", "u4"], "target_business_id": "t",
             "window_start": RICH_WINDOW, "size": 3, "split": "val"},
        ]
    )


# Primary metrics produced by evaluate_groups BEFORE the CLI/secondary-metric
# changes, on the fixture above: (hit, ndcg, group_score, worst_member, fairness_gap).
PRIMARY_SNAPSHOT = {
    ("gA", "random"): (1.0, 0.38685280723454163, 0.4975308641975309, 0.46296296296296297, 0.06296296296296294),
    ("gA", "popularity"): (1.0, 0.3562071871080222, 0.6, 0.5555555555555556, 0.12962962962962965),
    ("gA", "average"): (1.0, 0.43067655807339306, 0.6617283950617283, 0.5814814814814815, 0.2148148148148148),
    ("gA", "least_misery"): (1.0, 0.43067655807339306, 0.6333333333333333, 0.5703703703703704, 0.12962962962962954),
    ("gA", "fairness"): (1.0, 0.5, 0.6481481481481481, 0.5592592592592591, 0.1888888888888891),
    ("gA", "fairness_repeated"): (1.0, 0.3333333333333333, 0.617283950617284, 0.5925925925925926, 0.04444444444444451),
    ("gB", "random"): (1.0, 0.5, 0.532051282051282, 0.523076923076923, 0.023076923076923217),
    ("gB", "popularity"): (1.0, 0.38685280723454163, 0.5487179487179488, 0.49230769230769234, 0.12692307692307692),
    ("gB", "average"): (1.0, 0.6309297535714575, 0.6884615384615383, 0.5884615384615384, 0.20000000000000018),
    ("gB", "least_misery"): (1.0, 0.5, 0.6692307692307692, 0.5923076923076922, 0.14230769230769236),
    ("gB", "fairness"): (1.0, 0.5, 0.6679487179487179, 0.6384615384615385, 0.04615384615384599),
    ("gB", "fairness_repeated"): (1.0, 0.6309297535714575, 0.6679487179487179, 0.6615384615384615, 0.015384615384615441),
}


def _rich_run(**kwargs):
    results, drops = evaluate_groups(rich_groups(), rich_train(), RichModel(), **kwargs)
    return results


def test_primary_metrics_unchanged_vs_previous_output():
    results = _rich_run()
    assert len(results) == len(PRIMARY_SNAPSHOT)
    for r in results.itertuples():
        expected = PRIMARY_SNAPSHOT[(r.group_id, r.strategy)]
        got = (r.hit, r.ndcg, r.group_score, r.worst_member, r.fairness_gap)
        assert got == pytest.approx(expected, rel=0, abs=1e-12), (r.group_id, r.strategy)


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_strategy_runs_only_that_strategy(strategy):
    one = _rich_run(strategies=(strategy,))
    assert set(one["strategy"]) == {strategy}
    assert len(one) == 2  # both groups
    # identical to that strategy's rows from a full run
    full = _rich_run()
    cols = ["group_id", "hit", "ndcg", "group_score", "worst_member", "fairness_gap", "top_k"]
    pd.testing.assert_frame_equal(
        one[cols].reset_index(drop=True),
        full[full["strategy"] == strategy][cols].reset_index(drop=True),
    )


def test_cli_strategy_selection():
    parser = build_parser()
    assert selected_strategies(parser.parse_args([])) == ALL_STRATEGIES
    assert selected_strategies(parser.parse_args(["--all"])) == ALL_STRATEGIES
    assert selected_strategies(parser.parse_args(["--strategy", "least_misery"])) == ("least_misery",)
    with pytest.raises(SystemExit):
        parser.parse_args(["--all", "--strategy", "average"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--strategy", "borda"])
    args = parser.parse_args(["--strategy", "fairness", "--lam", "0.25", "--norm", "zscore"])
    assert (args.lam, args.normalization) == (0.25, "zscore")


def test_lam_reaches_fairness():
    lam0 = _rich_run(strategies=("fairness", "average"), lam=0.0)
    lists = lam0.pivot(index="group_id", columns="strategy", values="top_k")
    assert all(list(a) == list(b) for a, b in zip(lists["fairness"], lists["average"]))


def test_log_appends_one_row_per_strategy_and_never_touches_results_md(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results.md").write_text("B's file\n")
    results = _rich_run()
    configs = {s: strategy_config(s, "rank", 0.5, None) for s in ALL_STRATEGIES}

    def rows():
        lines = GROUPS_RESULTS.read_text(encoding="utf-8").splitlines()
        return [l for l in lines if l.startswith("| 20")]  # data rows start with the date

    assert log_results(results, "val", "first", configs) == len(ALL_STRATEGIES)
    first = rows()
    assert len(first) == len(ALL_STRATEGIES)
    assert [r.split(" | ")[2] for r in first] == list(ALL_STRATEGIES)
    assert all(r.rstrip().endswith("| first |") for r in first)

    one = results[results["strategy"] == "least_misery"]
    assert log_results(one, "val", "second", {"least_misery": configs["least_misery"]}) == 1
    second = rows()
    assert second[: len(first)] == first
    assert len(second) == len(first) + 1 and "| least_misery |" in second[-1]

    text = GROUPS_RESULTS.read_text(encoding="utf-8")
    assert text.count("| date | split | strategy |") == 1  # header written once
    assert "PREDICTED" in text
    assert (tmp_path / "results.md").read_text() == "B's file\n"
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        "results.md",
        "results/groups.md",
    ]


def test_secondary_metrics_hand_computed():
    # 2 members x 4 candidates (rank percentiles); the list is items 0, 1, 2.
    pct = np.array(
        [
            [0.0, 0.5, 1.0, 0.25],
            [1.0, 0.5, 0.0, 0.75],
        ]
    )
    m = _list_metrics(np.array([0, 1, 2]), target_idx=1, pct=pct)
    # primary (member means over the list): both members average 0.5
    assert m["group_score"] == pytest.approx(0.5)
    assert m["worst_member"] == pytest.approx(0.5)
    assert m["fairness_gap"] == pytest.approx(0.0)
    # secondary (per restaurant): mins [0, .5, 0] -> 1/6; spreads [1, 0, 1] -> 2/3
    assert m["worst_member_per_item"] == pytest.approx(1 / 6)
    assert m["fairness_gap_per_item"] == pytest.approx(2 / 3)
    assert m["hit"] == 1.0 and m["ndcg"] == pytest.approx(1 / np.log2(3))


def test_lift_cost_and_tercile_tables():
    results = add_buckets(_rich_run())
    lift = lift_cost_table(results)
    assert "average" not in lift.index and set(lift.index) == set(ALL_STRATEGIES) - {"average"}
    per_group = results.pivot(index="group_id", columns="strategy", values="worst_member")
    assert lift.loc["least_misery", "worst_member diff (PREDICTED)"] == pytest.approx(
        (per_group["least_misery"] - per_group["average"]).mean()
    )
    terciles = tercile_table(results)
    assert set(terciles.index.get_level_values("cohesion_tercile")) <= {"bottom", "top", "all"}
    assert (terciles.loc["all", "n"] == 2).all()
    assert {"worst_member_per_item", "fairness_gap_per_item"} <= set(terciles.columns)


def test_tau_sensitivity_includes_secondary_metrics():
    from src.group_eval import tau_sensitivity

    results = add_buckets(_rich_run(extra_taus=(0.3,)))
    # the fixture has two groups -> one per median-split bucket
    table = tau_sensitivity(results, taus=(0.1, 0.3), n_boot=50)
    metrics = set(table["metric"])
    for metric in ("worst_member_per_item", "fairness_gap_per_item"):
        assert f"{metric} (SECONDARY)" in metrics
        assert f"{metric} - average (SECONDARY)" in metrics
        assert f"{metric} - (tau=0.3) (SECONDARY)" in metrics
    level = table[(table["metric"] == "fairness_gap_per_item (SECONDARY)") & (table["tau"] == 0.1)]
    expected = results[results["strategy"] == "fairness_repeated"].set_index("group_id")
    for row in level.itertuples():
        gids = results[results["cohesion_bucket"] == row.cohesion_bucket]["group_id"].unique()
        assert row.value == pytest.approx(expected.loc[gids, "fairness_gap_per_item"].mean())
    # primary rows still come first and keep their metric labels
    assert list(table["metric"].iloc[:4]) == ["hit@10", "hit@10 - average", "hit@10", "hit@10 - average"]


def test_model_option_defaults_to_centered():
    from src.group_eval import DEFAULT_MODEL, MODELS

    parser = build_parser()
    assert parser.parse_args([]).model == DEFAULT_MODEL == "ContentBasedCentered"
    assert parser.parse_args(["--model", "ContentBasedDebiased"]).model == "ContentBasedDebiased"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "Nope"])
    assert set(MODELS) == {"ContentBasedCentered", "ContentBasedDebiased"}
    # default configs are unchanged; a non-default model is named in the logged config
    assert strategy_config("average", "rank", 0.5, None) == strategy_config("average", "rank", 0.5, None, DEFAULT_MODEL)
    assert "model=" not in strategy_config("average", "rank", 0.5, None)
    assert strategy_config("average", "rank", 0.5, None, "ContentBasedDebiased").startswith("model=ContentBasedDebiased")
