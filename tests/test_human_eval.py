import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import human_eval as he

GROUP_IDS = [f"g{i:02d}" for i in range(12)]


# ---------- randomization ----------


def test_trial_plan_covers_every_group_once_and_is_deterministic_per_code():
    plan = he.trial_plan("P01", GROUP_IDS)
    assert sorted(t["group_id"] for t in plan) == GROUP_IDS
    assert plan == he.trial_plan("P01", GROUP_IDS)  # resumable
    assert [t["group_id"] for t in plan] != [t["group_id"] for t in he.trial_plan("P02", GROUP_IDS)]


def test_both_strategies_appear_on_the_left():
    lefts = Counter(t["left_strategy"] for code in range(200) for t in he.trial_plan(f"P{code}", GROUP_IDS))
    assert set(lefts) == set(he.STRATEGIES)
    share = lefts["least_misery"] / sum(lefts.values())
    assert 0.45 < share < 0.55  # 2,400 trials: a fair coin per trial
    # within one participant, both sides appear too (12 trials; P01 is fixed by the seed)
    assert {t["left_strategy"] for t in he.trial_plan("P01", GROUP_IDS)} == set(he.STRATEGIES)


def test_group_order_is_shuffled_across_participants():
    firsts = Counter(he.trial_plan(f"P{code}", GROUP_IDS)[0]["group_id"] for code in range(300))
    assert len(firsts) == len(GROUP_IDS)  # every group is shown first to someone


@pytest.mark.parametrize(
    "choice, left, expected",
    [
        ("List 1", "least_misery", "least_misery"),
        ("List 2", "least_misery", "average"),
        ("List 1", "average", "average"),
        ("List 2", "average", "least_misery"),
        ("No preference", "average", he.NO_PREFERENCE),
        ("No preference", "least_misery", he.NO_PREFERENCE),
    ],
)
def test_choice_maps_to_the_strategy_behind_the_list(choice, left, expected):
    assert he.chose_strategy(choice, left) == expected


def test_unknown_choice_is_rejected():
    with pytest.raises(ValueError, match="choice"):
        he.chose_strategy("List 3", "average")


@pytest.mark.parametrize("code, ok", [("P01", True), ("a-b_9", True), ("x", False), ("Jane Doe", False),
                                      ("", False), ("p" * 33, False)])
def test_valid_code(code, ok):
    assert he.valid_code(code) is ok


def test_responses_round_trip_with_the_recorded_mapping(tmp_path):
    path = tmp_path / "human_eval" / "responses.csv"
    he.append_response(he.response_row("P01", "g03", "low", "average", "List 2", " too pricey ", "t1"), path)
    he.append_response(he.response_row("P01", "g07", "high", "least_misery", "No preference", "", "t2"), path)
    rows = he.load_responses(path)
    assert list(rows.columns) == list(he.RESPONSE_COLUMNS)
    assert rows.to_dict("records") == [
        {"participant_code": "P01", "group_id": "g03", "cohesion_bucket": "low", "left_strategy": "average",
         "choice": "List 2", "chose_strategy": "least_misery", "reason": "too pricey", "timestamp": "t1"},
        {"participant_code": "P01", "group_id": "g07", "cohesion_bucket": "high", "left_strategy": "least_misery",
         "choice": "No preference", "chose_strategy": "none", "reason": "", "timestamp": "t2"},
    ]
    assert he.answered_groups("P01", path) == {"g03", "g07"} and he.answered_groups("P02", path) == set()


# ---------- group selection ----------


def _groups_and_taste(n_groups=40):
    """Groups of 3 whose cohesion rises with the group index; one group has an unknown member."""
    rng = np.random.default_rng(0)
    taste, rows = {}, []
    base = np.eye(8)[0]
    for g in range(n_groups):
        members = [f"u{g}_{m}" for m in range(3)]
        spread = 1.0 - g / n_groups
        for m in members:
            v = base + spread * rng.normal(size=8)
            taste[m] = v / np.linalg.norm(v)
        rows.append({"group_id": f"g{g:02d}", "members": np.array(members), "split": "test" if g % 4 else "val"})
    rows.append({"group_id": "gx", "members": np.array(["nobody", "u1_0", "u1_1"]), "split": "test"})
    return pd.DataFrame(rows), taste


def test_select_groups_takes_six_test_groups_per_cohesion_half():
    groups, taste = _groups_and_taste()
    selected = he.select_groups(groups, taste)
    assert list(selected["cohesion_bucket"]) == ["high"] * 6 + ["low"] * 6
    test_ids = set(groups.loc[groups["split"] == "test", "group_id"]) - {"gx"}
    assert set(selected["group_id"]) <= test_ids and selected["group_id"].is_unique
    high = selected[selected["cohesion_bucket"] == "high"]["cohesion"]
    low = selected[selected["cohesion_bucket"] == "low"]["cohesion"]
    assert high.min() > low.max()
    assert selected.equals(he.select_groups(groups, taste))  # seeded
    assert selected.loc[0, "members"].count(";") == 2


def test_select_groups_refuses_when_a_bucket_is_too_small():
    groups, taste = _groups_and_taste(n_groups=12)
    with pytest.raises(ValueError, match="cohesion test groups"):
        he.select_groups(groups, taste)


# ---------- analysis on a fake CSV ----------


def _fake_responses(path: Path) -> Path:
    """3 participants, 12 judgments each.
    high: 12 least_misery, 4 average, 2 none.  low: 6 least_misery, 10 average, 2 none."""
    rows = []
    plan = {"high": ["least_misery"] * 12 + ["average"] * 4 + ["none"] * 2,
            "low": ["least_misery"] * 6 + ["average"] * 10 + ["none"] * 2}
    for bucket, chosen in plan.items():
        for i, strategy in enumerate(chosen):
            left = he.STRATEGIES[i % 2]
            choice = ("No preference" if strategy == "none"
                      else "List 1" if strategy == left else "List 2")
            rows.append(he.response_row(f"P{i % 3}", f"{bucket}{i // 3}", bucket, left, choice, "", "t"))
    for row in rows:
        he.append_response(row, path)
    return path


def test_analysis_on_a_fake_csv(tmp_path):
    result = he.analyze(he.load_responses(_fake_responses(tmp_path / "responses.csv")))
    assert result["n_participants"] == 3 and result["n_judgments"] == 36
    assert result["no_preference_rate"] == pytest.approx(4 / 36)

    overall = result["overall"]
    assert (overall["k"], overall["n"]) == (18, 32) and overall["rate"] == pytest.approx(18 / 32)
    lo, hi = overall["ci"]
    assert (lo, hi) == pytest.approx((0.3933, 0.7183), abs=1e-4)  # Wilson; matches scipy proportion_ci
    assert overall["p"] == pytest.approx(0.5966, abs=1e-4)  # exact two-sided binomial

    high, low = result["by_bucket"]["high"], result["by_bucket"]["low"]
    assert (high["k"], high["n"], low["k"], low["n"]) == (12, 16, 6, 16)
    assert high["ci"] == pytest.approx((0.5050, 0.8982), abs=1e-4) and high["p"] == pytest.approx(0.0768, abs=1e-4)
    assert low["ci"] == pytest.approx((0.1848, 0.6136), abs=1e-4) and low["p"] == pytest.approx(0.4545, abs=1e-4)


def test_wilson_matches_known_values_and_handles_edges():
    from scipy.stats import binomtest

    for k, n in [(18, 32), (1, 9), (7, 7)]:
        ci = binomtest(k, n).proportion_ci(method="wilson")
        assert he.wilson(k, n) == pytest.approx((ci.low, ci.high), abs=1e-6)
    assert he.wilson(0, 10)[0] == pytest.approx(0.0) and he.wilson(10, 10)[1] == pytest.approx(1.0)
    assert all(math.isnan(x) for x in he.wilson(0, 0))


def test_all_no_preference_gives_nan_rates_not_errors(tmp_path):
    path = tmp_path / "r.csv"
    he.append_response(he.response_row("P1", "g1", "high", "average", "No preference", "", "t"), path)
    result = he.analyze(he.load_responses(path))
    assert result["overall"]["n"] == 0 and math.isnan(result["overall"]["rate"])
    assert result["no_preference_rate"] == 1.0


def test_log_appends_rows_under_a_convenience_sample_header(tmp_path, monkeypatch):
    responses = _fake_responses(tmp_path / "responses.csv")
    monkeypatch.setattr(he, "RESULTS_PATH", tmp_path / "results" / "human_eval.md")
    he.main(["--responses", str(responses), "--log", "pilot"])
    he.main(["--responses", str(responses), "--log", "again"])
    text = he.RESULTS_PATH.read_text(encoding="utf-8")
    assert text.count("small convenience sample") == 1 and text.count("# Human evaluation") == 1
    rows = [line for line in text.splitlines() if line.startswith("| 20")]
    assert len(rows) == 2 and rows[0].endswith("| pilot |")
    assert "| 3 | 36 | 0.111 | 0.562 [0.393, 0.718] | 0.597 |" in rows[0]


# ---------- the page ----------

PAGE = str(Path(__file__).resolve().parents[1] / "src" / "pages" / "3_Blind_Eval.py")


def _restaurants(prefix):
    return [{"business_id": f"{prefix}{i}", "name": f"{prefix.upper()} place {i}", "categories": "Thai",
             "price_level": 2, "score": 0.9, "member_scores": {}} for i in range(3)]


@pytest.fixture
def page(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from src import ui_common

    monkeypatch.setenv("DATA_SOURCE", "files")
    monkeypatch.setattr(he, "GROUPS_CSV", tmp_path / "groups.csv")
    monkeypatch.setattr(he, "RESPONSES_CSV", tmp_path / "responses.csv")
    pd.DataFrame({"group_id": ["g1", "g2"], "cohesion_bucket": ["high", "low"], "cohesion": [0.9, 0.1],
                  "n_members": [2, 2], "members": ["a;b", "c;d"]}).to_csv(he.GROUPS_CSV, index=False)

    def post(path, json=None):
        assert path == "/group/compare" and json["norm"] == "rank"
        assert sorted(json["strategies"]) == sorted(he.STRATEGIES)
        # strategies come back in request order; lm = least_misery, av = average
        return {"results": [{"strategy": s, "restaurants": _restaurants(s[:2] if s == "average" else "lm")}
                            for s in json["strategies"]]}

    monkeypatch.setattr(ui_common, "post", post)
    monkeypatch.setattr(ui_common, "get", lambda path, params=None: {
        "users": [{"user_id": u, "profile": [{"business_id": "x", "name": f"fav of {u}", "stars": 5.0}]}
                  for u in params["user_ids"]]})
    return AppTest.from_file(PAGE, default_timeout=30)


def _start(at, code="P01"):
    at.run()
    at.text_input[0].input(code)
    at.button[0].click()
    return at.run()


def _shown_lists(at):
    text = " ".join(m.value for m in at.markdown)
    return text.index("LM place 0") < text.index("AV place 0")  # True: least_misery is List 1


def test_page_records_the_hidden_mapping_and_never_shows_strategy_names(page):
    at = _start(page)
    plan = he.trial_plan("P01", ["g1", "g2"])
    first = plan[0]
    text = " ".join(m.value for m in at.markdown)
    assert "least_misery" not in text and "average" not in text
    assert "List 1" in text and "List 2" in text
    assert _shown_lists(at) == (first["left_strategy"] == "least_misery")

    at.text_area[0].input("closer to everyone")
    next(b for b in at.button if b.label == "List 1").click()
    at = at.run()
    rows = he.load_responses()
    assert len(rows) == 1
    row = rows.iloc[0]
    assert (row["participant_code"], row["group_id"]) == ("P01", first["group_id"])
    assert row["left_strategy"] == first["left_strategy"] and row["chose_strategy"] == first["left_strategy"]
    assert row["reason"] == "closer to everyone"

    next(b for b in at.button if b.label == "List 2").click()
    at = at.run()
    row = he.load_responses().iloc[1]
    assert row["group_id"] == plan[1]["group_id"]
    assert row["chose_strategy"] == he.right_strategy(plan[1]["left_strategy"])
    assert any("All done" in s.value for s in at.success)


def test_page_rejects_names_as_codes(page):
    at = _start(page, code="Jane Doe")
    assert at.error and "participant_code" not in at.session_state


def test_page_refuses_to_run_on_the_hosted_demo(page, monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "db")
    at = page.run()
    assert "locally only" in at.error[0].value
    assert not at.text_input
