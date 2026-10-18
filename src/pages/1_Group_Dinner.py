"""Group Dinner page (Person A): compare two aggregation strategies for a group.

All data comes from the API through src/ui_common.py; this page never imports
model code. The pure helpers below are tested in tests/test_api_group.py; the
Streamlit UI runs only when Streamlit executes the page (__name__ == "__main__").

Deep link: ?cohesion=low&seed=42&n=4 draws a sample group on load; optional
&strategy=...&strategy_b=... set the strategies and &compare=0 selects single mode.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `src.`

STRATEGIES = ["average", "least_misery", "fairness", "fairness_repeated"]
MIN_MEMBERS, MAX_MEMBERS = 2, 6
README_CALIBRATION = "https://github.com/raga-puli/weat/blob/main/README.md#group-evaluation"


# --- pure helpers (no Streamlit) ------------------------------------------------------


def parse_member_ids(text: str) -> tuple[list[str], str | None]:
    """Comma/whitespace-separated user IDs -> (ids, error message or None)."""
    ids = [t for t in re.split(r"[\s,;]+", text.strip()) if t]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        return ids, f"Each member can appear only once; repeated: {', '.join(duplicates)}"
    if len(ids) < MIN_MEMBERS:
        return ids, f"Enter at least {MIN_MEMBERS} user IDs (got {len(ids)})."
    if len(ids) > MAX_MEMBERS:
        return ids, f"Enter at most {MAX_MEMBERS} user IDs (got {len(ids)})."
    return ids, None


def member_labels(members: list[str]) -> dict[str, str]:
    """Short column labels, M1..Mn, in the order given."""
    return {m: f"M{i}" for i, m in enumerate(members, 1)}


def format_price(level) -> str:
    return "$" * int(level) if level else "–"


def short_categories(categories: str | None, limit: int = 3) -> str:
    """Up to `limit` Yelp categories, skipping the generic ones, with "…" if cut."""
    cats = [c.strip() for c in (categories or "").split(",")
            if c.strip() and c.strip() not in ("Restaurants", "Food")]
    return ", ".join(cats[:limit]) + (", …" if len(cats) > limit else "")


def collapse_chains(restaurants: list[dict]) -> list[dict]:
    """Display only: rows with the same name become one row (the best-ranked
    one), with "(N locations)" and every business_id kept. Rank is the
    original 1-based rank of that best location."""
    out, by_name = [], {}
    for rank, r in enumerate(restaurants, 1):
        name = r.get("name") or r["business_id"]
        if name in by_name:
            row = by_name[name]
            row["locations"] += 1
            row["business_ids"].append(r["business_id"])
            continue
        row = {**r, "rank": rank, "name": name, "locations": 1, "business_ids": [r["business_id"]]}
        by_name[name] = row
        out.append(row)
    for row in out:
        if row["locations"] > 1:
            row["display_name"] = f"{row['name']} ({row['locations']} locations)"
        else:
            row["display_name"] = row["name"]
    return out


def display_rows(result: dict, other_ids: set[str], labels: dict[str, str]) -> list[dict]:
    """Table rows for one strategy's list: rank, name, categories, price, score,
    one column per member (by label), and whether it is also in the other list
    (any of a collapsed row's locations counts)."""
    rows = []
    for r in collapse_chains(result["restaurants"]):
        row = {
            "rank": r["rank"],
            "name": r["display_name"],
            "categories": short_categories(r.get("categories")),
            "price": format_price(r.get("price_level")),
            "score": round(float(r["score"]), 3),
        }
        for member, label in labels.items():
            row[label] = round(float(r["member_scores"][member]), 3)
        row["in_both"] = bool(set(r["business_ids"]) & other_ids)
        rows.append(row)
    return rows


def dinner_rows(result: dict, labels: dict[str, str]) -> list[dict]:
    """fairness_repeated per-dinner view: the pick, then each member's score
    and weight for that dinner."""
    names = {r["business_id"]: r.get("name") or r["business_id"] for r in result["restaurants"]}
    rows = []
    for h in result.get("history", []):
        row = {"dinner": h["dinner"], "pick": names.get(h["business_id"], h["business_id"])}
        for member, label in labels.items():
            row[f"{label} score"] = round(float(h["member_scores"][member]), 3)
            row[f"{label} weight"] = round(float(h["member_weights"][member]), 3)
        rows.append(row)
    return rows


# Val-group cohesion terciles under the API's definition (mean pairwise cosine of
# members' whole-history taste vectors), measured 2026-10-18 over 7,228 val groups:
# 33rd percentile 0.18, median 0.25, 67th percentile 0.32.
COHESION_LOW, COHESION_HIGH = 0.18, 0.32
NORM_HELP = {
    "rank": "each member's percentile among the candidates (0 = their worst, 1 = their best)",
    "zscore": "each member's standardized score (mean 0, sd 1 over the candidates)",
    "raw": "raw similarity scores, not comparable across members",
}


def cohesion_reading(cohesion: float) -> str:
    """One plain-English line for a group's cohesion score."""
    if cohesion < COHESION_LOW:
        return "Very different tastes, so expect compromise picks that no one loves but everyone rates fairly well."
    if cohesion < COHESION_HIGH:
        return "Partly overlapping tastes; strategies will agree on some picks and differ on others."
    return "Similar tastes; most strategies will pick much the same places."


def metric_deltas(summary_a: dict, summary_b: dict) -> dict[str, float]:
    """Strategy B minus strategy A on the three predicted metrics."""
    keys = ("group_utility_predicted", "worst_member_predicted", "fairness_gap_predicted")
    return {k: float(summary_b[k]) - float(summary_a[k]) for k in keys}


def top_member_rows(result: dict, labels: dict[str, str], top: int = 5) -> list[dict]:
    """Per-member scores for the first `top` (chain-collapsed) restaurants;
    restaurant labels are prefixed with their rank so they sort in rank order."""
    rows = []
    for r in collapse_chains(result["restaurants"])[:top]:
        row = {"restaurant": f"{r['rank']}. {r['display_name']}"}
        row.update({label: float(r["member_scores"][m]) for m, label in labels.items()})
        rows.append(row)
    return rows


MAP_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "both": "#9467bd"}


def map_points(results: list[dict]) -> list[dict]:
    """One point per recommended location with coordinates. With two results,
    `list` is "A", "B" or "both"; with one, always "A"."""
    lists = [{r["business_id"]: r for r in res["restaurants"]} for res in results]
    points = []
    for business_id in dict.fromkeys(b for lst in lists for b in lst):
        in_lists = [i for i, lst in enumerate(lists) if business_id in lst]
        r = lists[in_lists[0]][business_id]
        if r.get("lat") is None or r.get("lon") is None:
            continue
        which = "both" if len(in_lists) == 2 else ("A", "B")[in_lists[0]]
        points.append({"lat": float(r["lat"]), "lon": float(r["lon"]), "name": r.get("name"),
                       "list": which, "color": MAP_COLORS[which]})
    return points


def member_score_bounds(results: list[dict], norm: str) -> tuple[float, float]:
    """Shared y-range for the per-member bars: [0, 1] for rank percentiles,
    otherwise the min/max over every shown member score."""
    if norm == "rank":
        return 0.0, 1.0
    values = [float(v) for res in results for r in res["restaurants"] for v in r["member_scores"].values()]
    return (min(values), max(values)) if values else (0.0, 1.0)


def format_api_error(message: str) -> str:
    """ui_common.APIError text -> what to show. For HTTP 422, pull out the
    API's `detail` (our own message, or pydantic's validation errors)."""
    match = re.search(r"HTTP 422: (.*)$", message, re.S)
    if not match:
        return message
    try:
        detail = json.loads(match.group(1)).get("detail")
    except (ValueError, AttributeError):
        return message
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        return "; ".join(
            f"{'.'.join(str(p) for p in d.get('loc', [])[1:]) or 'request'}: {d.get('msg', '')}" for d in detail
        )
    return message


# --- page --------------------------------------------------------------------------


def main() -> None:
    import pandas as pd
    import streamlit as st

    from src.ui_common import APIError, get, post

    st.set_page_config(page_title="Group Dinner", page_icon="🍽️", layout="wide")
    state = st.session_state
    params = st.query_params

    @st.cache_data(ttl=600, show_spinner="Asking the API…")
    def cached_post(path: str, payload_json: str) -> dict:
        return post(path, json=json.loads(payload_json))

    def show_error(e: Exception) -> None:
        st.error(format_api_error(str(e)))

    def load_sample(n: int, cohesion: str, seed: int) -> None:
        sample = get("/group/users/sample", params={"n": n, "cohesion": cohesion, "seed": seed})
        state["members"] = [u["user_id"] for u in sample["users"]]
        state["profiles"] = sample["users"]
        state["sample_note"] = (f"Sampled from val group {sample['group_id']}"
                                if "group_id" in sample else f"{n} random val users")

    # ---- sidebar: group -----------------------------------------------------------
    sb = st.sidebar
    sb.header("Group")
    cohesion_choices = ["any", "high", "low"]
    n = sb.slider("Group size", MIN_MEMBERS, MAX_MEMBERS, int(params.get("n", 4)))
    default_cohesion = params.get("cohesion", "low")
    cohesion_pick = sb.radio(
        "Cohesion", cohesion_choices, horizontal=True,
        index=cohesion_choices.index(default_cohesion) if default_cohesion in cohesion_choices else 2,
        help="high / low: members of a real val group from that half of the cohesion distribution; "
             "any: random val users",
    )
    seed = sb.number_input("Seed", min_value=0, value=int(params.get("seed", 42)), step=1)
    if sb.button("Sample a new group", type="primary"):
        try:
            load_sample(n, cohesion_pick, int(seed))
        except APIError as e:
            show_error(e)
    with sb.expander("Or enter user IDs"):
        typed = st.text_area(f"{MIN_MEMBERS}–{MAX_MEMBERS} user IDs, comma or newline separated", height=100)
        if st.button("Use these IDs"):
            ids, problem = parse_member_ids(typed)
            if problem:
                st.error(problem)
            else:
                try:
                    state["profiles"] = get("/group/users/profiles", params={"user_ids": ids})["users"]
                    state["members"] = ids
                    state["sample_note"] = "Entered by hand"
                except APIError as e:
                    show_error(e)

    # ---- sidebar: strategy ----------------------------------------------------------
    sb.header("Strategy")
    compare = sb.toggle("Compare two strategies", value=params.get("compare", "1") != "0")
    default_a = params.get("strategy", "average")
    strategy_a = sb.selectbox("Strategy A" if compare else "Strategy", STRATEGIES,
                              index=STRATEGIES.index(default_a) if default_a in STRATEGIES else 0)
    strategy_b = None
    if compare:
        default_b = params.get("strategy_b", "least_misery")
        strategy_b = sb.selectbox("Strategy B", STRATEGIES,
                                  index=STRATEGIES.index(default_b) if default_b in STRATEGIES else 1)
    norm = sb.selectbox(
        "Normalization", list(NORM_HELP), index=0,
        help="How each member's scores are put on a common scale before aggregating. "
             + "; ".join(f"{name}: {text}" for name, text in NORM_HELP.items()),
    )
    lam = 0.5
    if "fairness" in (strategy_a, strategy_b):
        lam = sb.slider("λ (fairness)", 0.0, 1.0, 0.5, 0.05,
                        help="fairness score = mean − λ·(max − min) over members")
    k = sb.number_input("Results (k)", min_value=1, max_value=50, value=10)
    include_closed = sb.toggle("Include closed restaurants", value=False)

    # deep link: ?cohesion=…&seed=…&n=… draws a group once on load
    if "cohesion" in params and "members" not in state:
        try:
            load_sample(int(params.get("n", 4)), params["cohesion"], int(params.get("seed", 42)))
        except APIError as e:
            show_error(e)

    # ---- main ------------------------------------------------------------------------
    st.title("Group Dinner")
    members = state.get("members")
    if not members:
        st.info("Sample a group or enter user IDs in the sidebar to begin.")
        return
    labels = member_labels(members)

    payload = {"members": members, "norm": norm, "lam": lam, "k": int(k), "include_closed": include_closed}
    if compare:
        payload["strategies"] = [strategy_a, strategy_b]
    else:
        payload["strategy"] = strategy_a
    try:
        response = cached_post("/group/compare" if compare else "/group/recommend",
                               json.dumps(payload, sort_keys=True))
    except APIError as e:
        show_error(e)
        return
    results = response["results"] if compare else [response]

    # group and cohesion
    cohesion = results[0]["cohesion"]
    c1, c2 = st.columns([1, 4])
    c1.metric("Group cohesion", f"{cohesion:.2f}",
              help=f"Mean pairwise similarity of the members' taste profiles. Among val groups, a third "
                   f"are below {COHESION_LOW} and a third above {COHESION_HIGH}.")
    c2.markdown(f"**{cohesion_reading(cohesion)}**  \n{state.get('sample_note', '')} · {len(members)} members")
    for col, user in zip(st.columns(len(members)), state.get("profiles", [])):
        with col:
            st.markdown(f"**{labels[user['user_id']]}** · `{user['user_id'][:10]}…`")
            st.caption("Top-rated: " + "; ".join(f"{p['name']} ({p['stars']:.0f}★)" for p in user["profile"]))

    y_min, y_max = member_score_bounds(results, norm)
    member_bars = st.column_config.BarChartColumn(
        " · ".join(labels.values()), y_min=y_min, y_max=y_max, width=110,
        help=f"Each member's predicted score ({norm}), M1 to Mn left to right",
    )

    def metrics(summary: dict, deltas: dict | None = None) -> None:
        m1, m2, m3 = st.columns(3)
        d = deltas or {}
        m1.metric("Group score (predicted)", f"{summary['group_utility_predicted']:.3f}",
                  delta=f"{d['group_utility_predicted']:+.3f} vs A" if d else None)
        m2.metric("Worst member (predicted)", f"{summary['worst_member_predicted']:.3f}",
                  delta=f"{d['worst_member_predicted']:+.3f} vs A" if d else None)
        m3.metric("Fairness gap (predicted)", f"{summary['fairness_gap_predicted']:.3f}",
                  delta=f"{d['fairness_gap_predicted']:+.3f} vs A" if d else None, delta_color="inverse")

    def per_dinner(res: dict) -> None:
        if res["strategy"] == "fairness_repeated":
            with st.expander("Per-dinner view (fairness_repeated: one pick per dinner, in order)"):
                st.caption("Weights show how much each member counted when that dinner was picked; "
                           "members served worst so far get more weight.")
                st.dataframe(pd.DataFrame(dinner_rows(res, labels)), hide_index=True, width="stretch")

    if compare:
        id_sets = [{r["business_id"] for r in res["restaurants"]} for res in results]
        st.subheader(f"{strategy_a} vs. {strategy_b}")
        st.markdown(f"**List overlap: {response['overlap']} of {len(results[0]['restaurants'])}** "
                    "(highlighted rows appear in both lists)")
        deltas = metric_deltas(results[0]["summary"], results[1]["summary"])
        for i, (col, res, other) in enumerate(zip(st.columns(2), results, reversed(id_sets))):
            with col:
                st.markdown(f"### {'A' if i == 0 else 'B'}: {res['strategy']}")
                metrics(res["summary"], deltas if i == 1 else None)
                table = pd.DataFrame(display_rows(res, other, labels))
                table["members"] = table[list(labels.values())].values.tolist()
                styled = table.style.apply(
                    lambda row: ["background-color: rgba(255, 196, 0, 0.25)" if row["in_both"] else ""
                                 for _ in row],
                    axis=1,
                )
                st.dataframe(styled, hide_index=True, width="stretch", column_config={
                    "in_both": None,
                    "rank": st.column_config.NumberColumn("#", width=30),
                    "name": st.column_config.TextColumn("name", width=170),
                    "categories": st.column_config.TextColumn("categories", width=130),
                    "price": st.column_config.TextColumn("price", width=45),
                    "score": st.column_config.NumberColumn("score", format="%.3f", width=55),
                    "members": member_bars,
                    **{label: None for label in labels.values()},
                })
                per_dinner(res)
    else:
        res = results[0]
        st.subheader(res["strategy"])
        metrics(res["summary"])
        left, right = st.columns([3, 2])
        with left:
            table = pd.DataFrame(display_rows(res, set(), labels))
            st.dataframe(table[["rank", "name", "categories", "price", "score"]], hide_index=True,
                         width="stretch", column_config={
                             "rank": st.column_config.NumberColumn("#", width=30),
                             "score": st.column_config.NumberColumn("score", format="%.3f"),
                         })
        with right:
            st.markdown("**Top 5: each member's predicted score**")
            chart = pd.DataFrame(top_member_rows(res, labels)).set_index("restaurant")
            st.bar_chart(chart, stack=False, horizontal=True, height=320,
                         x_label=f"predicted score ({norm})", y_label="")
        per_dinner(res)

    # map
    points = pd.DataFrame(map_points(results))
    st.subheader("Map")
    if points.empty:
        st.caption("No coordinates for these restaurants.")
    else:
        st.map(points, latitude="lat", longitude="lon", color="color", size=40, height=360)
        if compare:
            st.caption("Blue: only in A · orange: only in B · purple: in both.")

    with st.expander("How this works", expanded=False):
        st.markdown(
            "- Each member's taste is the average embedding of the restaurants they rated 4+ stars, and each "
            "restaurant's embedding comes from its review text. Scores are **predicted** from those review "
            "embeddings, not measured satisfaction.\n"
            "- Scores are put on a common per-member scale (the *Normalization* setting), then combined by the "
            "strategy: **average** (mean over members), **least_misery** (the least happy member), "
            "**fairness** (mean − λ·spread), or **fairness_repeated** (a sequence of dinners that takes turns "
            "favoring whoever has been served worst so far).\n"
            "- Predicted scores reflect similarity to each member's history and are **weakly calibrated** to "
            "actual star ratings (Spearman ≈ 0.01 on held-out reviews). "
            f"See [README: Group evaluation]({README_CALIBRATION}).\n"
            "- Places a member already reviewed are left out, and closed restaurants are hidden unless "
            "*Include closed restaurants* is on. A chain appears once, as \"(N locations)\"."
        )


if __name__ == "__main__":
    main()
