"""Group Dinner page (Person A): compare two aggregation strategies for a group.

All data comes from the API through src/ui_common.py; this page never imports
model code. The pure helpers below are tested in tests/test_api_group.py; the
Streamlit UI runs only when Streamlit executes the page (__name__ == "__main__").

Deep link: ?cohesion=low&seed=42&n=4 draws a sample group and runs the
comparison on load (used for screenshots and sharing).
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
    st.title("Group Dinner")
    st.caption("Pick a group, choose two ways to aggregate their tastes, and compare the lists.")
    state = st.session_state
    params = st.query_params

    def show_error(e: Exception) -> None:
        st.error(format_api_error(str(e)))

    def load_sample(n: int, cohesion: str, seed: int) -> None:
        sample = get("/group/users/sample", params={"n": n, "cohesion": cohesion, "seed": seed})
        state["members"] = [u["user_id"] for u in sample["users"]]
        state["profiles"] = sample["users"]
        state["sample_note"] = (f"Sampled from val group {sample['group_id']} (cohesion {sample['cohesion']:.2f})"
                                if "group_id" in sample else "Random val users")
        state.pop("comparison", None)

    # ---- 1. members
    st.subheader("1. Members")
    draw_col, type_col = st.columns(2)
    with draw_col:
        st.markdown("**Draw a sample group**")
        cohesion = st.radio("Cohesion", ["low", "high", "any"], horizontal=True,
                            index=["low", "high", "any"].index(params.get("cohesion", "low"))
                            if params.get("cohesion", "low") in ("low", "high", "any") else 0)
        seed = st.number_input("Seed", min_value=0, value=int(params.get("seed", 42)), step=1)
        n = st.slider("Members", MIN_MEMBERS, MAX_MEMBERS, int(params.get("n", 4)))
        if st.button("Draw sample group"):
            try:
                load_sample(n, cohesion, int(seed))
            except APIError as e:
                show_error(e)
    with type_col:
        st.markdown("**Or enter user IDs**")
        typed = st.text_area(f"{MIN_MEMBERS}–{MAX_MEMBERS} user IDs, comma or newline separated", height=110)
        if st.button("Use these IDs"):
            ids, problem = parse_member_ids(typed)
            if problem:
                st.error(problem)
            else:
                try:
                    state["profiles"] = get("/group/users/profiles", params={"user_ids": ids})["users"]
                    state["members"] = ids
                    state["sample_note"] = "Entered by hand"
                    state.pop("comparison", None)
                except APIError as e:
                    show_error(e)

    # deep link: ?cohesion=…&seed=…&n=… draws and compares once on load
    if "cohesion" in params and "members" not in state:
        try:
            load_sample(int(params.get("n", 4)), params["cohesion"], int(params.get("seed", 42)))
            state["autorun"] = True
        except APIError as e:
            show_error(e)

    members = state.get("members")
    if not members:
        st.info("Draw a sample group or enter user IDs to begin.")
        return
    labels = member_labels(members)
    st.caption(state.get("sample_note", ""))
    profile_cols = st.columns(len(members))
    for col, user in zip(profile_cols, state.get("profiles", [])):
        with col:
            st.markdown(f"**{labels[user['user_id']]}** · `{user['user_id'][:10]}…`")
            for p in user["profile"]:
                st.markdown(f"- {p['name']} ({p['stars']:.0f}★)")

    # ---- 2. options
    st.subheader("2. Options")
    o1, o2, o3, o4 = st.columns([2, 2, 1, 1])
    strategy_a = o1.selectbox("Strategy A", STRATEGIES, index=0)
    strategy_b = o2.selectbox("Strategy B", STRATEGIES, index=1)
    k = o3.number_input("k", min_value=1, max_value=50, value=10)
    include_closed = o4.toggle("Include closed", value=False)

    if st.button("Compare", type="primary") or state.pop("autorun", False):
        try:
            state["comparison"] = post("/group/compare", json={
                "members": members, "strategies": [strategy_a, strategy_b],
                "k": int(k), "include_closed": include_closed,
            })
        except APIError as e:
            show_error(e)
            state.pop("comparison", None)

    comparison = state.get("comparison")
    if not comparison:
        return

    # ---- 3/4. side-by-side lists and summaries
    st.subheader("3. Comparison")
    results = comparison["results"]
    id_sets = [{r["business_id"] for r in res["restaurants"]} for res in results]
    st.markdown(f"**List overlap: {comparison['overlap']} of {len(results[0]['restaurants'])}** "
                "(highlighted rows appear in both lists)")
    for col, res, other in zip(st.columns(2), results, reversed(id_sets)):
        with col:
            summary = res["summary"]
            st.markdown(f"### {res['strategy']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Group score (predicted)", f"{summary['group_utility_predicted']:.3f}")
            m2.metric("Worst member (predicted)", f"{summary['worst_member_predicted']:.3f}")
            m3.metric("Fairness gap (predicted)", f"{summary['fairness_gap_predicted']:.3f}")
            table = pd.DataFrame(display_rows(res, other, labels))
            # one compact cell per row: a small bar per member, in M1..Mn order
            table["members"] = table[list(labels.values())].values.tolist()
            styled = table.style.apply(
                lambda row: ["background-color: rgba(255, 196, 0, 0.25)" if row["in_both"] else "" for _ in row],
                axis=1,
            )
            column_config = {
                "in_both": None,
                "rank": st.column_config.NumberColumn("#", width=30),
                "name": st.column_config.TextColumn("name", width=170),
                "categories": st.column_config.TextColumn("categories", width=130),
                "price": st.column_config.TextColumn("price", width=45),
                "score": st.column_config.NumberColumn("score", format="%.3f", width=55),
                "members": st.column_config.BarChartColumn(
                    " · ".join(labels.values()), y_min=0.0, y_max=1.0, width=110,
                    help="Each member's predicted score for this restaurant, M1 to Mn left to right",
                ),
                **{label: None for label in labels.values()},  # numbers stay in the data, not as columns
            }
            st.dataframe(styled, column_config=column_config, hide_index=True, width="stretch")
            if res["strategy"] == "fairness_repeated":
                with st.expander("Per-dinner view (fairness_repeated)", expanded=False):
                    st.dataframe(pd.DataFrame(dinner_rows(res, labels)), hide_index=True,
                                 width="stretch")
    st.caption(
        "Predicted scores reflect similarity to each member's history and are weakly calibrated to actual "
        f"star ratings ([README: Group evaluation]({README_CALIBRATION}))."
    )


if __name__ == "__main__":
    main()
