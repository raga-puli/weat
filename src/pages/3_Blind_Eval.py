"""Blind Eval page (Person B): pairwise preference between least_misery and average, for the
12 fixed groups in data/human_eval/groups.csv (python -m src.human_eval --select-groups).

LOCAL ONLY: it writes data/human_eval/responses.csv and refuses to run when DATA_SOURCE=db (the
hosted demo). Lists come from the API's /group/compare through src/ui_common.py; the page shows
only "List 1" / "List 2", never strategy names or scores. The trial order and left/right
assignment come from src.human_eval.trial_plan (seeded by the participant code). Protocol:
docs/human_eval_protocol.md.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `src.`

# Must match the exact wording in docs/human_eval_protocol.md, section 4.
QUESTION = "Which list would you pick for this group's dinner?"


def main() -> None:
    import streamlit as st
    from dotenv import load_dotenv

    from src import human_eval as he
    from src import ui_common

    load_dotenv()
    st.set_page_config(page_title="Blind Eval", page_icon=":material/balance:", layout="wide")
    st.title("Blind evaluation")

    if os.getenv("DATA_SOURCE", "files").strip().lower() == "db":
        st.error("This page runs locally only (DATA_SOURCE=db looks like the hosted demo).")
        st.stop()
    if not he.GROUPS_CSV.exists():
        st.error(f"{he.GROUPS_CSV} is missing. Run `python -m src.human_eval --select-groups` once first.")
        st.stop()

    state = st.session_state
    if "participant_code" not in state:
        st.markdown("Enter the anonymous participant code you were given. **Do not enter your name.**")
        with st.form("code"):
            code = st.text_input("Participant code", max_chars=32)
            if st.form_submit_button("Start"):
                code = code.strip()
                if he.valid_code(code):
                    state.participant_code = code
                    st.rerun()
                st.error("Codes are 2-32 letters, digits, '-' or '_'.")
        st.stop()

    code = state.participant_code
    groups = he.load_groups().set_index("group_id")
    plan = he.trial_plan(code, list(groups.index))
    done = he.answered_groups(code)
    remaining = [t for t in plan if t["group_id"] not in done]
    st.caption(f"Participant {code} · {len(plan) - len(remaining)} of {len(plan)} done")
    st.progress((len(plan) - len(remaining)) / len(plan))
    if not remaining:
        st.success("All done. Thank you!")
        st.stop()

    trial = remaining[0]
    group = groups.loc[trial["group_id"]]
    members = list(group["members"])
    key = f"trial:{trial['group_id']}"
    if key not in state:
        try:
            profiles = ui_common.get("/group/users/profiles", params={"user_ids": members})["users"]
            comparison = ui_common.post("/group/compare", json={
                "members": members, "strategies": list(he.STRATEGIES), "norm": he.NORM})
        except ui_common.APIError as e:
            st.error(str(e))
            st.stop()
        by_strategy = {r["strategy"]: r["restaurants"] for r in comparison["results"]}
        state[key] = {
            "profiles": profiles,
            "lists": [by_strategy[trial["left_strategy"]], by_strategy[he.right_strategy(trial["left_strategy"])]],
        }
    shown = state[key]

    st.subheader(f"Group {len(plan) - len(remaining) + 1}")
    st.markdown("Each member's top-rated restaurants:")
    for i, (col, user) in enumerate(zip(st.columns(len(shown["profiles"])), shown["profiles"]), start=1):
        with col, st.container(border=True):
            st.markdown(f"**Member {i}**")
            for p in user["profile"]:
                st.markdown(f"- {p['name']} ({p['stars']:.0f}★)")

    st.markdown(f"### {QUESTION}")
    for col, label, restaurants in zip(st.columns(2), ("List 1", "List 2"), shown["lists"]):
        with col, st.container(border=True):
            st.markdown(f"**{label}**")
            for rank, r in enumerate(restaurants, start=1):
                price = "$" * r["price_level"] if r.get("price_level") else ""
                details = " · ".join(x for x in (r.get("categories") or "", price) if x)
                st.markdown(f"{rank}. {r['name']}" + (f"  \n<small>{details}</small>" if details else ""),
                            unsafe_allow_html=True)

    reason = st.text_area("Why? (optional)", key=f"reason:{trial['group_id']}")
    for col, choice in zip(st.columns(len(he.CHOICES)), he.CHOICES):
        if col.button(choice, key=f"choice:{choice}", use_container_width=True):
            he.append_response(he.response_row(code, trial["group_id"], group["cohesion_bucket"],
                                               trial["left_strategy"], choice, reason))
            del state[key]
            st.rerun()


if __name__ == "__main__":
    main()
