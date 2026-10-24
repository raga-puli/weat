"""Search page: natural-language restaurant search (Person B).

Talks to the API only through src/ui_common.py (POST /search, GET /search/anchors); no model code is
imported here. The parse is shown as chips; "Edit filters" sends a corrected parse back as
`parsed`, which the API runs without re-parsing (engine_used = "manual").
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run src/app.py` puts src/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import ui_common  # noqa: E402

EXAMPLES = [
    "cheap vegetarian near Penn campus, somewhere cozy",
    "thai within 1 mile of Rittenhouse, good for a date",
    "halal open now near Temple",
    "lively brunch spot in Fishtown under $30",
]
# The API's ParsedQuery dietary tags (src/query_parser.DIETARY); kept literal so the page imports no model code.
DIETARY = ["vegetarian", "vegan", "gluten_free", "halal", "kosher"]
# Selectbox options are strings: a literal None option is treated as "no selection" and snaps back.
ANY_PRICE, ANYWHERE = "Any", "Anywhere"
NO_RESULTS = "No restaurants match all filters — try loosening price or distance."
FILTER_COLOR, VIBE_COLOR = "blue", "violet"
ANCHOR_COLOR, RESTAURANT_COLOR = "#e4572e", "#2e86de"

st.set_page_config(page_title="Search · WEat", page_icon="🔎", layout="wide")
st.title("Search restaurants")

state = st.session_state
state.setdefault("query", "")
state.setdefault("result", None)  # last successful /search response
state.setdefault("error", None)  # (kind, message) from the last failed request
state.setdefault("requested_engine", "rules")
state.setdefault("runs", 0)  # bumps on every search, so the "Edit filters" widgets refill from the new parse


# ---------- API ----------

# STOPGAP until ui_common.py gets shared wake/503 handling in the next scaffolding PR: while the API's
# encoder warms up, /search returns 503 with Retry-After. ui_common.APIError keeps only the message,
# so the delay is read from the API's message text ("Retry in N s"), which repeats Retry-After.
WARMING_RE = re.compile(r"HTTP 503: .*starting up.*Retry in (\d+) s")
MAX_WARMING_RETRIES = 6
WARMING_MESSAGE = "Search is starting up, retrying..."


def warming_retry_after(err: Exception) -> int | None:
    """Seconds to wait if err is the API's warm-up 503, else None."""
    match = WARMING_RE.search(str(err))
    return int(match.group(1)) if match else None


def post_with_warmup_retry(path, payload, *, post, sleep, on_retry=lambda attempt, delay: None,
                           max_retries=MAX_WARMING_RETRIES):
    """post(path, json=payload), retried up to max_retries times while the API is warming up, waiting
    each time as long as it asks. Any other error, or the last warm-up 503, is raised."""
    for attempt in range(max_retries + 1):
        try:
            return post(path, json=payload)
        except ui_common.APIError as err:
            delay = warming_retry_after(err)
            if delay is None or attempt == max_retries:
                raise
            on_retry(attempt + 1, delay)
            sleep(delay)


@st.cache_data(ttl=3600, show_spinner=False)
def load_anchors() -> dict[str, tuple[float, float]]:
    return {a["name"]: (a["lat"], a["lon"]) for a in ui_common.get("/search/anchors")}


def classify_error(err: ui_common.APIError) -> tuple[str, str]:
    """APIError carries only a message (ui_common is frozen), so the status is read from its text."""
    text = str(err)
    if warming_retry_after(err) is not None:
        return "warming", "Search is still starting up. Try again in a minute."
    if "HTTP 503" in text:
        return "db_down", (
            "The search database is unavailable. Switch the backend to **numpy** in the sidebar, "
            "or start Postgres with `docker compose up -d`."
        )
    if "HTTP 422" in text:
        return "invalid", f"The search request was rejected: {text.split(':', 1)[-1].strip()}"
    if "Can't reach the API" in text:
        return "api_down", f"{text}"
    if "didn't answer" in text:
        return "timeout", (
            f"{text} The LLM parser can be slow on this machine; try the **rules** engine in the sidebar."
        )
    return "other", text


def run_search(text: str, parsed: dict | None = None) -> None:
    payload = {
        "text": text,
        "engine": state.engine,
        "backend": state.backend,
        "k": state.k,
        "include_closed": state.include_closed,
    }
    if parsed is not None:
        payload["parsed"] = parsed
    state.requested_engine = "manual" if parsed is not None else state.engine
    notice = st.empty()
    try:
        with st.spinner("Searching…"):
            state.result = post_with_warmup_retry(
                "/search", payload, post=ui_common.post, sleep=time.sleep,
                on_retry=lambda attempt, delay: notice.info(WARMING_MESSAGE, icon=":material/hourglass_top:"),
            )
        state.error = None
    except ui_common.APIError as err:
        state.error = classify_error(err)
        notice.empty()
        if parsed is None:  # a failed manual rerun keeps the last results (and the filters being edited)
            state.result = None
        return
    notice.empty()
    state.runs += 1


def use_example(text: str) -> None:
    state.query = text
    state.pending = True


# ---------- formatting ----------


def price_label(level) -> str:
    return "$" * int(level) if level else "–"


def filter_chips(parsed: dict) -> list[str]:
    chips = []
    if parsed["max_price_level"]:
        chips.append(f"price ≤ {price_label(parsed['max_price_level'])}")
    if parsed["near"]:
        km = parsed["max_distance_km"]
        chips.append(f"near {parsed['near']}" + (f" (≤ {km:g} km)" if km else ""))
    elif parsed["max_distance_km"]:
        chips.append(f"within {parsed['max_distance_km']:g} km (ignored: no place named)")
    chips += [d.replace("_", "-") for d in parsed["dietary"]]
    if parsed["open_at"]:
        chips.append(f"open {parsed['open_at']}")
    chips += [f"category: {c}" for c in parsed["categories"]]
    return chips


def results_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": [r["name"] for r in results],
            "Categories": [", ".join(r["categories"]) for r in results],
            "Price": [price_label(r["price_level"]) for r in results],
            "Distance (km)": [r["distance_km"] for r in results],
            "Similarity": [r["similarity"] for r in results],
        }
    )


def map_points(results: list[dict], anchor: tuple[str, tuple[float, float]] | None) -> pd.DataFrame:
    rows = [
        {"lat": r["lat"], "lon": r["lon"], "color": RESTAURANT_COLOR, "size": 40}
        for r in results
        if r["lat"] is not None and r["lon"] is not None
    ]
    if anchor:
        lat, lon = anchor[1]
        rows.append({"lat": lat, "lon": lon, "color": ANCHOR_COLOR, "size": 120})
    return pd.DataFrame(rows, columns=["lat", "lon", "color", "size"])


# ---------- sidebar ----------

with st.sidebar:
    st.header("Search settings")
    st.radio(
        "Parser engine",
        ["rules", "llm"],
        format_func={"rules": "Rules (instant)", "llm": "LLM (slow, may fall back)"}.get,
        key="engine",
    )
    st.radio("Backend", ["sql", "numpy"], format_func={"sql": "SQL (Postgres)", "numpy": "numpy"}.get, key="backend")
    st.slider("Results (k)", 1, 50, 10, key="k")
    st.checkbox("Include closed restaurants", value=False, key="include_closed")


# ---------- query ----------

with st.form("search_form", border=False):
    cols = st.columns([5, 1], vertical_alignment="bottom")
    cols[0].text_input(
        "What are you looking for?", key="query", max_chars=300, placeholder="e.g. " + EXAMPLES[0]
    )
    submitted = cols[1].form_submit_button("Search", type="primary", width="stretch")

st.caption("Try an example:")
for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    col.button(example, on_click=use_example, args=(example,), width="stretch")

if submitted or state.pop("pending", False):
    if state.query.strip():
        run_search(state.query.strip())
    else:
        state.result, state.error = None, ("invalid", "Type what you're looking for first.")


# ---------- results ----------

if state.error:
    kind, message = state.error
    {"invalid": st.warning, "timeout": st.warning, "warming": st.warning}.get(kind, st.error)(message)

result = state.result
if result:
    parsed, engine_used = result["parsed"], result["engine_used"]

    chip_cols = st.container(horizontal=True)
    with chip_cols:
        for chip in filter_chips(parsed):
            st.badge(f"filter · {chip}", color=FILTER_COLOR, icon=":material/filter_alt:")
        if parsed["soft"]:
            st.badge(f"vibe match · {parsed['soft']}", color=VIBE_COLOR, icon=":material/auto_awesome:")
        st.badge(f"engine · {engine_used}", color="gray", icon=":material/memory:")
    if not filter_chips(parsed) and not parsed["soft"]:
        st.caption("Nothing was parsed from this query, so results are ranked by popularity.")
    if state.requested_engine == "llm" and engine_used == "rules":
        st.warning(
            "The LLM parser was unavailable or too slow, so the **rules** parser ran instead. "
            "Check the chips above and fix anything wrong under **Edit filters**."
        )

    t = result["timings_ms"]
    st.caption(f"parse {t['parse']:.0f} ms · search {t['search']:.0f} ms · total {t['total']:.0f} ms")

    results = result["results"]
    if not results:
        st.info(NO_RESULTS)
    else:
        st.dataframe(
            results_table(results),
            hide_index=True,
            width="stretch",
            column_config={
                "Distance (km)": st.column_config.NumberColumn(format="%.2f"),
                "Similarity": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        try:
            anchors = load_anchors()
        except ui_common.APIError:
            anchors = {}
        anchor = (parsed["near"], anchors[parsed["near"]]) if parsed["near"] in anchors else None
        st.map(map_points(results, anchor), latitude="lat", longitude="lon", color="color", size="size")
        if anchor:
            st.caption(f"🔴 {anchor[0]}  ·  🔵 results")

    # ---------- edit filters ----------

    with st.expander("Edit filters", expanded=False):
        st.caption("Correct the parse and rerun, instead of retyping the query.")
        try:
            anchor_names = list(load_anchors())
        except ui_common.APIError:
            anchor_names = [parsed["near"]] if parsed["near"] else []
        run = state.runs
        with st.form(f"edit_filters_{run}"):
            c1, c2 = st.columns(2)
            price_options = [ANY_PRICE] + [price_label(level) for level in (1, 2, 3, 4)]
            price = c1.selectbox(
                "Max price", price_options, index=parsed["max_price_level"] or 0,  # "$" * n sits at index n
            )
            near_options = [ANYWHERE] + anchor_names
            near = c2.selectbox(
                "Near", near_options,
                index=near_options.index(parsed["near"]) if parsed["near"] in near_options else 0,
            )
            distance = c1.number_input(
                "Max distance (km, 0 = default)", min_value=0.0, max_value=50.0, step=0.5,
                value=float(parsed["max_distance_km"] or 0.0),
            )
            open_at = c2.text_input("Open at ('now' or e.g. 'fri 19:00')", value=parsed["open_at"] or "")
            dietary = st.multiselect(
                "Dietary", DIETARY, default=[d for d in parsed["dietary"] if d in DIETARY],
                format_func=lambda d: d.replace("_", "-"),
            )
            categories = st.text_input("Categories (comma-separated)", value=", ".join(parsed["categories"]))
            soft = st.text_input("Vibe match text", value=parsed["soft"])
            if st.form_submit_button("Rerun with these filters", type="primary"):
                run_search(
                    state.query.strip() or "manual filters",
                    {
                        "max_price_level": None if price == ANY_PRICE else len(price),
                        "near": None if near == ANYWHERE else near,
                        "max_distance_km": distance or None,
                        "dietary": dietary,
                        "open_at": open_at.strip().lower() or None,
                        "categories": [c.strip().lower() for c in categories.split(",") if c.strip()],
                        "soft": soft.strip(),
                    },
                )
                st.rerun()
