"""Streamlit landing page: `streamlit run src/app.py` (shared and frozen).

Feature pages live in src/pages/ and talk to the API through src/ui_common.py.
"""

import sys
from pathlib import Path

import streamlit as st

# `streamlit run src/app.py` puts src/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui_common import API_URL, APIError, get  # noqa: E402

st.set_page_config(page_title="WEat", page_icon="🍽️")
st.title("WEat: group restaurant recommender")
st.write(
    "WEat recommends Philadelphia restaurants for a group, balancing everyone's tastes using "
    "sentence embeddings of Yelp reviews. "
    "Use the pages in the sidebar to plan a group dinner or search for a specific kind of place."
)

try:
    get("/health")
    st.badge(f"API online ({API_URL})", icon=":material/check_circle:", color="green")
except APIError as e:
    st.badge("API offline", icon=":material/error:", color="red")
    st.caption(str(e))
