# ── Add this near the other overview_* helpers in frontend/api.py ────────────

@st.cache_data(ttl=600, show_spinner=False)
def overview_geography() -> dict:
    return get("/overview/geography")
