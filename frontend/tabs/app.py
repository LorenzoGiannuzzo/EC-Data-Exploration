"""Data Visualizator — Streamlit frontend."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

# st.set_page_config MUST be the first Streamlit call → resolve favicon now.
ASSETS_DIR = Path(__file__).parent / "assets"
_favicon_candidates = [
    ASSETS_DIR / "logo_polito.png",
    ASSETS_DIR / "logo_ec_polito.png",
    ASSETS_DIR / "logo_energy_center.png",
]
_favicon = next((str(p) for p in _favicon_candidates if p.exists()), "📊")

st.set_page_config(
    page_title="PoliTo — Data Lake",
    page_icon=_favicon,
    layout="wide",
    menu_items={
        "Get help":     None,
        "Report a bug": None,
        "About": (
            "### Data Visualizator — PoliTo.\n\n"
            "Electrical load-profile analysis service.\n\n"
            "---\n\n"
            "**For support, contact:**\n\n"
            "Lorenzo Giannuzzo\n\n"
            "✉ lorenzo.giannuzzo@polito.it\n\n"
            "Energy Center Lab, DENERG, Politecnico di Torino."
        ),
    },
)

import api
from charts import fmt_int
from tabs import arera, clustering, gse, load_profiler, outliers, overview

st.markdown("""
<style>
    [data-testid="stApp"], .main, [data-testid="stMain"] {
        background-color: #0d1f3c !important;
    }
    [data-testid="stSidebar"] { background-color: #1a3a6b !important; }
    .stTabs [data-baseweb="tab"] { color: #e8f4fd; }
    .stTabs [aria-selected="true"] { color: white; border-bottom-color: white !important; }
    h1, h2, h3, h4, p, span, label, div { color: #e8f4fd; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Pushes the PoliTo-EC logo down so it visually centers with the title row */
    .header-logo-wrap { padding-top: 22px; }

    /* Tech-logo strip sits directly under the title's caption */
    .tech-logo-row {
        display: flex; align-items: center; gap: 28px;
        margin: 10px 0 36px 0; padding: 0;
        min-height: 32px;
    }
    .tech-logo-row img {
        height: 32px; width: auto; object-fit: contain;
        filter: brightness(0) invert(1) opacity(0.85);
    }

    /* Sub-caption sitting tightly under each st.metric */
    [data-testid="stMetric"] + div p {
        font-size: 12px !important;
        color: #aac4e6 !important;
        margin-top: -8px !important;
    }
</style>
""", unsafe_allow_html=True)


def _first_existing(*candidates: str) -> Path | None:
    for c in candidates:
        p = ASSETS_DIR / c
        if p.exists():
            return p
    return None


def _img_to_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


main_logo = _first_existing(
    "logo_energy_center.png", "logo_ec_polito.png", "logo_polito.png",
)
TECH_LOGO_FILES = [
    "docker_logo.png",
    "postgresql_logo.jpg",
    "fastapi_logo.png",
]

# ── Header: logo + title side by side ────────────────────────────────────────
c_logo, c_title = st.columns([1.5, 4.5])
with c_logo:
    st.markdown('<div class="header-logo-wrap"></div>', unsafe_allow_html=True)
    if main_logo is not None:
        st.image(str(main_logo), width=420)
    else:
        st.image(
            "https://upload.wikimedia.org/wikipedia/it/4/45/"
            "Politecnico_di_Torino_-_Logo.svg",
            width=300,
        )
with c_title:
    st.title("Electrical Profiles Data Visualizator — PoliTo")
    st.caption(
        "Energy Center Lab, DENERG, Politecnico di Torino — "
        "Author: Lorenzo Giannuzzo  ✉ lorenzo.giannuzzo@polito.it."
    )
    tech_logos = [_first_existing(f) for f in TECH_LOGO_FILES]
    tech_logos = [p for p in tech_logos if p is not None]
    if tech_logos:
        imgs = "".join(
            f'<img src="{_img_to_data_uri(p)}" alt="{p.stem}" />'
            for p in tech_logos
        )
        st.markdown(f'<div class="tech-logo-row">{imgs}</div>',
                    unsafe_allow_html=True)

# ── DB status with grade-of-detail captions ──────────────────────────────────
try:
    info = api.info()
except api.BackendError as e:
    st.error(f"Backend unreachable: {e}.")
    st.stop()
except Exception as e:
    st.error(f"Backend not responding at `{api.BACKEND_URL}`: {e}.")
    st.stop()

tables = info.get("tables", {})

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("PODs:", fmt_int(tables.get('pod_metadata', 0)))
    st.caption("Unique POD identifiers.")
with c2:
    st.metric("Measurements:", fmt_int(tables.get('measurements', 0)))
    st.caption("Quarter-hourly resolution [15 min].")
with c3:
    st.metric("GSE Profiles:", fmt_int(tables.get('reference_profiles_gse', 0)))
    st.caption("Domestic / Other Uses / Public Lighting / Air Conditioning.")
with c4:
    st.metric("ARERA Profiles:",
              fmt_int(tables.get('reference_profiles_arera', 0)))
    st.caption("Households only (Residenti / Non Residenti).")

tab_overview, tab_loadprof, tab_cluster, tab_outliers, tab_gse, tab_arera = st.tabs([
    "Overview",
    "Load Profiler",
    "Clustering Explorer",
    "Outliers Detection",
    "GSE Profile Comparison",
    "ARERA Profile Comparison",
])
with tab_overview:
    overview.render()
with tab_loadprof:
    load_profiler.render()
with tab_cluster:
    clustering.render()
with tab_outliers:
    outliers.render()
with tab_gse:
    gse.render()
with tab_arera:
    arera.render()
