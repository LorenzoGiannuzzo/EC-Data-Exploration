"""
================================================================================
ATECO HIERARCHICAL CLUSTERING DASHBOARD
================================================================================
Author: Lorenzo Giannuzzo - Energy Center Lab, DENERG, Politecnico di Torino

Interactive hierarchical exploration of ATECO codes with automatic clustering.
Three-level checkbox system: L1 → L2 → L3, each triggering independent
clustering of daily load profiles for the selected PODs.

Launch:  streamlit run dashboard_ateco_clustering.py
================================================================================
"""


import io
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = Path(__file__).parent / "data"

MESI_IT = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
}
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
DIR_PATTERN = re.compile(r"^([a-zA-Z]{3})(\d{2})$")
Q_COLS = [f"Q{i}" for i in range(1, 97)]
Q_TIME_LABELS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]

META_TARGET_COLS = {
    "POD": ["POD", "pod", "Pod", "Codice POD"],
    "D_49DES": ["D_49DES", "d_49des", "D_49_DES", "D49DES"],
    "FDESC": ["FDESC", "fdesc", "F_DESC"],
    "TATE3DES": ["TATE3DES", "tate3des", "TATE3_DES"],
    "CCATETE": ["CCATETE", "ccatete", "Ccatete", "CCATE", "CodiceATECO"],
}

POTCONTR_VARIANTS = [
    "PotenzaContrattuale", "potenzacontrattuale", "POTENZACONTRATTUALE",
    "Potenza Contrattuale", "POTCONTR", "POT_CONTR", "POTCON",
    "PotContr", "potcontr", "Pot_Contr", "pot_contr",
    "Potenza_Contrattuale", "potenza_contrattuale",
]

COLORS_PLOTLY = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1

# Help texts for clustering quality metrics
METRIC_HELP = {
    "Silhouette Score": (
        "Measures how similar each POD is to its own cluster vs. other clusters.\n\n"
        "Range: -1 to +1\n"
        "• > 0.70: Strong structure (clusters are well-separated)\n"
        "• 0.50 – 0.70: Reasonable structure\n"
        "• 0.25 – 0.50: Weak structure (clusters overlap)\n"
        "• < 0.25: Poor or no meaningful structure\n\n"
        "Example: 0.65 means most PODs fit well in their cluster "
        "and are clearly distinct from neighboring clusters."
    ),
    "Calinski-Harabasz Index": (
        "Ratio of between-cluster dispersion to within-cluster dispersion. "
        "Higher = better defined clusters.\n\n"
        "No fixed range (depends on data size), but guidelines:\n"
        "• > 500: Excellent separation\n"
        "• 200 – 500: Good separation\n"
        "• 50 – 200: Moderate separation\n"
        "• < 50: Clusters are not well-separated\n\n"
        "Example: 350 means clusters are fairly compact internally "
        "and well-spaced from each other. Compare across different k "
        "values — the highest CH suggests the best k."
    ),
    "Davies-Bouldin Index": (
        "Average similarity between each cluster and its most similar one. "
        "Lower = better (clusters are more distinct).\n\n"
        "Range: 0 to ∞\n"
        "• < 0.5: Excellent separation\n"
        "• 0.5 – 1.0: Good separation\n"
        "• 1.0 – 1.5: Moderate (some overlap)\n"
        "• > 1.5: Poor separation (clusters blend into each other)\n\n"
        "Example: 0.8 means clusters have some overlap but are "
        "still distinguishable. Values close to 0 are ideal."
    ),
}

PEARSON_HELP = (
    "Pearson correlation coefficient (r) between cluster centroids.\n\n"
    "Measures the linear similarity of daily load shapes between clusters.\n\n"
    "Range: -1 to +1\n"
    "• r ≈ 1.0: Almost identical load shapes (same peaks and valleys)\n"
    "• r = 0.7 – 0.9: Similar overall shape, different magnitudes or slight shifts\n"
    "• r = 0.3 – 0.7: Partially similar (some common features)\n"
    "• r ≈ 0: Completely unrelated load patterns\n"
    "• r < 0: Opposite patterns (one peaks when the other dips)\n\n"
    "Example: r = 0.92 between Cl.1 and Cl.3 means they have nearly "
    "the same daily profile shape — they might be mergeable. "
    "r = 0.15 means very different consumption behaviors."
)

# Distributor-specific codes
DISTRIBUTOR_CODES = {
    "DO": "Domestic / Residential",
    "DO.01": "Domestic - Resident",
    "DO.02": "Domestic - Non-Resident",
    "CO": "Condominium services",
    "CO.01": "Condominium services - Resident",
    "CO.02": "Condominium services - Non-Resident",
    "IL": "Public lighting",
    "IL.01": "Public lighting",
    "DA": "Food products, beverages and tobacco",
    "DB": "Textiles and textile products",
    "DC": "Leather and leather products",
    "DD": "Wood and wood products",
    "DE": "Pulp, paper, publishing and printing",
    "DF": "Coke, petroleum products and nuclear fuel",
    "DG": "Chemicals and chemical products",
    "DH": "Rubber and plastic products",
    "DI": "Other non-metallic mineral products",
    "DJ": "Basic metals and fabricated metal products",
    "DK": "Machinery and equipment n.e.c.",
    "DL": "Electrical and optical equipment",
    "DM": "Transport equipment",
    "DN": "Manufacturing n.e.c.",
    "CA": "Mining of coal and lignite; extraction of peat",
    "CB": "Mining of metal ores",
}

ATECO_LOOKUP: dict[str, str] = {}
ATECO_EXCEL_NAME = "Note-esplicative-ATECO-2025-italiano-inglese.xlsx"


# ==============================================================================
# UTILITY FUNCTIONS (shared with main dashboard)
# ==============================================================================

def load_ateco_classification():
    global ATECO_LOOKUP
    excel_path = DATA_DIR / ATECO_EXCEL_NAME
    lookup = {}
    if excel_path.exists():
        try:
            df = pd.read_excel(excel_path, sheet_name=1, dtype=str,
                               usecols=["CODICE_ATECO_2025", "TITOLO_ITALIANO_ATECO_2025"])
            unique = df.drop_duplicates(subset=["CODICE_ATECO_2025"])
            for _, row in unique.iterrows():
                code = str(row["CODICE_ATECO_2025"]).strip()
                title = str(row["TITOLO_ITALIANO_ATECO_2025"]).strip()
                if code and title and code != "nan" and title != "nan":
                    lookup[code] = title
        except Exception:
            pass
    for code, desc in DISTRIBUTOR_CODES.items():
        if code not in lookup:
            lookup[code] = desc
    ATECO_LOOKUP = lookup
    return lookup


def lookup_ateco_description(code: str) -> str:
    if not code or code == "N/A":
        return ""
    code = str(code).strip()
    if code in ATECO_LOOKUP:
        return ATECO_LOOKUP[code]
    parts = code.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in ATECO_LOOKUP:
            return ATECO_LOOKUP[candidate]
    if parts[0] in ATECO_LOOKUP:
        return ATECO_LOOKUP[parts[0]]
    return ""


def parse_directory_name(dirname: str):
    match = DIR_PATTERN.match(dirname)
    if not match:
        return None
    mese_str = match.group(1).lower()
    anno_short = int(match.group(2))
    if mese_str not in MESI_IT:
        return None
    return mese_str, MESI_IT[mese_str], 2000 + anno_short


def find_metadata_file(directory: Path, dirname: str):
    exact = directory / f"Metadati POD {dirname}.xlsx"
    if exact.exists():
        return exact
    for f in directory.iterdir():
        if f.suffix.lower() == ".xlsx" and "metadati" in f.name.lower():
            return f
    return None


def find_measures_file(directory: Path):
    for f in directory.iterdir():
        if f.name.lower().startswith("misure_") and f.suffix.lower() == ".csv":
            return f
    for f in directory.iterdir():
        if f.suffix.lower() == ".csv" and "misure" in f.name.lower():
            return f
    return None


def find_column(df_columns, target_name, variants):
    for v in variants:
        if v in df_columns:
            return v
    col_map = {c.strip().lower(): c for c in df_columns}
    for v in variants:
        if v.strip().lower() in col_map:
            return col_map[v.strip().lower()]
    return None


def normalize_pod(series):
    return series.astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)


def parse_ateco(code: str) -> dict:
    if pd.isna(code) or str(code).strip() == "":
        return {"ATECO_L1": None, "ATECO_L2": None, "ATECO_L3": None}
    code = str(code).strip()
    parts = code.split(".")
    l1 = parts[0] if len(parts) >= 1 and parts[0] else None
    l2 = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 and parts[1] else None
    l3 = f"{parts[0]}.{parts[1]}.{parts[2]}" if len(parts) >= 3 and parts[2] else None
    return {"ATECO_L1": l1, "ATECO_L2": l2, "ATECO_L3": l3}


def fig_to_png_bytes(fig, width=1600, height=900, scale=2):
    """Robust PNG export with kaleido version handling."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale,
                            engine="kaleido")
    except Exception as e1:
        try:
            return fig.to_image(format="png", width=width, height=height, scale=scale)
        except Exception:
            raise RuntimeError(
                f"PNG export failed. Fix: pip install kaleido==0.2.1\n"
                f"Error: {e1}"
            )


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_all_data():
    if not DATA_DIR.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=["POD", "POTCONTR"]), []

    all_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    valid_dirs = []
    issues = []

    for d in all_dirs:
        parsed = parse_directory_name(d.name)
        if parsed:
            valid_dirs.append((d, parsed))
    valid_dirs.sort(key=lambda x: (x[1][2], x[1][1]))
    n_dirs = len(valid_dirs)

    if n_dirs == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=["POD", "POTCONTR"]), \
               ["No valid directories found"]

    meta_frames, meas_frames, potcontr_frames = [], [], []
    progress_bar = st.progress(0, text="Preparing...")
    status_text = st.empty()

    for i, (d, (mese_str, mese_num, anno)) in enumerate(valid_dirs):
        dirname = d.name
        periodo = f"{anno}-{mese_num:02d}"
        pct = int((i / n_dirs) * 100)
        progress_bar.progress(pct / 100, text=f"Loading {periodo} ({pct}%) - {dirname}")
        status_text.caption(f"Directory {i + 1}/{n_dirs}: {dirname}")

        # --- Metadata ---
        meta_path = find_metadata_file(d, dirname)
        if meta_path:
            try:
                df_m = pd.read_excel(meta_path, dtype=str)
                col_map = {}
                for target, variants in META_TARGET_COLS.items():
                    found = find_column(df_m.columns, target, variants)
                    if found:
                        col_map[found] = target
                if "POD" not in col_map.values():
                    continue
                df_m = df_m.rename(columns=col_map)
                meta_cols = [c for c in ["POD", "D_49DES", "FDESC", "TATE3DES", "CCATETE"]
                             if c in df_m.columns]
                df_m = df_m[meta_cols].copy()
                df_m["POD"] = normalize_pod(df_m["POD"])
                df_m = df_m[df_m["POD"].notna() & (df_m["POD"] != "") & (df_m["POD"] != "NAN")]
                df_m["Periodo"] = periodo
                meta_frames.append(df_m)
            except Exception as e:
                issues.append(f"Metadata error {dirname}: {e}")

        # --- Measures ---
        meas_path = find_measures_file(d)
        if meas_path:
            try:
                df_ms = pd.read_csv(meas_path, sep=None, engine="python", dtype=str)
                available_q = [c for c in Q_COLS if c in df_ms.columns]
                if "DataMisura" not in df_ms.columns or "POD" not in df_ms.columns:
                    continue

                # PotenzaContrattuale
                pot_col_found = None
                for v in POTCONTR_VARIANTS:
                    if v in df_ms.columns:
                        pot_col_found = v
                        break
                if pot_col_found is None:
                    col_map_lower = {c.strip().lower(): c for c in df_ms.columns}
                    for v in POTCONTR_VARIANTS:
                        if v.strip().lower() in col_map_lower:
                            pot_col_found = col_map_lower[v.strip().lower()]
                            break
                if pot_col_found is None:
                    for c in df_ms.columns:
                        cl = c.strip().lower().replace("_", "").replace(" ", "")
                        if "potenzacontrattuale" in cl or "potcontr" in cl:
                            pot_col_found = c
                            break
                if pot_col_found is None:
                    for c in df_ms.columns:
                        if "potenza" in c.strip().lower():
                            pot_col_found = c
                            break

                if pot_col_found is not None:
                    df_pot = df_ms[["POD", pot_col_found]].copy()
                    df_pot["POD"] = normalize_pod(df_pot["POD"])
                    df_pot = df_pot.rename(columns={pot_col_found: "POTCONTR"})
                    df_pot["POTCONTR"] = pd.to_numeric(
                        df_pot["POTCONTR"].astype(str)
                        .str.replace(",", ".", regex=False).str.strip(),
                        errors="coerce"
                    )
                    df_pot = df_pot.dropna(subset=["POTCONTR"])
                    df_pot = df_pot.drop_duplicates(subset=["POD"], keep="last")
                    if not df_pot.empty:
                        potcontr_frames.append(df_pot)

                cols = ["DataMisura", "POD"] + available_q
                # Tipologia column
                tipologia_col = None
                for candidate in ["Tipologia", "tipologia", "TIPOLOGIA", "Tipo", "tipo", "TIPO"]:
                    if candidate in df_ms.columns:
                        tipologia_col = candidate
                        break
                if tipologia_col is None:
                    col_map_tip = {c.strip().lower(): c for c in df_ms.columns}
                    if "tipologia" in col_map_tip:
                        tipologia_col = col_map_tip["tipologia"]
                if tipologia_col is not None:
                    cols.append(tipologia_col)

                df_ms = df_ms[cols].copy()
                if tipologia_col is not None and tipologia_col != "Tipologia":
                    df_ms = df_ms.rename(columns={tipologia_col: "Tipologia"})
                if "Tipologia" in df_ms.columns:
                    df_ms["Tipologia"] = df_ms["Tipologia"].astype(str).str.strip().str.upper()

                df_ms["POD"] = normalize_pod(df_ms["POD"])
                for col in available_q:
                    df_ms[col] = pd.to_numeric(
                        df_ms[col].astype(str).str.replace(",", ".", regex=False).str.strip(),
                        errors="coerce").astype(np.float32)
                df_ms["DataMisura"] = pd.to_datetime(df_ms["DataMisura"], dayfirst=True, errors="coerce")
                df_ms = df_ms[df_ms["POD"].notna() & (df_ms["POD"] != "") & (df_ms["POD"] != "NAN")]
                df_ms["Periodo"] = periodo
                meas_frames.append(df_ms)
            except Exception as e:
                issues.append(f"Measures error {dirname}: {e}")

    progress_bar.progress(0.95, text="Merging dataframes (95%)...")
    status_text.caption("Concatenating data...")

    df_meta = pd.concat(meta_frames, ignore_index=True) if meta_frames else pd.DataFrame()
    df_meas = pd.concat(meas_frames, ignore_index=True) if meas_frames else pd.DataFrame()
    df_potcontr = (
        pd.concat(potcontr_frames, ignore_index=True)
        .drop_duplicates(subset=["POD"], keep="last")
        if potcontr_frames else pd.DataFrame(columns=["POD", "POTCONTR"])
    )

    progress_bar.progress(1.0, text="Loading complete!")
    status_text.empty()
    progress_bar.empty()

    return df_meta, df_meas, df_potcontr, issues


@st.cache_data(show_spinner=False)
def prepare_metadata(df_meta, df_potcontr):
    df_unique = (
        df_meta.sort_values("Periodo")
        .drop_duplicates(subset=["POD"], keep="last").copy()
    )
    if not df_potcontr.empty:
        df_unique = df_unique.merge(
            df_potcontr[["POD", "POTCONTR"]].rename(columns={"POTCONTR": "POTCONTR_kW"}),
            on="POD", how="left"
        )
    else:
        df_unique["POTCONTR_kW"] = np.nan

    if "CCATETE" not in df_unique.columns:
        return df_unique, False
    ateco_parsed = df_unique["CCATETE"].apply(parse_ateco).apply(pd.Series)
    df_unique = pd.concat([df_unique, ateco_parsed], axis=1)
    for col in ["ATECO_L1", "ATECO_L2", "ATECO_L3"]:
        df_unique[col] = df_unique[col].fillna("N/A")
    return df_unique, True


@st.cache_data(show_spinner=False)
def compute_pods_with_12_months(_df_meas_periodi):
    pod_months = _df_meas_periodi.groupby("POD")["Periodo"].nunique()
    return set(pod_months[pod_months >= 12].index)


# ==============================================================================
# PROFILE COMPUTATION
# ==============================================================================

def compute_daily_profiles(df_meas, _prog=None):
    available_q = [c for c in Q_COLS if c in df_meas.columns]
    if not available_q:
        if _prog:
            _prog.empty()
        return pd.DataFrame(), pd.DataFrame()

    n_q = len(available_q)
    if _prog:
        _prog.progress(0.05, text="Extracting calendar month (5%)...")

    month_col = df_meas["DataMisura"].dt.month

    if _prog:
        _prog.progress(0.15, text="Averaging daily profiles per POD per month (15%)...")

    grouped = (
        df_meas[available_q]
        .astype(np.float32)
        .groupby([df_meas["POD"], month_col])
        .mean()
    )

    if _prog:
        _prog.progress(0.40, text="Building profile matrix (40%)...")

    unstacked = grouped.unstack(level=1)
    flat_cols = [f"M{int(month):02d}_{q}" for q, month in unstacked.columns]
    unstacked.columns = flat_cols

    # Lorenzo Giannuzzo: Step 1 — linear interpolation along the time axis
    # to fill small isolated NaN gaps within each month block before any
    # row-level filtering. limit=4 caps fills at 4 consecutive slots (1h)
    # to avoid over-extrapolating large gaps.
    unstacked = unstacked.interpolate(
        axis=1, method="linear", limit=4, limit_direction="both"
    )

    # Lorenzo Giannuzzo: Step 2 — for each month block independently,
    # forward-fill then backward-fill any residual NaN within that month.
    # This handles leading/trailing gaps within a single month that
    # interpolate() cannot reach with a finite limit.
    for m in range(1, 13):
        prefix = f"M{m:02d}_"
        m_cols = [c for c in unstacked.columns if c.startswith(prefix)]
        if not m_cols:
            continue
        unstacked[m_cols] = (
            unstacked[m_cols]
            .ffill(axis=1)
            .bfill(axis=1)
        )

    # Lorenzo Giannuzzo: Step 3 — any month block still entirely NaN
    # (POD has no data at all for that month) is filled with the row's
    # median across all other months, so the POD is not lost.
    # If even the row median is NaN (no data anywhere), fill with 0.
    row_median = unstacked.median(axis=1)
    for col in unstacked.columns:
        mask = unstacked[col].isna()
        if mask.any():
            unstacked.loc[mask, col] = row_median[mask]
    unstacked = unstacked.fillna(0)

    # Lorenzo Giannuzzo: Step 4 — drop only rows that are entirely zero
    # across all columns (POD with literally no measurements at all).
    all_zero_mask = (unstacked == 0).all(axis=1)
    unstacked = unstacked[~all_zero_mask]

    if _prog:
        _prog.progress(0.60, text="Interpolating missing values (60%)...")

    profile_raw = unstacked.copy()

    if _prog:
        _prog.progress(0.80, text="Normalizing profiles (min-max per month) (80%)...")

    profile_norm = profile_raw.copy()
    for m in range(1, 13):
        prefix = f"M{m:02d}_"
        m_cols = [c for c in profile_norm.columns if c.startswith(prefix)]
        if not m_cols:
            continue
        m_data = profile_norm[m_cols]
        m_min = m_data.min(axis=1)
        m_max = m_data.max(axis=1)
        m_range = m_max - m_min

        # Lorenzo Giannuzzo: when range == 0 (flat profile for the entire
        # month) the normalized value is 0 everywhere instead of NaN,
        # so the POD is retained rather than silently dropped.
        has_range = m_range > 0
        normalized = m_data.sub(m_min, axis=0).div(
            m_range.where(has_range, other=1.0), axis=0
        )
        # For flat rows, force the result to 0 (min-max of a constant = 0)
        normalized.loc[~has_range] = 0.0
        profile_norm[m_cols] = normalized

    # Lorenzo Giannuzzo: No more dropna() here — NaN cannot exist at this
    # point because all gaps were filled above. Clip to [0, 1] as a safety
    # guard against floating-point rounding at the boundaries.
    profile_norm = profile_norm.clip(0.0, 1.0)

    if _prog:
        _prog.progress(1.0, text="Profiles complete!")
        _prog.empty()

    return profile_norm, profile_raw


def get_overall_avg_profile(profile_df):
    all_months_data = []
    for m in range(1, 13):
        prefix = f"M{m:02d}_"
        cols = [c for c in profile_df.columns if c.startswith(prefix)]
        if not cols:
            continue
        sub = profile_df[cols].copy()
        sub.columns = [c.replace(prefix, "") for c in cols]
        all_months_data.append(sub)
    if not all_months_data:
        return pd.DataFrame()
    stacked = pd.concat(all_months_data, keys=range(len(all_months_data)))
    return stacked.groupby(level=1).mean()


def get_single_month_profile(profile_df, month_number):
    """Extract normalized profile for a single month (1-12)."""
    prefix = f"M{month_number:02d}_"
    cols = [c for c in profile_df.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame()
    sub = profile_df[cols].copy()
    sub.columns = [c.replace(prefix, "") for c in cols]
    return sub.dropna()


# Month labels for the selector
MONTH_LABELS = [
    "Overall Average",
    "January (01)", "February (02)", "March (03)", "April (04)",
    "May (05)", "June (06)", "July (07)", "August (08)",
    "September (09)", "October (10)", "November (11)", "December (12)",
]


# ==============================================================================
# CLUSTERING ENGINE
# ==============================================================================

def find_optimal_k(X, Z, k_range):
    n = len(X)
    if n < 4:
        return 3, {}

    results = {k: {"sil": None, "ch": None, "db": None, "inertia": None} for k in k_range}

    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        n_unique = len(set(labels))
        if n_unique < 2 or n_unique > n - 1:
            continue
        try:
            results[k]["sil"] = silhouette_score(X, labels)
        except Exception:
            pass
        try:
            results[k]["ch"] = calinski_harabasz_score(X, labels)
        except Exception:
            pass
        try:
            results[k]["db"] = davies_bouldin_score(X, labels)
        except Exception:
            pass
        inertia = 0
        for c in set(labels):
            members = X[labels == c]
            centroid = members.mean(axis=0)
            inertia += ((members - centroid) ** 2).sum()
        results[k]["inertia"] = inertia

    votes = {k: 0 for k in k_range}
    method_picks = {}

    sil_vals = {k: v["sil"] for k, v in results.items() if v["sil"] is not None}
    if sil_vals:
        best = max(sil_vals, key=sil_vals.get)
        votes[best] += 1
        method_picks["Silhouette"] = best

    ch_vals = {k: v["ch"] for k, v in results.items() if v["ch"] is not None}
    if ch_vals:
        best = max(ch_vals, key=ch_vals.get)
        votes[best] += 1
        method_picks["Calinski-Harabasz"] = best

    db_vals = {k: v["db"] for k, v in results.items() if v["db"] is not None}
    if db_vals:
        best = min(db_vals, key=db_vals.get)
        votes[best] += 1
        method_picks["Davies-Bouldin"] = best

    elbow_k = None
    inertia_vals = {k: v["inertia"] for k, v in results.items() if v["inertia"] is not None}
    if len(inertia_vals) >= 3:
        ks_sorted = sorted(inertia_vals.keys())
        drops = {}
        for i in range(1, len(ks_sorted)):
            prev_k, curr_k = ks_sorted[i - 1], ks_sorted[i]
            prev_i, curr_i = inertia_vals[prev_k], inertia_vals[curr_k]
            if prev_i > 0:
                drops[curr_k] = (prev_i - curr_i) / prev_i
        if drops:
            avg_drop = np.mean(list(drops.values()))
            for k in sorted(drops.keys()):
                if drops[k] < avg_drop:
                    elbow_k = k
                    break
            if elbow_k is None:
                elbow_k = sorted(drops.keys())[0]
            votes[elbow_k] += 1
            method_picks["Elbow"] = elbow_k

    try:
        n_ref = 10
        gap_vals = {}
        for k in k_range:
            labels_k = fcluster(Z, t=k, criterion="maxclust")
            wk = 0
            for c in set(labels_k):
                members = X[labels_k == c]
                if len(members) > 1:
                    wk += pdist(members).sum() / len(members)
            wk_refs = []
            for _ in range(n_ref):
                X_rand = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
                if len(X_rand) > 1:
                    Z_rand = linkage(X_rand, method="average")
                    labels_rand = fcluster(Z_rand, t=k, criterion="maxclust")
                    wk_r = 0
                    for c in set(labels_rand):
                        mem = X_rand[labels_rand == c]
                        if len(mem) > 1:
                            wk_r += pdist(mem).sum() / len(mem)
                    wk_refs.append(np.log(max(wk_r, 1e-10)))
            if wk_refs:
                gap_vals[k] = np.mean(wk_refs) - np.log(max(wk, 1e-10))
        if gap_vals:
            best = max(gap_vals, key=gap_vals.get)
            votes[best] += 1
            method_picks["Gap Statistic"] = best
    except Exception:
        pass

    max_votes = max(votes.values()) if any(v > 0 for v in votes.values()) else 0
    if max_votes > 0:
        candidates = [k for k, v in votes.items() if v == max_votes]
        if len(candidates) == 1:
            optimal_k = candidates[0]
        else:
            if elbow_k is not None and elbow_k in candidates:
                optimal_k = elbow_k
            else:
                optimal_k = min(candidates)
    else:
        optimal_k = 3

    return optimal_k, {"votes": votes, "method_picks": method_picks, "metrics": results}


def run_clustering_for_pods(profile_norm, pod_list, n_clusters=None,
                           profile_month=0):
    """Run clustering on a subset of PODs.
    profile_month: 0 = overall average, 1-12 = specific month.
    If n_clusters is None, auto-detect optimal k.
    Returns: (X_df_with_cluster, optimal_k, details, Z, error_msg)
    """
    available = profile_norm.index.intersection(pod_list)
    sub = profile_norm.loc[available]

    if profile_month == 0:
        X_df = get_overall_avg_profile(sub)
    else:
        X_df = get_single_month_profile(sub, profile_month)
    X_df = X_df.dropna()

    if len(X_df) < 5:
        return None, None, None, None, f"Need ≥5 PODs with profiles (have {len(X_df)})"

    X = X_df.values
    Z = linkage(X, method="average", metric="euclidean")

    if n_clusters is not None:
        optimal_k = n_clusters
        details = {}
    else:
        max_k = min(10, max(3, int(np.sqrt(len(X)))))
        k_range = range(3, max_k + 1)
        optimal_k, details = find_optimal_k(X, Z, k_range)

    labels = fcluster(Z, t=optimal_k, criterion="maxclust")
    X_df = X_df.copy()
    X_df["Cluster"] = labels

    return X_df, optimal_k, details, Z, None


# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

def compute_centroid_pearson(X_df):
    """Compute Pearson correlation matrix between cluster centroids."""
    q_cols = [c for c in X_df.columns if c.startswith("Q")]
    clusters = sorted(X_df["Cluster"].unique())
    centroids = {}
    for cl in clusters:
        centroids[cl] = X_df[X_df["Cluster"] == cl][q_cols].mean().values

    n_cl = len(clusters)
    corr_matrix = np.ones((n_cl, n_cl))
    pval_matrix = np.zeros((n_cl, n_cl))

    for i in range(n_cl):
        for j in range(i + 1, n_cl):
            r, p = pearsonr(centroids[clusters[i]], centroids[clusters[j]])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
            pval_matrix[i, j] = p
            pval_matrix[j, i] = p

    labels = [f"Cl.{cl}" for cl in clusters]
    df_corr = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    df_pval = pd.DataFrame(pval_matrix, index=labels, columns=labels)
    return df_corr, df_pval, centroids, clusters


def compute_cluster_stats(X_df):
    """Compute comprehensive cluster statistics."""
    q_cols = [c for c in X_df.columns if c.startswith("Q")]
    clusters = sorted(X_df["Cluster"].unique())
    rows = []

    for cl in clusters:
        cl_data = X_df[X_df["Cluster"] == cl][q_cols]
        profile_mean = cl_data.mean()
        profile_std = cl_data.std()

        peak_idx = profile_mean.idxmax()
        trough_idx = profile_mean.idxmin()
        peak_q = int(peak_idx.replace("Q", "")) - 1
        trough_q = int(trough_idx.replace("Q", "")) - 1
        peak_time = Q_TIME_LABELS[peak_q] if peak_q < len(Q_TIME_LABELS) else "?"
        trough_time = Q_TIME_LABELS[trough_q] if trough_q < len(Q_TIME_LABELS) else "?"

        if len(cl_data) > 1:
            intra_dist = pdist(cl_data.values).mean()
        else:
            intra_dist = 0.0

        mean_std = profile_std.mean()

        rows.append({
            "Cluster": cl,
            "N_PODs": len(cl_data),
            "% Total": f"{len(cl_data)/len(X_df)*100:.1f}%",
            "Peak Time": peak_time,
            "Trough Time": trough_time,
            "Max (norm)": round(profile_mean.max(), 4),
            "Min (norm)": round(profile_mean.min(), 4),
            "Mean (norm)": round(profile_mean.mean(), 4),
            "Avg Intra-Dist": round(intra_dist, 4),
            "Avg Profile Std": round(mean_std, 4),
        })

    return pd.DataFrame(rows)


def compute_global_metrics(X_df):
    """Compute global clustering quality metrics."""
    q_cols = [c for c in X_df.columns if c.startswith("Q")]
    X = X_df[q_cols].values
    labels = X_df["Cluster"].values

    metrics = {}
    try:
        metrics["Silhouette Score"] = round(silhouette_score(X, labels), 4)
    except Exception:
        metrics["Silhouette Score"] = "N/A"
    try:
        metrics["Calinski-Harabasz Index"] = round(calinski_harabasz_score(X, labels), 2)
    except Exception:
        metrics["Calinski-Harabasz Index"] = "N/A"
    try:
        metrics["Davies-Bouldin Index"] = round(davies_bouldin_score(X, labels), 4)
    except Exception:
        metrics["Davies-Bouldin Index"] = "N/A"

    return metrics


# ==============================================================================
# EXPORT HELPERS — NEW
# ==============================================================================

def build_all_centroids_df(valid_results: dict) -> pd.DataFrame:
    """Build a single DataFrame with centroids from ALL computed levels.

    Lorenzo Giannuzzo: returns a DataFrame indexed by (Level, Cluster) with
    Q1..Q96 centroid values. Each row identifies the level (L1/L2/L3),
    the cluster number, the number of PODs in that cluster, and the
    normalised centroid profile across all 96 quarter-hour slots.
    """
    rows = []
    for level_name, res in valid_results.items():
        if "error" in res:
            continue
        X_df = res["X_df"]
        k = res["k"]
        q_cols = [c for c in X_df.columns if c.startswith("Q")]
        for cl in sorted(X_df["Cluster"].unique()):
            cl_data = X_df[X_df["Cluster"] == cl][q_cols]
            centroid = cl_data.mean()
            row = {"Level": level_name, "Cluster": int(cl),
                   "N_PODs": len(cl_data), "k_total": int(k)}
            for q in q_cols:
                row[q] = round(float(centroid[q]), 6)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    meta_cols = ["Level", "Cluster", "N_PODs", "k_total"]
    q_cols_out = [c for c in df.columns if c.startswith("Q")]
    return df[meta_cols + q_cols_out]


def build_pod_cluster_map(valid_results: dict) -> dict:
    """Return a dict: {level_name: {POD: cluster_id}} for all valid levels.

    Lorenzo Giannuzzo: used to add multi-level cluster columns to the
    Profiles_with_Cluster export sheet.
    """
    mapping = {}
    for level_name, res in valid_results.items():
        if "error" in res:
            continue
        X_df = res["X_df"]
        mapping[level_name] = X_df["Cluster"].to_dict()  # index = POD
    return mapping


def build_ateco_dominant_cluster_df(valid_results: dict, df_base: pd.DataFrame) -> pd.DataFrame:
    """For every ATECO code (at every level), find the dominant cluster
    for each computed level.

    Lorenzo Giannuzzo: returns a DataFrame with columns:
        ATECO_Level | ATECO_Code | Description | Total_PODs |
        Cluster_L1 (dominant) | Count_L1 | Pct_L1 |
        Cluster_L2 (dominant) | Count_L2 | Pct_L2 |
        Cluster_L3 (dominant) | Count_L3 | Pct_L3
    Only levels that were actually computed are included.
    Rows are sorted by ATECO_Level then descending Total_PODs.
    """
    # Build POD→cluster mapping for each level
    pod_cluster = build_pod_cluster_map(valid_results)
    if not pod_cluster:
        return pd.DataFrame()

    # For each ATECO level column that exists in df_base, collect dominant cluster
    ateco_level_cols = [c for c in ["ATECO_L1", "ATECO_L2", "ATECO_L3"]
                        if c in df_base.columns]

    computed_levels = list(pod_cluster.keys())  # e.g. ["L1", "L2", "L3"]

    all_rows = []
    for ateco_col in ateco_level_cols:
        ateco_label = ateco_col  # e.g. "ATECO_L1"
        sub = df_base[["POD", ateco_col]].drop_duplicates("POD")
        sub = sub[sub[ateco_col] != "N/A"]

        for ateco_code, grp in sub.groupby(ateco_col):
            pods_in_group = set(grp["POD"].tolist())
            total_pods = len(pods_in_group)
            desc = lookup_ateco_description(str(ateco_code))
            row = {
                "ATECO_Level": ateco_label,
                "ATECO_Code": ateco_code,
                "Description": desc,
                "Total_PODs": total_pods,
            }
            for lv in computed_levels:
                mapping = pod_cluster[lv]
                # Restrict to PODs that were actually clustered at this level
                clustered_pods = {p: mapping[p] for p in pods_in_group
                                  if p in mapping}
                if not clustered_pods:
                    row[f"Dominant_Cluster_{lv}"] = "—"
                    row[f"Count_{lv}"] = 0
                    row[f"Pct_{lv}"] = "—"
                else:
                    counts = pd.Series(clustered_pods).value_counts()
                    dom_cluster = int(counts.index[0])
                    dom_count = int(counts.iloc[0])
                    dom_pct = round(dom_count / len(clustered_pods) * 100, 1)
                    row[f"Dominant_Cluster_{lv}"] = dom_cluster
                    row[f"Count_{lv}"] = dom_count
                    row[f"Pct_{lv}"] = f"{dom_pct}%"
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(all_rows)
    # Sort: ATECO_Level, then descending Total_PODs
    df_out = df_out.sort_values(["ATECO_Level", "Total_PODs"],
                                ascending=[True, False]).reset_index(drop=True)
    return df_out


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_cluster_profiles(X_df, k, title_suffix=""):
    """Plot average daily load profiles per cluster."""
    q_cols = [c for c in X_df.columns if c.startswith("Q")]
    fig = go.Figure()
    clusters_sorted = sorted(X_df["Cluster"].unique())
    for idx, cl in enumerate(clusters_sorted):
        cl_data = X_df[X_df["Cluster"] == cl][q_cols]
        mean_p = cl_data.mean()
        std_p = cl_data.std()
        n = len(cl_data)
        x_labels = Q_TIME_LABELS[:len(q_cols)]
        color = COLORS_PLOTLY[idx % len(COLORS_PLOTLY)]

        fig.add_trace(go.Scatter(
            x=x_labels, y=mean_p, mode="lines",
            name=f"Cl.{cl} (n={n})",
            line=dict(width=2.5, color=color),
        ))
        try:
            if color.startswith("#"):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            elif color.startswith("rgb"):
                parts = color.replace("rgb(", "").replace(")", "").split(",")
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                r, g, b = 128, 128, 128
            fill_color = f"rgba({r},{g},{b},0.10)"
        except Exception:
            fill_color = "rgba(128,128,128,0.10)"

        fig.add_trace(go.Scatter(
            x=x_labels + x_labels[::-1],
            y=list((mean_p + std_p).clip(0, 1)) +
              list((mean_p - std_p).clip(0, 1))[::-1],
            fill="toself", fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False, name=f"±1σ Cl.{cl}",
        ))

    fig.update_layout(
        title=f"Daily Load Profiles (k={k}){title_suffix}",
        xaxis_title="Time of Day",
        yaxis_title="Normalized (0-1)",
        height=400,
        yaxis=dict(range=[-0.05, 1.05]),
        xaxis=dict(dtick=4, tickangle=-45, tickfont=dict(size=9)),
        margin=dict(t=40, b=60, l=50, r=20),
        legend=dict(font=dict(size=9)),
    )
    return fig


def plot_pearson_heatmap(df_corr):
    """Pearson correlation heatmap between cluster centroids."""
    labels = df_corr.columns.tolist()
    z = df_corr.values.tolist()
    text = [[f"{df_corr.iloc[i, j]:.3f}" for j in range(len(labels))]
            for i in range(len(labels))]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(
        title="Pearson Correlation Between Cluster Centroids",
        height=350,
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(side="bottom"),
    )
    return fig


# ==============================================================================
# CLUSTER COMPOSITION CHART
# ==============================================================================

def plot_cluster_composition(X_df, df_base, ateco_col, level_label):
    """Stacked horizontal bar showing ATECO composition within each cluster."""
    pod_cl = X_df[["Cluster"]].reset_index()
    pod_cl.columns = ["POD", "Cluster"]
    merged = pod_cl.merge(
        df_base[["POD", ateco_col]].drop_duplicates("POD"),
        on="POD", how="left"
    )
    merged["AtecoLabel"] = merged[ateco_col].apply(
        lambda c: f"{c} — {lookup_ateco_description(c)[:40]}"
        if lookup_ateco_description(c) else str(c)
    )
    comp = merged.groupby(["Cluster", "AtecoLabel"]).size().reset_index(name="Count")
    totals = merged.groupby("Cluster").size().reset_index(name="Total")
    comp = comp.merge(totals, on="Cluster")
    comp["Pct"] = (comp["Count"] / comp["Total"] * 100).round(1)
    comp["Text"] = comp.apply(
        lambda r: f"{int(r['Count'])} ({r['Pct']:.0f}%)" if r["Pct"] >= 8 else "",
        axis=1
    )
    comp["ClusterLabel"] = comp["Cluster"].apply(lambda c: f"Cluster {c}")

    n_clusters = comp["Cluster"].nunique()
    fig = px.bar(
        comp, y="ClusterLabel", x="Count", color="AtecoLabel",
        orientation="h", text="Text",
        color_discrete_sequence=COLORS_PLOTLY,
    )
    fig.update_layout(
        barmode="stack",
        title=f"Cluster Composition — {level_label}",
        xaxis_title="Number of PODs",
        yaxis_title="",
        legend_title="ATECO Code",
        height=max(300, n_clusters * 70 + 140),
        margin=dict(t=40, b=50, l=100, r=20),
        legend=dict(font=dict(size=9)),
        yaxis=dict(automargin=True),
        xaxis=dict(tickangle=0),
    )
    fig.update_traces(
        textposition="inside", textfont_size=9,
        insidetextanchor="middle", textangle=0,
    )
    summary = comp[["ClusterLabel", "AtecoLabel", "Count", "Pct"]].copy()
    summary.columns = ["Cluster", "ATECO", "N. PODs", "%"]
    return fig, summary


def build_ateco_cluster_breakdown(X_df, df_base, ateco_col, level_label):
    """Build a table showing, for each ATECO code, how its PODs distribute
    across clusters (count + percentage)."""
    pod_cl = X_df[["Cluster"]].reset_index()
    pod_cl.columns = ["POD", "Cluster"]
    merged = pod_cl.merge(
        df_base[["POD", ateco_col]].drop_duplicates("POD"),
        on="POD", how="left"
    )
    merged["AtecoDesc"] = merged[ateco_col].apply(
        lambda c: lookup_ateco_description(c)[:50]
        if lookup_ateco_description(c) else ""
    )
    cross = merged.groupby([ateco_col, "AtecoDesc", "Cluster"]).size() \
        .reset_index(name="Count")
    ateco_totals = merged.groupby(ateco_col).size().reset_index(name="Total")
    cross = cross.merge(ateco_totals, on=ateco_col)
    cross["Pct"] = (cross["Count"] / cross["Total"] * 100).round(1)
    cross["Label"] = cross.apply(
        lambda r: f"{int(r['Count'])} ({r['Pct']:.1f}%)", axis=1
    )
    pivot = cross.pivot_table(
        index=[ateco_col, "AtecoDesc", "Total"],
        columns="Cluster", values="Label",
        aggfunc="first", fill_value="—"
    ).reset_index()
    pivot.columns.name = None
    cl_cols = [c for c in pivot.columns if isinstance(c, (int, np.integer))]
    rename_map = {c: f"Cluster {c}" for c in cl_cols}
    pivot = pivot.rename(columns=rename_map)
    pivot = pivot.rename(columns={
        ateco_col: "ATECO Code", "AtecoDesc": "Description", "Total": "Total PODs"
    })
    pivot = pivot.sort_values("Total PODs", ascending=False)
    return pivot


def build_ateco_breakdown_html(pivot_df, level_label, k):
    """Convert ATECO breakdown pivot table to styled HTML."""
    html = (
        "<html><head><style>"
        "body{font-family:Arial,sans-serif;padding:20px;}"
        "h2{color:#333;}"
        "table{border-collapse:collapse;width:100%;}"
        "th{background:#2c3e50;color:white;padding:8px 10px;"
        "text-align:center;font-size:12px;}"
        "td{border:1px solid #ddd;padding:6px 10px;text-align:center;"
        "font-size:11px;}"
        "td:first-child,td:nth-child(2){text-align:left;}"
        "tr:nth-child(even){background:#f9f9f9;}"
        "tr:hover{background:#e8f4fd;}"
        ".total{font-weight:bold;}"
        "</style></head><body>"
        f"<h2>ATECO → Cluster Breakdown — {level_label} (k={k})</h2>"
        "<p>Each cell shows: count (percentage of that ATECO code's PODs)</p>"
    )
    html += pivot_df.to_html(index=False, escape=False, na_rep="—")
    html += "</body></html>"
    return html


# ==============================================================================
# CLUSTERING BLOCK RENDERER
# ==============================================================================

def render_clustering_block(profile_norm, pod_list, label, key_prefix,
                            manual_k=None):
    n_pods = len(pod_list)
    pods_with_profile = set(profile_norm.index) & set(pod_list)
    n_with_profile = len(pods_with_profile)

    if n_pods == 0:
        st.info(f"**{label}**: No PODs selected.")
        return False

    if n_with_profile < 5:
        st.warning(f"**{label}**: {n_pods} PODs selected, but only "
                   f"{n_with_profile} have load profiles (need ≥5).")
        return False

    X_df, optimal_k, details, Z, error = run_clustering_for_pods(
        profile_norm, list(pods_with_profile), n_clusters=manual_k
    )

    if error:
        st.error(f"**{label}**: {error}")
        return False

    k_source = "manual" if manual_k else "auto"
    st.markdown(f"**{label}** — {n_with_profile} PODs, k={optimal_k} ({k_source})")

    fig = plot_cluster_profiles(X_df, optimal_k, f" | {label}")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_profiles")

    stats_df = compute_cluster_stats(X_df)
    st.dataframe(stats_df, hide_index=True, use_container_width=True,
                 key=f"{key_prefix}_stats")

    global_metrics = compute_global_metrics(X_df)
    m_cols = st.columns(len(global_metrics))
    for col, (metric_name, metric_val) in zip(m_cols, global_metrics.items()):
        col.metric(metric_name, metric_val, help=METRIC_HELP.get(metric_name, ""))

    if optimal_k >= 2:
        df_corr, df_pval, _, _ = compute_centroid_pearson(X_df)
        fig_corr = plot_pearson_heatmap(df_corr)
        st.plotly_chart(fig_corr, use_container_width=True,
                        key=f"{key_prefix}_pearson")
        with st.expander("ℹ️ How to read Pearson r / P-values"):
            st.markdown(PEARSON_HELP)
            st.markdown("---")
            st.markdown("**P-values matrix:**")
            st.dataframe(df_pval.style.format("{:.2e}"), use_container_width=True)

    return True


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    st.set_page_config(
        page_title="ATECO Hierarchical Clustering",
        page_icon="📊",
        layout="wide",
    )

    st.title("ATECO Hierarchical Clustering Explorer")
    st.caption("Energy Center Lab, DENERG, Politecnico di Torino")

    st.markdown("""
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCheckbox"] label {
        white-space: nowrap;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow"] {
        overflow-x: auto !important;
        overflow-y: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
        "Interactive three-level ATECO code selection with automatic hierarchical "
        "clustering. Select codes at each level to trigger independent clustering "
        "of the corresponding POD subsets."
    )

    if "ateco_loaded" not in st.session_state:
        load_ateco_classification()
        st.session_state["ateco_loaded"] = True
        if len(ATECO_LOOKUP) <= len(DISTRIBUTOR_CODES):
            st.warning(
                f"ATECO 2025 classification file not found in `data/{ATECO_EXCEL_NAME}`. "
                "Using distributor codes only."
            )
    elif not ATECO_LOOKUP:
        load_ateco_classification()

    if "data_loaded" not in st.session_state or "df_potcontr" not in st.session_state:
        for key in ["data_loaded", "df_meta", "df_meas", "df_potcontr", "issues",
                     "profile_norm", "profile_raw", "pods_12m"]:
            st.session_state.pop(key, None)

        df_meta, df_meas, df_potcontr, issues = load_all_data()

        if not df_meas.empty:
            q_present = [c for c in Q_COLS if c in df_meas.columns]
            df_meas[q_present] = df_meas[q_present].apply(
                pd.to_numeric, errors="coerce"
            ).astype(np.float32)
            for cat_col in ["POD", "Tipologia"]:
                if cat_col in df_meas.columns:
                    df_meas[cat_col] = df_meas[cat_col].astype("category")
            if "DataMisura" in df_meas.columns and not pd.api.types.is_datetime64_any_dtype(df_meas["DataMisura"]):
                df_meas["DataMisura"] = pd.to_datetime(df_meas["DataMisura"], errors="coerce")

        st.session_state["df_meta"] = df_meta
        st.session_state["df_meas"] = df_meas
        st.session_state["df_potcontr"] = df_potcontr
        st.session_state["issues"] = issues
        st.session_state["data_loaded"] = True
    else:
        df_meta = st.session_state["df_meta"]
        df_meas = st.session_state["df_meas"]
        df_potcontr = st.session_state["df_potcontr"]
        issues = st.session_state["issues"]

    if df_meta.empty:
        st.error("No data found in data/ folder.")
        return

    df_unique, has_ateco = prepare_metadata(df_meta, df_potcontr)
    if not has_ateco:
        st.error("CCATETE column not found in metadata.")
        return

    # =========================================================================
    # SIDEBAR FILTERS
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Global Filters")

        if st.button("🔄 Reload Data"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        has_tipologia = ("Tipologia" in df_meas.columns
                         and df_meas["Tipologia"].notna().any())
        if has_tipologia:
            tip_values = sorted(df_meas["Tipologia"].dropna().unique().tolist())
            tip_counts = df_meas.groupby("Tipologia")["POD"].nunique()
            sel_tipologia = st.radio(
                "Measure Type (Tipologia)",
                ["All"] + tip_values,
                format_func=lambda x: (
                    f"All ({df_meas['POD'].nunique():,} PODs)" if x == "All"
                    else f"{x} ({tip_counts.get(x, 0):,} PODs)"
                ),
                index=0,
                key="tip_filter",
                help=(
                    "Codici tipologia misura dal distributore:\n\n"
                    "ENERGIA ATTIVA (kWh):\n"
                    "• AP = Attiva Prelevata (consumo dalla rete)\n"
                    "• AN = Attiva iNiettata (immissione in rete, es. FV)\n\n"
                    "ENERGIA REATTIVA INDUTTIVA (kvarh):\n"
                    "• RLP = Reattiva induttiva (Lavorata) Prelevata\n"
                    "• RLN = Reattiva induttiva (Lavorata) iNiettata\n\n"
                    "ENERGIA REATTIVA CAPACITIVA (kvarh):\n"
                    "• RCN = Reattiva Capacitiva iNiettata\n"
                    "• RCP = Reattiva Capacitiva Prelevata\n\n"
                    "Per analisi consumi usare AP. Per analisi immissione (FV) usare AN. "
                    "Le reattive servono per analisi power quality e penali cosφ."
                )
            )
            st.divider()
        else:
            sel_tipologia = "All"

        if sel_tipologia != "All" and has_tipologia:
            df_meas_filtered = df_meas[df_meas["Tipologia"] == sel_tipologia].copy()
        else:
            df_meas_filtered = df_meas

        coverage_opt = st.radio(
            "Data coverage",
            ["All PODs", "12+ months only"],
            index=0,
            help="Filter to PODs with ≥12 distinct months of data."
        )
        use_12m_filter = (coverage_opt == "12+ months only")

        profile_period = st.selectbox(
            "Profile period",
            MONTH_LABELS,
            index=0,
            help=(
                "Select which period to use for clustering:\n\n"
                "• **Overall Average**: averages the daily load profile across "
                "all available months (default, most robust).\n"
                "• **Single month**: clusters based only on that month's "
                "profile. Useful to detect seasonal patterns or compare "
                "summer vs winter behaviors."
            ),
        )
        profile_month = MONTH_LABELS.index(profile_period)

        st.divider()

        st.header("⚡ Contractual Power")
        has_potcontr = (
            "POTCONTR_kW" in df_unique.columns
            and df_unique["POTCONTR_kW"].notna().any()
        )

        pot_filter_mask = pd.Series(True, index=df_unique.index)

        if has_potcontr:
            power_levels = (
                df_unique.dropna(subset=["POTCONTR_kW"])
                .groupby("POTCONTR_kW")["POD"].nunique()
                .sort_index()
            )
            enable_pot_filter = st.checkbox(
                "Enable power filter", value=False, key="pot_en"
            )
            if enable_pot_filter:
                selected_levels = []
                with st.container(height=200):
                    for kw_val, n_pods_kw in power_levels.items():
                        default = st.session_state.get(f"pot2_{kw_val}", True)
                        checked = st.checkbox(
                            f"{kw_val:.1f} kW ({n_pods_kw})",
                            value=default, key=f"pot2_{kw_val}",
                        )
                        if checked:
                            selected_levels.append(kw_val)
                include_missing = st.checkbox(
                    "Include PODs with missing power data",
                    value=False, key="pot2_na"
                )
                if selected_levels:
                    pot_filter_mask = df_unique["POTCONTR_kW"].isin(selected_levels)
                else:
                    pot_filter_mask = pd.Series(False, index=df_unique.index)
                if include_missing:
                    pot_filter_mask = pot_filter_mask | df_unique["POTCONTR_kW"].isna()
        else:
            st.info("No contractual power data available.")
            enable_pot_filter = False

        st.divider()

        st.header("⚙️ Cluster Settings")
        cluster_mode = st.radio(
            "Number of clusters",
            ["Automatic (majority vote)", "Manual"],
            index=0, key="cluster_mode"
        )
        manual_k = None
        if cluster_mode == "Manual":
            manual_k = st.slider("k (number of clusters)", 2, 15, 4, key="manual_k_slider")

        st.divider()

        st.header("📊 Statistics")
        n_pod_total = df_unique["POD"].nunique()
        n_pod_meas = df_meas_filtered["POD"].nunique()
        st.metric("PODs (metadata)", f"{n_pod_total:,}")
        st.metric("PODs (measures)", f"{n_pod_meas:,}")
        if has_tipologia and sel_tipologia != "All":
            st.caption(f"Filtered: {sel_tipologia}")

        if issues:
            with st.expander(f"⚠ {len(issues)} warnings"):
                for iss in issues[:20]:
                    st.text(iss)

    # =========================================================================
    # PROFILE COMPUTATION
    # =========================================================================
    prev_tipologia = st.session_state.get("_prev_tip2", None)
    if prev_tipologia is not None and prev_tipologia != sel_tipologia:
        st.session_state.pop("profile_norm", None)
        st.session_state.pop("profile_raw", None)
        st.session_state.pop("pods_12m", None)
    st.session_state["_prev_tip2"] = sel_tipologia

    if "profile_norm" not in st.session_state:
        prog = st.progress(0, text="Computing daily load profiles...")
        profile_norm, profile_raw = compute_daily_profiles(df_meas_filtered, prog)
        st.session_state["profile_norm"] = profile_norm
        st.session_state["profile_raw"] = profile_raw
    else:
        profile_norm = st.session_state["profile_norm"]

    if "pods_12m" not in st.session_state:
        st.session_state["pods_12m"] = compute_pods_with_12_months(
            df_meas_filtered[["POD", "Periodo"]].drop_duplicates()
        )
    pods_12m = st.session_state["pods_12m"]

    # =========================================================================
    # APPLY GLOBAL FILTERS
    # =========================================================================
    df_base = df_unique.copy()

    if use_12m_filter:
        df_base = df_base[df_base["POD"].isin(pods_12m)]

    if has_potcontr and enable_pot_filter:
        df_base = df_base[pot_filter_mask.reindex(df_base.index, fill_value=False)]

    meas_pods = set(df_meas_filtered["POD"].unique())
    df_base = df_base[df_base["POD"].isin(meas_pods)]

    n_base = df_base["POD"].nunique()

    filter_parts = []
    if has_tipologia and sel_tipologia != "All":
        filter_parts.append(f"Type={sel_tipologia}")
    if use_12m_filter:
        filter_parts.append("12+ months")
    if has_potcontr and enable_pot_filter:
        filter_parts.append("Power filter ON")
    if profile_month > 0:
        filter_parts.append(f"Month={MONTH_LABELS[profile_month]}")

    if filter_parts:
        st.info(f"**Active filters:** {' | '.join(filter_parts)} → "
                f"**{n_base:,} PODs** available")
    else:
        st.info(f"No global filters → **{n_base:,} PODs** available")

    if profile_norm.empty:
        st.error("No load profiles available. Check data.")
        return

    l1_counts = (
        df_base.groupby('ATECO_L1')['POD'].nunique()
        .reset_index().rename(columns={'POD': 'N'})
        .sort_values('N', ascending=False)
    )
    ateco_clustering_section(df_base, profile_norm, df_unique, manual_k,
                             profile_month, l1_counts)


# ==============================================================================
# FRAGMENT: ATECO SELECTION + CLUSTERING
# ==============================================================================

@st.fragment
def ateco_clustering_section(df_base, profile_norm, df_unique, manual_k,
                             profile_month=0, l1_counts=None):

    def _frag_rerun():
        try:
            st.rerun(scope="fragment")
        except Exception:
            st.rerun()

    st.markdown("---")
    st.subheader("ATECO Level Selection")
    st.markdown(
        "Select ATECO codes at each level. L2 options appear based on L1 selections, "
        "L3 based on L2. Press **Run Clustering** to compute results."
    )

    all_l1_codes = l1_counts["ATECO_L1"].tolist()

    col_l1, col_l2, col_l3 = st.columns(3)

    # ===================== LEVEL 1 =====================
    selected_l1 = []
    with col_l1:
        st.markdown("#### Level 1 (Section)")
        sa1, da1 = st.columns(2)
        with sa1:
            if st.button("Select All", key="sa_l1", use_container_width=True):
                for c in all_l1_codes:
                    st.session_state[f"cb_l1_{c}"] = True
                _frag_rerun()
        with da1:
            if st.button("Deselect All", key="da_l1", use_container_width=True):
                for c in all_l1_codes:
                    st.session_state[f"cb_l1_{c}"] = False
                _frag_rerun()

        with st.container(height=350):
            for _, row in l1_counts.iterrows():
                code = row["ATECO_L1"]
                n_pods = row["N"]
                desc = lookup_ateco_description(code)
                label = f"{code} — {desc}" if desc else code
                if f"cb_l1_{code}" not in st.session_state:
                    st.session_state[f"cb_l1_{code}"] = False
                checked = st.checkbox(f"{label} ({n_pods})", key=f"cb_l1_{code}")
                if checked:
                    selected_l1.append(code)

    # ===================== LEVEL 2 =====================
    selected_l2 = []
    with col_l2:
        st.markdown("#### Level 2 (Division)")
        if not selected_l1:
            st.caption("← Select at least one L1 code to see L2 options.")
        else:
            df_l1_filtered = df_base[df_base["ATECO_L1"].isin(selected_l1)]
            l2_data = (
                df_l1_filtered.groupby(["ATECO_L1", "ATECO_L2"])["POD"].nunique()
                .reset_index().rename(columns={"POD": "N"})
            )
            l2_data = l2_data[l2_data["ATECO_L2"] != "N/A"]

            if l2_data.empty:
                st.caption("No L2 codes available for the selected L1 codes.")
            else:
                all_l2_codes = l2_data["ATECO_L2"].tolist()
                sa2, da2 = st.columns(2)
                with sa2:
                    if st.button("Select All", key="sa_l2", use_container_width=True):
                        for c in all_l2_codes:
                            st.session_state[f"cb_l2_{c}"] = True
                        for l1c in selected_l1:
                            st.session_state[f"cb_l2par_{l1c}"] = True
                        _frag_rerun()
                with da2:
                    if st.button("Deselect All", key="da_l2", use_container_width=True):
                        for c in all_l2_codes:
                            st.session_state[f"cb_l2_{c}"] = False
                        for l1c in selected_l1:
                            st.session_state[f"cb_l2par_{l1c}"] = False
                        _frag_rerun()

                with st.container(height=350):
                    for l1_code in selected_l1:
                        l2_of_l1 = l2_data[l2_data["ATECO_L1"] == l1_code] \
                            .sort_values("N", ascending=False)
                        if l2_of_l1.empty:
                            continue

                        l1_desc = lookup_ateco_description(l1_code)
                        l1_total = int(l2_of_l1["N"].sum())
                        child_codes_l2 = l2_of_l1["ATECO_L2"].tolist()

                        all_kids_on = all(
                            st.session_state.get(f"cb_l2_{c}", False)
                            for c in child_codes_l2
                        )
                        prev_par = st.session_state.get(f"_pv_l2par_{l1_code}", None)

                        if f"cb_l2par_{l1_code}" not in st.session_state:
                            st.session_state[f"cb_l2par_{l1_code}"] = all_kids_on
                        parent_val = st.checkbox(
                            f"▸ {l1_code} — {l1_desc} ({l1_total})",
                            key=f"cb_l2par_{l1_code}",
                        )
                        if prev_par is not None and parent_val != prev_par:
                            for c in child_codes_l2:
                                st.session_state[f"cb_l2_{c}"] = parent_val
                            st.session_state[f"_pv_l2par_{l1_code}"] = parent_val
                            _frag_rerun()
                        st.session_state[f"_pv_l2par_{l1_code}"] = parent_val

                        with st.expander(
                            f"  ↳ show {len(child_codes_l2)} sub-codes",
                            expanded=st.session_state.get(f"_exp_l2_{l1_code}", True),
                        ):
                            for _, row in l2_of_l1.iterrows():
                                code = row["ATECO_L2"]
                                n_pods = row["N"]
                                desc = lookup_ateco_description(code)
                                label = f"{code} — {desc}" if desc else str(code)
                                if f"cb_l2_{code}" not in st.session_state:
                                    st.session_state[f"cb_l2_{code}"] = False
                                checked = st.checkbox(
                                    f"{label} ({n_pods})", key=f"cb_l2_{code}"
                                )
                                if checked:
                                    selected_l2.append(code)

    # ===================== LEVEL 3 =====================
    selected_l3 = []
    with col_l3:
        st.markdown("#### Level 3 (Class)")
        if not selected_l2:
            st.caption("← Select at least one L2 code to see L3 options.")
        else:
            df_l2_filtered = df_base[df_base["ATECO_L2"].isin(selected_l2)]
            l3_data = (
                df_l2_filtered.groupby(["ATECO_L1", "ATECO_L2", "ATECO_L3"])["POD"]
                .nunique().reset_index().rename(columns={"POD": "N"})
            )
            l3_data = l3_data[l3_data["ATECO_L3"] != "N/A"]

            if l3_data.empty:
                st.caption("No L3 codes available for the selected L2 codes.")
            else:
                all_l3_codes = l3_data["ATECO_L3"].tolist()
                sa3, da3 = st.columns(2)
                with sa3:
                    if st.button("Select All", key="sa_l3", use_container_width=True):
                        for c in all_l3_codes:
                            st.session_state[f"cb_l3_{c}"] = True
                        for l2c in selected_l2:
                            st.session_state[f"cb_l3par_{l2c}"] = True
                        _frag_rerun()
                with da3:
                    if st.button("Deselect All", key="da_l3", use_container_width=True):
                        for c in all_l3_codes:
                            st.session_state[f"cb_l3_{c}"] = False
                        for l2c in selected_l2:
                            st.session_state[f"cb_l3par_{l2c}"] = False
                        _frag_rerun()

                with st.container(height=350):
                    for l2_code in selected_l2:
                        l3_of_l2 = l3_data[l3_data["ATECO_L2"] == l2_code] \
                            .sort_values("N", ascending=False)
                        if l3_of_l2.empty:
                            continue

                        l2_desc = lookup_ateco_description(l2_code)
                        l2_total = int(l3_of_l2["N"].sum())
                        child_codes_l3 = l3_of_l2["ATECO_L3"].tolist()

                        all_kids_on_l3 = all(
                            st.session_state.get(f"cb_l3_{c}", False)
                            for c in child_codes_l3
                        )
                        prev_par_l3 = st.session_state.get(f"_pv_l3par_{l2_code}", None)

                        if f"cb_l3par_{l2_code}" not in st.session_state:
                            st.session_state[f"cb_l3par_{l2_code}"] = all_kids_on_l3
                        parent_val_l3 = st.checkbox(
                            f"▸ {l2_code} — {l2_desc} ({l2_total})",
                            key=f"cb_l3par_{l2_code}",
                        )
                        if prev_par_l3 is not None and parent_val_l3 != prev_par_l3:
                            for c in child_codes_l3:
                                st.session_state[f"cb_l3_{c}"] = parent_val_l3
                            st.session_state[f"_pv_l3par_{l2_code}"] = parent_val_l3
                            _frag_rerun()
                        st.session_state[f"_pv_l3par_{l2_code}"] = parent_val_l3

                        with st.expander(
                            f"  ⤷ show {len(child_codes_l3)} sub-codes",
                            expanded=st.session_state.get(f"_exp_l3_{l2_code}", True),
                        ):
                            for _, row in l3_of_l2.iterrows():
                                code = row["ATECO_L3"]
                                n_pods = row["N"]
                                desc = lookup_ateco_description(code)
                                label = f"{code} — {desc}" if desc else str(code)
                                if f"cb_l3_{code}" not in st.session_state:
                                    st.session_state[f"cb_l3_{code}"] = False
                                checked = st.checkbox(
                                    f"{label} ({n_pods})", key=f"cb_l3_{code}"
                                )
                                if checked:
                                    selected_l3.append(code)

    # =========================================================================
    # SELECTION SUMMARY + RUN BUTTON
    # =========================================================================
    if not selected_l1 and not selected_l2 and not selected_l3:
        st.info("Select ATECO codes above, then press **Run Clustering**.")
        return

    pods_l1 = list(df_base[df_base["ATECO_L1"].isin(selected_l1)]["POD"].unique()) \
        if selected_l1 else []
    pods_l2 = list(df_base[df_base["ATECO_L2"].isin(selected_l2)]["POD"].unique()) \
        if selected_l2 else []
    pods_l3 = list(df_base[df_base["ATECO_L3"].isin(selected_l3)]["POD"].unique()) \
        if selected_l3 else []

    active_levels = []
    if selected_l1:
        active_levels.append(("L1", selected_l1, pods_l1, "ATECO_L1"))
    if selected_l2:
        active_levels.append(("L2", selected_l2, pods_l2, "ATECO_L2"))
    if selected_l3:
        active_levels.append(("L3", selected_l3, pods_l3, "ATECO_L3"))

    summary_parts = []
    for lname, codes, pods, _ in active_levels:
        codes_str = ", ".join(codes[:3]) + ("..." if len(codes) > 3 else "")
        n_prof = len(set(profile_norm.index) & set(pods))
        summary_parts.append(f"**{lname}**: {codes_str} ({n_prof} PODs with profile)")
    st.markdown(" | ".join(summary_parts))

    sel_fingerprint = (
        tuple(sorted(selected_l1)),
        tuple(sorted(selected_l2)),
        tuple(sorted(selected_l3)),
        manual_k,
        profile_month,
    )
    cached_fp = st.session_state.get("_cl_fingerprint", None)
    has_cached = (
        cached_fp == sel_fingerprint and "_cl_results" in st.session_state
    )

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        run_clicked = st.button("▶ Run Clustering", type="primary", key="run_cl_btn")
    with btn_col2:
        if has_cached:
            st.success("Results cached ✓")
        elif cached_fp is not None and cached_fp != sel_fingerprint:
            st.warning("Selection changed — click Run")

    # =========================================================================
    # COMPUTE OR LOAD CACHED
    # =========================================================================
    if run_clicked:
        results = {}
        for level_name, codes, pods, ateco_col in active_levels:
            pods_with_profile = list(set(profile_norm.index) & set(pods))
            if len(pods_with_profile) < 5:
                results[level_name] = {
                    "error": f"Need ≥5 PODs (have {len(pods_with_profile)})"
                }
                continue

            with st.spinner(f"Clustering {level_name} ({len(pods_with_profile)} PODs)..."):
                X_df, k, details, Z, err = run_clustering_for_pods(
                    profile_norm, pods_with_profile, n_clusters=manual_k,
                    profile_month=profile_month
                )
            if err:
                results[level_name] = {"error": err}
            else:
                results[level_name] = {
                    "X_df": X_df, "k": k, "details": details, "Z": Z,
                    "n_pods": len(pods_with_profile),
                    "codes": codes, "ateco_col": ateco_col,
                    "profile_month": profile_month,
                }

        st.session_state["_cl_results"] = results
        st.session_state["_cl_fingerprint"] = sel_fingerprint
        st.session_state["_cl_active_levels"] = active_levels
        st.session_state.pop("_export_zip", None)
        st.session_state.pop("_export_png_errors", None)
        has_cached = True

    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================
    if not has_cached:
        st.info("Configure your selection above and press **▶ Run Clustering**.")
        return

    results = st.session_state["_cl_results"]
    cached_levels = st.session_state.get("_cl_active_levels", active_levels)

    st.markdown("---")
    st.subheader("Clustering Results")

    level_results = [
        (lname, results.get(lname, {}))
        for lname, _, _, _ in cached_levels
        if lname in results
    ]
    n_cols = len(level_results)

    if n_cols == 0:
        st.warning("No results available.")
        return

    cols = st.columns(n_cols) if n_cols > 1 else [st.container()]

    for col_widget, (level_name, res) in zip(cols, level_results):
        with col_widget:
            if "error" in res:
                st.error(f"**{level_name}**: {res['error']}")
                continue

            X_df = res["X_df"]
            k = res["k"]
            codes = res["codes"]
            n_pods = res["n_pods"]
            ateco_col = res.get("ateco_col", f"ATECO_{level_name}")

            codes_str = ", ".join(codes[:4]) + ("..." if len(codes) > 4 else "")
            k_source = "manual" if manual_k else "auto"
            pm = res.get("profile_month", 0)
            period_lbl = MONTH_LABELS[pm] if pm < len(MONTH_LABELS) else "Overall Average"
            st.markdown(
                f"**{level_name}: {codes_str}** — {n_pods} PODs, "
                f"k={k} ({k_source}) | {period_lbl}"
            )

            fig = plot_cluster_profiles(X_df, k, f" | {level_name}")
            st.plotly_chart(fig, use_container_width=True,
                            key=f"frag_prof_{level_name}")

            if len(codes) >= 2:
                with st.expander(f"📊 Cluster Composition by {ateco_col}", expanded=True):
                    fig_comp, comp_table = plot_cluster_composition(
                        X_df, df_base, ateco_col, level_name
                    )
                    st.plotly_chart(fig_comp, use_container_width=True,
                                    key=f"frag_comp_{level_name}")
                    st.dataframe(comp_table, hide_index=True,
                                 use_container_width=True,
                                 key=f"frag_comptbl_{level_name}")

            parent_cols = []
            if level_name == "L2" and "ATECO_L1" in df_base.columns:
                parent_cols = [("ATECO_L1", "L1 Section")]
            elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                parent_cols = [("ATECO_L1", "L1 Section"),
                               ("ATECO_L2", "L2 Division")]

            for par_col, par_label in parent_cols:
                with st.expander(f"📊 Cluster Composition by {par_label}", expanded=False):
                    fig_par, tbl_par = plot_cluster_composition(
                        X_df, df_base, par_col, f"{level_name} → {par_label}"
                    )
                    st.plotly_chart(fig_par, use_container_width=True,
                                    key=f"frag_comp_{level_name}_{par_col}")
                    st.dataframe(tbl_par, hide_index=True,
                                 use_container_width=True,
                                 key=f"frag_comptbl_{level_name}_{par_col}")

            with st.expander("📋 ATECO → Cluster Breakdown", expanded=False):
                bkd = build_ateco_cluster_breakdown(X_df, df_base, ateco_col, level_name)
                st.dataframe(bkd, hide_index=True, use_container_width=True,
                             key=f"frag_bkd_{level_name}")
                bkd_parents = []
                if level_name == "L2" and "ATECO_L1" in df_base.columns:
                    bkd_parents = [("ATECO_L1", "L1 Section")]
                elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                    bkd_parents = [("ATECO_L1", "L1 Section"),
                                   ("ATECO_L2", "L2 Division")]
                for par_c, par_lbl in bkd_parents:
                    st.markdown(f"**Breakdown by {par_lbl}:**")
                    bkd_par = build_ateco_cluster_breakdown(
                        X_df, df_base, par_c, f"{level_name} → {par_lbl}"
                    )
                    st.dataframe(bkd_par, hide_index=True,
                                 use_container_width=True,
                                 key=f"frag_bkd_{level_name}_{par_c}")

            stats_df = compute_cluster_stats(X_df)
            st.dataframe(stats_df, hide_index=True, use_container_width=True,
                         key=f"frag_stats_{level_name}")

            global_metrics = compute_global_metrics(X_df)
            m_cols = st.columns(len(global_metrics))
            for mc, (mname, mval) in zip(m_cols, global_metrics.items()):
                mc.metric(mname, mval, help=METRIC_HELP.get(mname, ""))

            if k >= 2:
                df_corr, df_pval, _, _ = compute_centroid_pearson(X_df)
                fig_corr = plot_pearson_heatmap(df_corr)
                st.plotly_chart(fig_corr, use_container_width=True,
                                key=f"frag_pears_{level_name}")
                with st.expander("ℹ️ How to read Pearson r / P-values"):
                    st.markdown(PEARSON_HELP)
                with st.expander("P-values"):
                    st.dataframe(df_pval.style.format("{:.2e}"),
                                 use_container_width=True)

    # =========================================================================
    # CROSS-LEVEL COMPARISON
    # =========================================================================
    valid_results = {
        lname: res for lname, res in results.items()
        if "error" not in res
    }

    if len(valid_results) >= 2:
        st.markdown("---")
        st.subheader("Cross-Level Comparison")

        all_centroids = {}
        for level_name, res in valid_results.items():
            X_df = res["X_df"]
            q_cols = [c for c in X_df.columns if c.startswith("Q")]
            for cl in sorted(X_df["Cluster"].unique()):
                centroid = X_df[X_df["Cluster"] == cl][q_cols].mean().values
                all_centroids[f"{level_name}_Cl{cl}"] = centroid

        if len(all_centroids) >= 2:
            labels_all = list(all_centroids.keys())
            n_c = len(labels_all)
            cross_corr = np.ones((n_c, n_c))

            for i in range(n_c):
                for j in range(i + 1, n_c):
                    r, _ = pearsonr(all_centroids[labels_all[i]],
                                    all_centroids[labels_all[j]])
                    cross_corr[i, j] = r
                    cross_corr[j, i] = r

            text_matrix = [[f"{cross_corr[i, j]:.3f}"
                            for j in range(n_c)] for i in range(n_c)]

            fig_cross = go.Figure(data=go.Heatmap(
                z=cross_corr.tolist(),
                x=labels_all, y=labels_all,
                text=text_matrix, texttemplate="%{text}",
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                colorbar=dict(title="Pearson r"),
            ))
            fig_cross.update_layout(
                title="Cross-Level Centroid Correlation",
                height=max(400, n_c * 35 + 100),
                margin=dict(t=40, b=20, l=20, r=20),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)),
            )
            st.plotly_chart(fig_cross, use_container_width=True,
                            key="frag_cross_corr")
            with st.expander("ℹ️ How to read Pearson r"):
                st.markdown(PEARSON_HELP)

            fig_overlay = go.Figure()
            colors = COLORS_PLOTLY
            for idx, (lbl, centroid) in enumerate(all_centroids.items()):
                x_labels = Q_TIME_LABELS[:len(centroid)]
                fig_overlay.add_trace(go.Scatter(
                    x=x_labels, y=centroid, mode="lines",
                    name=lbl, line=dict(width=2, color=colors[idx % len(colors)]),
                ))
            fig_overlay.update_layout(
                title="All Cluster Centroids Overlaid",
                xaxis_title="Time of Day",
                yaxis_title="Normalized (0-1)",
                height=450,
                yaxis=dict(range=[-0.05, 1.05]),
                xaxis=dict(dtick=8, tickangle=-45),
                legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig_overlay, use_container_width=True,
                            key="frag_overlay")

    # =========================================================================
    # EXPORT
    # =========================================================================
    st.markdown("---")
    st.subheader("📥 Export Results")

    chart_format = st.radio(
        "Chart export format",
        ["HTML (interactive, fast)", "PNG (static, slow — needs kaleido)"],
        index=0, key="frag_export_fmt", horizontal=True,
        help=(
            "**HTML**: instant export, open in any browser with full "
            "interactivity (zoom, hover, pan). Recommended.\n\n"
            "**PNG**: static images, requires kaleido installed. "
            "Can take 30-60 seconds."
        ),
    )
    use_html = chart_format.startswith("HTML")

    if st.button("Prepare Export ZIP", type="primary", key="frag_export_btn"):
        buf = io.BytesIO()
        png_errors = []

        n_levels = len(valid_results)
        steps_per_level = 6
        total_steps = max(1, n_levels * steps_per_level)
        step = 0
        export_progress = st.progress(0, text="Preparing export...")

        # ------------------------------------------------------------------
        # Lorenzo Giannuzzo: pre-compute cross-level shared artefacts
        # ------------------------------------------------------------------
        # 1. All centroids DataFrame (all levels combined)
        all_centroids_df = build_all_centroids_df(valid_results)

        # 2. POD → cluster mapping for every computed level
        pod_cluster_map = build_pod_cluster_map(valid_results)

        # 3. ATECO dominant-cluster table (all ATECO levels x all computed levels)
        ateco_dominant_df = build_ateco_dominant_cluster_df(valid_results, df_base)

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for level_name, res in valid_results.items():
                X_df = res["X_df"]
                k = res["k"]
                ateco_col = res.get("ateco_col", f"ATECO_{level_name}")

                # --- Excel ---
                step += 1
                export_progress.progress(
                    step / total_steps,
                    text=f"Building Excel for {level_name}..."
                )
                stats_df = compute_cluster_stats(X_df)
                global_m = compute_global_metrics(X_df)
                global_df = pd.DataFrame([global_m])

                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    # --- Standard sheets (unchanged) ---
                    stats_df.to_excel(writer, index=False,
                                      sheet_name="Cluster_Stats")
                    global_df.to_excel(writer, index=False,
                                       sheet_name="Global_Metrics")
                    if k >= 2:
                        df_corr, df_pval, _, _ = compute_centroid_pearson(X_df)
                        df_corr.to_excel(writer, sheet_name="Pearson_Corr")
                        df_pval.to_excel(writer, sheet_name="Pearson_PValues")

                    pod_cl = X_df[["Cluster"]].reset_index()
                    pod_cl.columns = ["POD", "Cluster"]
                    meta_cols = ["POD", "CCATETE", "D_49DES", "POTCONTR_kW",
                                 "ATECO_L1", "ATECO_L2", "ATECO_L3"]
                    avail_cols = [c for c in meta_cols if c in df_unique.columns]
                    pod_cl = pod_cl.merge(
                        df_unique[avail_cols].drop_duplicates(subset=["POD"]),
                        on="POD", how="left"
                    )
                    pod_cl.to_excel(writer, index=False, sheet_name="POD_List")

                    if len(res.get("codes", [])) >= 2:
                        _, comp_tbl = plot_cluster_composition(
                            X_df, df_base, ateco_col, level_name
                        )
                        comp_tbl.to_excel(writer, index=False,
                                          sheet_name="Cluster_Composition")

                    par_cols_exp = []
                    if level_name == "L2" and "ATECO_L1" in df_base.columns:
                        par_cols_exp = [("ATECO_L1", "Composition_by_L1")]
                    elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                        par_cols_exp = [("ATECO_L1", "Composition_by_L1"),
                                        ("ATECO_L2", "Composition_by_L2")]
                    for par_c, sheet_name in par_cols_exp:
                        _, par_tbl = plot_cluster_composition(
                            X_df, df_base, par_c, level_name
                        )
                        par_tbl.to_excel(writer, index=False,
                                         sheet_name=sheet_name)

                    # -------------------------------------------------------
                    # Lorenzo Giannuzzo: Profiles_with_Cluster — now includes
                    # cluster_L1, cluster_L2, cluster_L3 columns for all
                    # computed levels in addition to own "Cluster" column.
                    # -------------------------------------------------------
                    profiles_export = X_df.copy()
                    profiles_export.index.name = "POD"
                    profiles_export = profiles_export.reset_index()
                    q_exp = [c for c in profiles_export.columns if c.startswith("Q")]

                    # Rename own Cluster column to Cluster_<level>
                    profiles_export = profiles_export.rename(
                        columns={"Cluster": f"Cluster_{level_name}"}
                    )

                    # Add cluster columns from the other levels
                    for other_level, other_mapping in pod_cluster_map.items():
                        if other_level == level_name:
                            continue
                        col_name = f"Cluster_{other_level}"
                        profiles_export[col_name] = profiles_export["POD"].map(
                            other_mapping
                        )
                        # Convert to nullable int so missing = <NA> not NaN float
                        profiles_export[col_name] = pd.array(
                            profiles_export[col_name], dtype=pd.Int64Dtype()
                        )

                    # Build ordered column list: POD, all Cluster_Lx cols, metadata, Q
                    cluster_cols_ordered = sorted(
                        [c for c in profiles_export.columns if c.startswith("Cluster_")],
                        key=lambda x: x  # L1 < L2 < L3 alphabetically
                    )
                    meta_merge = ["CCATETE", "D_49DES", "POTCONTR_kW",
                                  "ATECO_L1", "ATECO_L2", "ATECO_L3"]
                    avail_meta = [c for c in meta_merge if c in df_unique.columns]
                    if avail_meta:
                        profiles_export = profiles_export.merge(
                            df_unique[["POD"] + avail_meta]
                            .drop_duplicates(subset=["POD"]),
                            on="POD", how="left"
                        )
                    final_cols = (
                        ["POD"] + cluster_cols_ordered + avail_meta + q_exp
                    )
                    # Keep only columns that exist
                    final_cols = [c for c in final_cols if c in profiles_export.columns]
                    profiles_export = profiles_export[final_cols]
                    profiles_export.to_excel(
                        writer, index=False, sheet_name="Profiles_with_Cluster"
                    )

                    # -------------------------------------------------------
                    # Lorenzo Giannuzzo: All_Centroids sheet — centroids of
                    # ALL levels (L1, L2, L3) in every per-level Excel file.
                    # -------------------------------------------------------
                    if not all_centroids_df.empty:
                        all_centroids_df.to_excel(
                            writer, index=False, sheet_name="All_Centroids"
                        )

                    # -------------------------------------------------------
                    # Lorenzo Giannuzzo: ATECO_Dominant_Cluster sheet — for
                    # every ATECO code, the dominant cluster at each level.
                    # -------------------------------------------------------
                    if not ateco_dominant_df.empty:
                        ateco_dominant_df.to_excel(
                            writer, index=False,
                            sheet_name="ATECO_Dominant_Cluster"
                        )

                    # ATECO → Cluster breakdown tables
                    bkd_own = build_ateco_cluster_breakdown(
                        X_df, df_base, ateco_col, level_name
                    )
                    bkd_own.to_excel(writer, index=False,
                                     sheet_name="ATECO_Breakdown")
                    bkd_par_exp = []
                    if level_name == "L2" and "ATECO_L1" in df_base.columns:
                        bkd_par_exp = [("ATECO_L1", "Breakdown_by_L1")]
                    elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                        bkd_par_exp = [("ATECO_L1", "Breakdown_by_L1"),
                                       ("ATECO_L2", "Breakdown_by_L2")]
                    for bp_col, bp_sheet in bkd_par_exp:
                        bp_df = build_ateco_cluster_breakdown(
                            X_df, df_base, bp_col, level_name
                        )
                        bp_df.to_excel(writer, index=False, sheet_name=bp_sheet)

                zf.writestr(f"{level_name}_k{k}_stats.xlsx", excel_buf.getvalue())

                # --- Profile chart ---
                step += 1
                export_progress.progress(
                    step / total_steps,
                    text=f"Exporting profile chart for {level_name}..."
                )
                try:
                    fig = plot_cluster_profiles(X_df, k, f" | {level_name}")
                    if use_html:
                        zf.writestr(f"{level_name}_k{k}_profiles.html",
                                    fig.to_html(include_plotlyjs="cdn"))
                    else:
                        zf.writestr(f"{level_name}_k{k}_profiles.png",
                                    fig_to_png_bytes(fig, width=1600, height=800))
                except Exception as e:
                    png_errors.append(f"{level_name} profiles: {e}")

                # --- Pearson chart ---
                step += 1
                export_progress.progress(
                    step / total_steps,
                    text=f"Exporting Pearson heatmap for {level_name}..."
                )
                if k >= 2:
                    try:
                        df_corr, _, _, _ = compute_centroid_pearson(X_df)
                        fig_h = plot_pearson_heatmap(df_corr)
                        if use_html:
                            zf.writestr(f"{level_name}_k{k}_pearson.html",
                                        fig_h.to_html(include_plotlyjs="cdn"))
                        else:
                            zf.writestr(f"{level_name}_k{k}_pearson.png",
                                        fig_to_png_bytes(fig_h, width=800, height=600))
                    except Exception as e:
                        png_errors.append(f"{level_name} pearson: {e}")

                # --- Composition chart ---
                step += 1
                export_progress.progress(
                    step / total_steps,
                    text=f"Exporting composition chart for {level_name}..."
                )
                if len(res.get("codes", [])) >= 2:
                    try:
                        fig_c, _ = plot_cluster_composition(
                            X_df, df_base, ateco_col, level_name
                        )
                        if use_html:
                            zf.writestr(f"{level_name}_k{k}_composition.html",
                                        fig_c.to_html(include_plotlyjs="cdn"))
                        else:
                            zf.writestr(f"{level_name}_k{k}_composition.png",
                                        fig_to_png_bytes(fig_c, width=1200, height=600))
                    except Exception as e:
                        png_errors.append(f"{level_name} composition: {e}")

                # --- Stats table + metrics as HTML ---
                step += 1
                export_progress.progress(
                    min(step / total_steps, 1.0),
                    text=f"Exporting stats table for {level_name}..."
                )
                try:
                    stats_df = compute_cluster_stats(X_df)
                    global_m = compute_global_metrics(X_df)
                    stats_html = (
                        "<html><head>"
                        "<style>"
                        "body{font-family:Arial,sans-serif;padding:20px;}"
                        "table{border-collapse:collapse;width:100%;margin-bottom:20px;}"
                        "th,td{border:1px solid #ddd;padding:8px;text-align:center;}"
                        "th{background:#f2f2f2;font-weight:bold;}"
                        "tr:nth-child(even){background:#fafafa;}"
                        "h2{color:#333;}"
                        ".metrics{display:flex;gap:30px;margin:20px 0;}"
                        ".metric{text-align:center;}"
                        ".metric .value{font-size:28px;font-weight:bold;}"
                        ".metric .label{font-size:12px;color:#666;}"
                        "</style></head><body>"
                        f"<h2>Clustering Results — {level_name} (k={k})</h2>"
                        '<div class="metrics">'
                    )
                    for mname, mval in global_m.items():
                        stats_html += (
                            f'<div class="metric">'
                            f'<div class="value">{mval}</div>'
                            f'<div class="label">{mname}</div></div>'
                        )
                    stats_html += "</div>"
                    stats_html += stats_df.to_html(index=False)
                    stats_html += "</body></html>"
                    zf.writestr(f"{level_name}_k{k}_stats_table.html", stats_html)
                except Exception as e:
                    png_errors.append(f"{level_name} stats table: {e}")

                # --- Parent-level composition charts ---
                step += 1
                export_progress.progress(
                    min(step / total_steps, 1.0),
                    text=f"Exporting parent compositions for {level_name}..."
                )
                par_cols_chart = []
                if level_name == "L2" and "ATECO_L1" in df_base.columns:
                    par_cols_chart = [("ATECO_L1", "by_L1")]
                elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                    par_cols_chart = [("ATECO_L1", "by_L1"),
                                      ("ATECO_L2", "by_L2")]
                for par_c, suffix in par_cols_chart:
                    try:
                        fig_par, _ = plot_cluster_composition(
                            X_df, df_base, par_c, f"{level_name} → {par_c}"
                        )
                        if use_html:
                            zf.writestr(
                                f"{level_name}_k{k}_composition_{suffix}.html",
                                fig_par.to_html(include_plotlyjs="cdn"),
                            )
                        else:
                            zf.writestr(
                                f"{level_name}_k{k}_composition_{suffix}.png",
                                fig_to_png_bytes(fig_par, width=1200, height=600),
                            )
                    except Exception as e:
                        png_errors.append(f"{level_name} composition {suffix}: {e}")

                # --- ATECO breakdown HTML ---
                try:
                    bkd_exp = build_ateco_cluster_breakdown(
                        X_df, df_base, ateco_col, level_name
                    )
                    bkd_html = build_ateco_breakdown_html(bkd_exp, level_name, k)
                    zf.writestr(f"{level_name}_k{k}_ateco_breakdown.html", bkd_html)
                except Exception as e:
                    png_errors.append(f"{level_name} breakdown: {e}")

            if png_errors:
                err_txt = "PNG EXPORT ISSUES\n" + "=" * 40 + "\n"
                err_txt += "\n".join(f"- {e}" for e in png_errors)
                err_txt += "\n\nFix: pip install kaleido==0.2.1"
                zf.writestr("_PNG_EXPORT_ISSUES.txt", err_txt)

        buf.seek(0)
        export_progress.progress(1.0, text="Export ready!")

        st.session_state["_export_zip"] = buf.getvalue()
        st.session_state["_export_png_errors"] = png_errors
        export_progress.empty()

    if "_export_zip" in st.session_state:
        _exp_errors = st.session_state.get("_export_png_errors", [])
        if _exp_errors:
            st.warning(
                f"{len(_exp_errors)} chart export issues (Excel files OK). "
                "If using PNG mode, install kaleido: `pip install kaleido==0.2.1`"
            )
        st.download_button(
            label="⬇️ Download Results ZIP",
            data=st.session_state["_export_zip"],
            file_name="ateco_clustering_results.zip",
            mime="application/zip",
            type="primary",
            key="frag_download_btn",
        )


if __name__ == "__main__":
    main()