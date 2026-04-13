"""
================================================================================
ATECO HIERARCHICAL CLUSTERING DASHBOARD
================================================================================
Author: Lorenzo Giannuzzo - Energy Center Lab, DENERG, Politecnico di Torino
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

# ── Contractual power bins (finer below 3 kW) ────────────────────────────────
POTCONTR_BINS = [0, 1, 1.5, 2, 3, 6, 10, 16.5, 33, 55, 110, float("inf")]
POTCONTR_BIN_LABELS = [
    "≤1 kW", "1–1.5 kW", "1.5–2 kW", "2–3 kW",
    "3–6 kW", "6–10 kW", "10–16.5 kW", "16.5–33 kW",
    "33–55 kW", "55–110 kW", ">110 kW",
]
POTCONTR_COLORS = [
    "#2E86AB", "#5BA4CF", "#44BBA4", "#8ECAE6",
    "#F18F01", "#C73E1D", "#A23B72", "#7B2D8B",
    "#1A936F", "#393E41", "#F5A623",
]

METRIC_HELP = {
    "Silhouette Score": (
        "Measures how similar each POD is to its own cluster vs. other clusters.\n\n"
        "Range: -1 to +1\n"
        "• > 0.70: Strong structure\n• 0.50–0.70: Reasonable\n"
        "• 0.25–0.50: Weak\n• < 0.25: Poor"
    ),
    "Calinski-Harabasz Index": (
        "Ratio of between-cluster to within-cluster dispersion. Higher = better.\n\n"
        "• > 500: Excellent\n• 200–500: Good\n• 50–200: Moderate\n• < 50: Poor"
    ),
    "Davies-Bouldin Index": (
        "Average similarity between each cluster and its most similar one. Lower = better.\n\n"
        "• < 0.5: Excellent\n• 0.5–1.0: Good\n• 1.0–1.5: Moderate\n• > 1.5: Poor"
    ),
}

PEARSON_HELP = (
    "Pearson correlation coefficient (r) between cluster centroids.\n\n"
    "Range: -1 to +1\n"
    "• r ≈ 1.0: Almost identical load shapes\n"
    "• r = 0.7–0.9: Similar overall shape\n"
    "• r = 0.3–0.7: Partially similar\n"
    "• r ≈ 0: Completely unrelated\n"
    "• r < 0: Opposite patterns"
)

DISTRIBUTOR_CODES = {
    "DO": "Domestic (Resident + Non-Resident)",
    "DO.R": "Domestic - Resident",
    "DO.NR": "Domestic - Non-Resident",
    "CO": "Condominium services",
    "CO.01": "Condominium services - Resident", "CO.02": "Condominium services - Non-Resident",
    "IL": "Public lighting", "IL.01": "Public lighting",
    "DA": "Food products, beverages and tobacco", "DB": "Textiles and textile products",
    "DC": "Leather and leather products", "DD": "Wood and wood products",
    "DE": "Pulp, paper, publishing and printing", "DF": "Coke, petroleum products and nuclear fuel",
    "DG": "Chemicals and chemical products", "DH": "Rubber and plastic products",
    "DI": "Other non-metallic mineral products", "DJ": "Basic metals and fabricated metal products",
    "DK": "Machinery and equipment n.e.c.", "DL": "Electrical and optical equipment",
    "DM": "Transport equipment", "DN": "Manufacturing n.e.c.",
    "CA": "Mining of coal and lignite; extraction of peat", "CB": "Mining of metal ores",
}

ATECO_LOOKUP: dict[str, str] = {}
ATECO_EXCEL_NAME = "Note-esplicative-ATECO-2025-italiano-inglese.xlsx"
GSE_FILE_NAME = "profili GSE_prelievo_2025.xlsx"
ARERA_FILE_NAME   = "Copia di dati prelievo orario per provincia potenza6 anno 2024-mkt.xlsx"
ARERA_FILE_NAME_2 = "Copia di dati prelievo orario per provincia0_a_1_5 anno 2024mkt.xlsx"
ARERA_FILE_NAME_3 = "Copia di dati prelievo orario per provincia3_a_4_5 anno 2024mkt.xlsx"
ARERA_FILE_NAME_4 = "Copia di dati prelievo orario per provincia4_5_a_6 anno 2024TOT.xlsx"
ARERA_FILE_NAME_5 = "Copia di dati prelievo orario per provincia1_5_a_3 anno 2024TOT.xlsx"
ARERA_PROVINCE = "Trento"   # default province for ARERA comparison

ARERA_DAYTYPE_MAP = {
    "Giorno feriale": "Weekday",
    "Sabato": "Saturday",
    "Domenica": "Sunday",
}
ARERA_RESIDENZA_MAP = {
    "Residente": "DO.R",
    "Non Residente": "DO.NR",
    "Tutti": "All Domestic",
}

# Maps each ARERA Classe_potenza string to a human-readable label and a
# POTCONTR_kW filter lambda (applied to our df_unique["POTCONTR_kW"]).
ARERA_POWER_CLASSES: list[dict] = [
    {
        "classe": "0<potenza_impegnata<=1.5",
        "label":  "≤ 1.5 kW",
        "file":   ARERA_FILE_NAME_2,
        "filter": lambda kw: (kw > 0) & (kw <= 1.5),
        "markets": ["Maggior Tutela", "Mercato Libero"],
    },
    {
        "classe": "1.5<potenza_impegnata<=3",
        "label":  "1.5–3 kW",
        "file":   ARERA_FILE_NAME_5,
        "filter": lambda kw: (kw > 1.5) & (kw <= 3),
        "markets": ["Tutti"],
    },
    {
        "classe": "3<potenza_impegnata<=4.5",
        "label":  "3–4.5 kW",
        "file":   ARERA_FILE_NAME_3,
        "filter": lambda kw: (kw > 3) & (kw <= 4.5),
        "markets": ["Maggior Tutela", "Mercato Libero"],
    },
    {
        "classe": "4.5<potenza_impegnata<=6",
        "label":  "4.5–6 kW",
        "file":   ARERA_FILE_NAME_4,
        "filter": lambda kw: (kw > 4.5) & (kw <= 6),
        "markets": ["Maggior Tutela", "Mercato Libero"],
    },
    {
        "classe": "potenza_impegnata>6",
        "label":  "> 6 kW",
        "file":   ARERA_FILE_NAME,
        "filter": lambda kw: kw > 6,
        "markets": ["Maggior Tutela", "Mercato Libero"],
    },
]

# GSE profile columns and their labels
GSE_COLS = {
    "PDMM": "GSE Domestic Weekday",
    "PDMF": "GSE Domestic Weekend/Holiday",
    "PAUM": "GSE Other Weekday",
    "PAUF": "GSE Other Weekend/Holiday",
}

MONTH_LABELS = [
    "Overall Average",
    "January (01)", "February (02)", "March (03)", "April (04)",
    "May (05)", "June (06)", "July (07)", "August (08)",
    "September (09)", "October (10)", "November (11)", "December (12)",
]

# ==============================================================================
# UTILITY FUNCTIONS
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


def normalize_fdesc_domestic(fdesc_val) -> str | None:
    """
    Map FDESC strings for domestic users to simplified flat codes:
      DO.NR = Domestic Non-Resident  (USI DOMESTICI NON RESID.)
      DO.R  = Domestic Resident / generic domestic (everything else domestic)
    Returns None if the value is not domestic-related or is missing.
    Domestic users are kept at L1 only — L2 and L3 are always N/A.
    """
    if not fdesc_val or pd.isna(fdesc_val):
        return None
    s = str(fdesc_val).strip().upper()
    if "NON RESID" in s:
        return "DO.NR"
    if "RESID" in s or "DOMESTIC" in s:
        return "DO.R"
    return None


def fig_to_png_bytes(fig, width=1600, height=900, scale=2):
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale, engine="kaleido")
    except Exception as e1:
        try:
            return fig.to_image(format="png", width=width, height=height, scale=scale)
        except Exception:
            raise RuntimeError(f"PNG export failed. Fix: pip install kaleido==0.2.1\nError: {e1}")


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

        meas_path = find_measures_file(d)
        if meas_path:
            try:
                df_ms = pd.read_csv(meas_path, sep=None, engine="python", dtype=str)
                available_q = [c for c in Q_COLS if c in df_ms.columns]
                if "DataMisura" not in df_ms.columns or "POD" not in df_ms.columns:
                    continue

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
                        df_pot["POTCONTR"].astype(str).str.replace(",", ".", regex=False).str.strip(),
                        errors="coerce")
                    df_pot = df_pot.dropna(subset=["POTCONTR"])
                    df_pot = df_pot.drop_duplicates(subset=["POD"], keep="last")
                    if not df_pot.empty:
                        potcontr_frames.append(df_pot)

                cols = ["DataMisura", "POD"] + available_q
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

    # ── Refine domestic classification using FDESC ────────────────────────
    # Hierarchy: L1 = "DO"  (all domestics grouped)
    #            L2 = L3 = "DO.R" (resident) or "DO.NR" (non-resident)
    #
    # Step 1: FDESC-based mapping where FDESC is available.
    if "FDESC" in df_unique.columns:
        fdesc_mapped = df_unique["FDESC"].apply(normalize_fdesc_domestic)
        domestic_mask = fdesc_mapped.notna()
        if domestic_mask.any():
            df_unique.loc[domestic_mask, "ATECO_L1"] = "DO"
            df_unique.loc[domestic_mask, "ATECO_L2"] = fdesc_mapped[domestic_mask]
            df_unique.loc[domestic_mask, "ATECO_L3"] = "N/A"

    # Step 2: fix any leftover "DO*" variants in L1 (old DO.01/DO.02, stale
    # DO.R/DO.NR from cached runs, bare "DO" that parse_ateco produced, etc.)
    # → force L1="DO", L2="DO.R", L3="N/A".
    bad_do_l1 = (
        df_unique["ATECO_L1"].str.match(r"^DO", na=False)
        & (df_unique["ATECO_L1"] != "DO")
    )
    if bad_do_l1.any():
        df_unique.loc[bad_do_l1, "ATECO_L1"] = "DO"
        needs_l2 = bad_do_l1 & ~df_unique["ATECO_L2"].isin(["DO.R", "DO.NR"])
        df_unique.loc[needs_l2, "ATECO_L2"] = "DO.R"
        df_unique.loc[bad_do_l1, "ATECO_L3"] = "N/A"

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

    unstacked = unstacked.interpolate(axis=1, method="linear", limit=4, limit_direction="both")

    for m in range(1, 13):
        prefix = f"M{m:02d}_"
        m_cols = [c for c in unstacked.columns if c.startswith(prefix)]
        if not m_cols:
            continue
        unstacked[m_cols] = unstacked[m_cols].ffill(axis=1).bfill(axis=1)

    row_median = unstacked.median(axis=1)
    for col in unstacked.columns:
        mask = unstacked[col].isna()
        if mask.any():
            unstacked.loc[mask, col] = row_median[mask]
    unstacked = unstacked.fillna(0)

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
        has_range = m_range > 0
        normalized = m_data.sub(m_min, axis=0).div(m_range.where(has_range, other=1.0), axis=0)
        normalized.loc[~has_range] = 0.0
        profile_norm[m_cols] = normalized

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
    prefix = f"M{month_number:02d}_"
    cols = [c for c in profile_df.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame()
    sub = profile_df[cols].copy()
    sub.columns = [c.replace(prefix, "") for c in cols]
    return sub.dropna()


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


def run_clustering_for_pods(profile_norm, pod_list, n_clusters=None, profile_month=0):
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
    q_cols = [c for c in X_df.columns if c.startswith("Q")]
    X = X_df[q_cols].values
    labels = X_df["Cluster"].values

    metrics = {}
    try:
        metrics["Silhouette Score"] = round(silhouette_score(X, labels), 2)
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
# EXPORT HELPERS
# ==============================================================================

def build_all_centroids_df(valid_results: dict) -> pd.DataFrame:
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
    mapping = {}
    for level_name, res in valid_results.items():
        if "error" in res:
            continue
        X_df = res["X_df"]
        mapping[level_name] = X_df["Cluster"].to_dict()
    return mapping


def build_ateco_dominant_cluster_df(valid_results: dict, df_base: pd.DataFrame) -> pd.DataFrame:
    pod_cluster = build_pod_cluster_map(valid_results)
    if not pod_cluster:
        return pd.DataFrame()

    ateco_level_cols = [c for c in ["ATECO_L1", "ATECO_L2", "ATECO_L3"] if c in df_base.columns]
    computed_levels = list(pod_cluster.keys())

    all_rows = []
    for ateco_col in ateco_level_cols:
        ateco_label = ateco_col
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
                clustered_pods = {p: mapping[p] for p in pods_in_group if p in mapping}
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
    df_out = df_out.sort_values(["ATECO_Level", "Total_PODs"],
                                ascending=[True, False]).reset_index(drop=True)
    return df_out


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_cluster_profiles(X_df, k, title_suffix=""):
    """Plot average daily load profiles per cluster with ±1σ bands."""
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
            y=list((mean_p + std_p).clip(0, 1)) + list((mean_p - std_p).clip(0, 1))[::-1],
            fill="toself", fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False, name=f"±1σ Cl.{cl}",
        ))

    fig.update_layout(
        title=f"Daily Load Profiles — mean ±1σ (k={k}){title_suffix}",
        xaxis_title="Time of Day",
        yaxis_title="Normalized (0-1)",
        height=400,
        yaxis=dict(range=[-0.05, 1.05], gridcolor="#1e3a6b"),
        xaxis=dict(dtick=4, tickangle=-45, tickfont=dict(size=9, color="#e8f4fd"),
                   gridcolor="#1e3a6b", title_font=dict(color="#e8f4fd")),
        margin=dict(t=40, b=60, l=50, r=20),
        legend=dict(font=dict(size=9, color="#e8f4fd")),
        plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c",
        font=dict(family="Arial, sans-serif", color="#e8f4fd"),
    )
    return fig


def plot_centroids_only(X_df, k, title_suffix=""):
    """Clean centroid-only overlay chart — no ±1σ bands, thicker lines."""
    q_cols = [c for c in X_df.columns if c.startswith("Q")]
    fig = go.Figure()
    clusters_sorted = sorted(X_df["Cluster"].unique())
    x_labels = Q_TIME_LABELS[:len(q_cols)]

    for idx, cl in enumerate(clusters_sorted):
        cl_data = X_df[X_df["Cluster"] == cl][q_cols]
        mean_p = cl_data.mean()
        n = len(cl_data)
        color = COLORS_PLOTLY[idx % len(COLORS_PLOTLY)]
        fig.add_trace(go.Scatter(
            x=x_labels, y=mean_p, mode="lines+markers",
            name=f"Cl.{cl} (n={n})",
            line=dict(width=3, color=color),
            marker=dict(size=3, color=color),
        ))

    fig.update_layout(
        title=f"Cluster Centroids — Clean Overlay (k={k}){title_suffix}",
        xaxis_title="Time of Day",
        yaxis_title="Normalized (0-1)",
        height=300,
        yaxis=dict(range=[-0.05, 1.05], gridcolor="#1e3a6b",
                   title_font=dict(color="#e8f4fd"), tickfont=dict(size=9, color="#e8f4fd")),
        xaxis=dict(dtick=4, tickangle=-45, tickfont=dict(size=9, color="#e8f4fd"),
                   gridcolor="#1e3a6b", title_font=dict(color="#e8f4fd")),
        plot_bgcolor="#0d2144",
        paper_bgcolor="#0d1f3c",
        margin=dict(t=40, b=55, l=55, r=20),
        legend=dict(font=dict(size=9, color="#e8f4fd"), orientation="h", y=-0.30, x=0),
        font=dict(family="Arial, sans-serif", color="#e8f4fd"),
    )
    return fig


def plot_pearson_heatmap(df_corr):
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
        xaxis=dict(side="bottom", tickfont=dict(color="#e8f4fd")),
        yaxis=dict(tickfont=dict(color="#e8f4fd")),
        plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c",
        font=dict(family="Arial, sans-serif", color="#e8f4fd"),
    )
    return fig


# ==============================================================================
# CLUSTER COMPOSITION CHARTS
# ==============================================================================

def plot_cluster_composition(X_df, df_base, ateco_col, level_label):
    """Stacked horizontal bar: ATECO composition within each cluster."""
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
        legend=dict(font=dict(size=9, color="#e8f4fd")),
        yaxis=dict(automargin=True, tickfont=dict(color="#e8f4fd"),
                   gridcolor="#1e3a6b"),
        xaxis=dict(tickangle=0, tickfont=dict(color="#e8f4fd"),
                   title_font=dict(color="#e8f4fd"), gridcolor="#1e3a6b"),
        plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c",
        font=dict(family="Arial, sans-serif", color="#e8f4fd"),
    )
    fig.update_traces(
        textposition="inside", textfont_size=9,
        insidetextanchor="middle", textangle=0,
    )
    summary = comp[["ClusterLabel", "AtecoLabel", "Count", "Pct"]].copy()
    summary.columns = ["Cluster", "ATECO", "N. PODs", "%"]
    return fig, summary


def build_ateco_cluster_breakdown(X_df, df_base, ateco_col, level_label):
    """Pivot: rows = ATECO codes, columns = clusters (count + %)."""
    pod_cl = X_df[["Cluster"]].reset_index()
    pod_cl.columns = ["POD", "Cluster"]
    merged = pod_cl.merge(
        df_base[["POD", ateco_col]].drop_duplicates("POD"),
        on="POD", how="left"
    )
    merged["AtecoDesc"] = merged[ateco_col].apply(
        lambda c: lookup_ateco_description(c)[:50] if lookup_ateco_description(c) else ""
    )
    cross = merged.groupby([ateco_col, "AtecoDesc", "Cluster"]).size().reset_index(name="Count")
    ateco_totals = merged.groupby(ateco_col).size().reset_index(name="Total")
    cross = cross.merge(ateco_totals, on=ateco_col)
    cross["Pct"] = (cross["Count"] / cross["Total"] * 100).round(1)
    cross["Label"] = cross.apply(lambda r: f"{int(r['Count'])} ({r['Pct']:.1f}%)", axis=1)
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


def build_cluster_ateco_breakdown(X_df, df_base, ateco_col, level_label):
    """Inverted pivot: rows = clusters, columns = ATECO codes (count + %)."""
    pod_cl = X_df[["Cluster"]].reset_index()
    pod_cl.columns = ["POD", "Cluster"]
    merged = pod_cl.merge(
        df_base[["POD", ateco_col]].drop_duplicates("POD"),
        on="POD", how="left"
    )
    merged["AtecoDesc"] = merged[ateco_col].apply(
        lambda c: lookup_ateco_description(c)[:45] if lookup_ateco_description(c) else ""
    )
    merged["AtecoLabel"] = merged.apply(
        lambda r: f"{r[ateco_col]} — {r['AtecoDesc']}" if r["AtecoDesc"] else str(r[ateco_col]),
        axis=1
    )

    cross = merged.groupby(["Cluster", "AtecoLabel"]).size().reset_index(name="Count")
    cluster_totals = merged.groupby("Cluster").size().reset_index(name="Total PODs")
    cross = cross.merge(cluster_totals, on="Cluster")
    cross["Pct"] = (cross["Count"] / cross["Total PODs"] * 100).round(1)
    cross["CellLabel"] = cross.apply(
        lambda r: f"{int(r['Count'])} ({r['Pct']:.1f}%)", axis=1
    )

    pivot = cross.pivot_table(
        index=["Cluster", "Total PODs"],
        columns="AtecoLabel",
        values="CellLabel",
        aggfunc="first",
        fill_value="—"
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.sort_values("Cluster").reset_index(drop=True)

    pivot.insert(0, "Cluster Label",
                 pivot["Cluster"].apply(lambda c: f"Cluster {c}"))

    ateco_cols = [c for c in pivot.columns
                  if c not in ("Cluster Label", "Cluster", "Total PODs")]

    col_counts = {}
    for ac in ateco_cols:
        total = 0
        for _, row in cross[cross["AtecoLabel"] == ac].iterrows():
            total += row["Count"]
        col_counts[ac] = total
    ateco_cols_sorted = sorted(ateco_cols, key=lambda c: col_counts.get(c, 0), reverse=True)

    final_cols = ["Cluster Label", "Cluster", "Total PODs"] + ateco_cols_sorted
    final_cols = [c for c in final_cols if c in pivot.columns]
    pivot = pivot[final_cols].drop(columns=["Cluster"])

    return pivot


def build_ateco_breakdown_html(pivot_df, level_label, k, centroid_chart_html=""):
    """HTML export for ATECO → Cluster breakdown, with centroid chart on the RIGHT."""
    chart_section = ""
    if centroid_chart_html:
        chart_section = (
            f'<div style="min-width:620px;max-width:700px;flex-shrink:0;">'
            f"<h3>Cluster Centroids — {level_label} (k={k})</h3>"
            + centroid_chart_html +
            "</div>"
        )
        wrapper_open = '<div style="display:flex;gap:30px;align-items:flex-start;">'
        table_wrap_open = '<div style="flex:1;overflow-x:auto;">'
        table_wrap_close = "</div>"
        wrapper_close = "</div>"
    else:
        wrapper_open = ""
        table_wrap_open = ""
        table_wrap_close = ""
        wrapper_close = ""

    html = (
        "<html><head><style>"
        "body{font-family:Arial,sans-serif;padding:20px;}"
        "h2,h3{color:#333;}"
        "table{border-collapse:collapse;width:100%;}"
        "th{background:#2c3e50;color:white;padding:8px 10px;text-align:center;font-size:12px;}"
        "td{border:1px solid #ddd;padding:6px 10px;text-align:center;font-size:11px;}"
        "td:first-child,td:nth-child(2){text-align:left;}"
        "tr:nth-child(even){background:#f9f9f9;}"
        "tr:hover{background:#e8f4fd;}"
        "</style></head><body>"
        f"<h2>ATECO → Cluster Breakdown — {level_label} (k={k})</h2>"
        "<p>Each cell: count (% of that ATECO code's PODs in each cluster)</p>"
        + wrapper_open
        + table_wrap_open
    )
    html += pivot_df.to_html(index=False, escape=False, na_rep="—")
    html += table_wrap_close + chart_section + wrapper_close
    html += "</body></html>"
    return html


def build_cluster_ateco_breakdown_html(pivot_df, level_label, k, centroid_chart_html=""):
    """HTML export for Cluster → ATECO breakdown, with centroid chart BELOW."""
    html = (
        "<html><head><style>"
        "body{font-family:Arial,sans-serif;padding:20px;}"
        "h2,h3{color:#333;}"
        "table{border-collapse:collapse;width:100%;}"
        "th{background:#2c3e50;color:white;padding:8px 10px;text-align:center;font-size:12px;}"
        "td{border:1px solid #ddd;padding:6px 10px;text-align:center;font-size:11px;}"
        "td:first-child{text-align:left;font-weight:bold;}"
        "tr:nth-child(even){background:#f9f9f9;}"
        "tr:hover{background:#e8f4fd;}"
        ".centroid-section{margin-top:40px;}"
        "</style></head><body>"
        f"<h2>Cluster → ATECO Breakdown — {level_label} (k={k})</h2>"
        "<p>Each cell: count (% of that cluster's total PODs belonging to the ATECO code)</p>"
    )
    html += pivot_df.to_html(index=False, escape=False, na_rep="—")
    if centroid_chart_html:
        html += (
            '<div class="centroid-section">'
            f"<h3>Cluster Centroids — {level_label} (k={k})</h3>"
            + centroid_chart_html +
            "</div>"
        )
    html += "</body></html>"
    return html


# ==============================================================================
# MONTHLY CONSUMPTION DISTRIBUTION
# ==============================================================================

def compute_monthly_consumption_per_pod(df_meas):
    available_q = [c for c in Q_COLS if c in df_meas.columns]
    if not available_q or "DataMisura" not in df_meas.columns:
        return pd.DataFrame()

    df = df_meas[["POD", "DataMisura"] + available_q].copy()
    df["YearMonth"] = df["DataMisura"].dt.to_period("M").astype(str)
    df["Daily_kWh"] = df[available_q].sum(axis=1) / 1000
    monthly = (
        df.groupby(["POD", "YearMonth"])["Daily_kWh"]
        .sum().reset_index().rename(columns={"Daily_kWh": "Monthly_kWh"})
    )
    return monthly


def plot_consumption_distribution_top15(df_meas, df_unique, title_suffix=""):
    monthly = compute_monthly_consumption_per_pod(df_meas)
    if monthly.empty:
        return None, None

    meta_cols = ["POD", "CCATETE", "D_49DES", "FDESC", "TATE3DES",
                 "ATECO_L1", "ATECO_L2", "ATECO_L3"]
    avail_meta = [c for c in meta_cols if c in df_unique.columns]
    merged = monthly.merge(df_unique[avail_meta].drop_duplicates("POD"), on="POD", how="left")

    def make_label(row):
        for col in ["ATECO_L3", "ATECO_L2", "ATECO_L1", "CCATETE"]:
            code = row.get(col, None)
            if code and str(code) not in ("", "N/A", "nan"):
                desc = lookup_ateco_description(str(code))
                if desc:
                    return f"{code} | {desc[:45]}"
                return str(code)
        for col in ["D_49DES", "TATE3DES", "FDESC"]:
            val = row.get(col, None)
            if val and str(val) not in ("", "nan"):
                return str(val)[:55]
        return "Unknown"

    merged["Label"] = merged.apply(make_label, axis=1)
    merged = merged[merged["Label"] != "Unknown"]

    top15_labels = (
        merged.groupby("Label")["POD"].nunique()
        .sort_values(ascending=False).head(15).index.tolist()
    )

    df_top = merged[merged["Label"].isin(top15_labels)].copy()
    df_top = df_top[df_top["Monthly_kWh"] > 0]

    label_order = (
        df_top.groupby("Label")["POD"].nunique()
        .sort_values(ascending=True).index.tolist()
    )

    PROF_COLORS = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
        "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#7B2D8B",
        "#1A936F", "#C6483C", "#4059AD", "#6B4226", "#2CA58D",
    ]

    fig = go.Figure()
    for idx, label in enumerate(label_order):
        vals = df_top[df_top["Label"] == label]["Monthly_kWh"].dropna().values
        if len(vals) == 0:
            continue
        color = PROF_COLORS[idx % len(PROF_COLORS)]
        fig.add_trace(go.Box(
            x=vals, name=label, orientation="h",
            marker=dict(color=color, size=3, opacity=0.5),
            line=dict(color=color, width=1.8),
            fillcolor=color, opacity=0.75,
            boxmean=False, boxpoints=False, whiskerwidth=0.5, showlegend=False,
        ))

    n_pods_map = df_top.groupby("Label")["POD"].nunique()
    annotations = []
    for label in label_order:
        n = n_pods_map.get(label, 0)
        annotations.append(dict(
            x=0, y=label, xref="x", yref="y", text=f"n={n}",
            showarrow=False, xanchor="right", xshift=-6,
            font=dict(size=9, color="#e8f4fd"),
        ))

    n_total_top = merged[merged["Label"].isin(top15_labels)]["POD"].nunique()
    fig.update_layout(
        title=dict(
            text=f"Monthly Consumption Distribution — Top 15 User Typologies<br>"
                 f"<sup>{n_total_top:,} PODs | {title_suffix}</sup>",
            font=dict(size=14, color="#e8f4fd"), x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Monthly Consumption [kWh]",
            title_font=dict(size=11, color="#e8f4fd"),
            tickfont=dict(size=9, color="#e8f4fd"),
            gridcolor="#1e3a6b", showgrid=True,
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=9, color="#e8f4fd"),
            automargin=True,
        ),
        plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c",
        height=max(500, len(label_order) * 45 + 160),
        margin=dict(t=80, b=60, l=280, r=40),
        annotations=annotations,
        font=dict(family="Arial, sans-serif", color="#e8f4fd"),
    )

    summary_rows = []
    for label in reversed(label_order):
        vals = df_top[df_top["Label"] == label]["Monthly_kWh"].dropna()
        if len(vals) == 0:
            continue
        summary_rows.append({
            "Typology": label, "N PODs": int(n_pods_map.get(label, 0)),
            "Median [kWh]": round(float(vals.median()), 1),
            "Mean [kWh]": round(float(vals.mean()), 1),
            "P25 [kWh]": round(float(vals.quantile(0.25)), 1),
            "P75 [kWh]": round(float(vals.quantile(0.75)), 1),
            "Max [kWh]": round(float(vals.max()), 1),
        })
    return fig, pd.DataFrame(summary_rows)


def plot_potcontr_pie(df_unique):
    if "POTCONTR_kW" not in df_unique.columns:
        return None
    df = df_unique.dropna(subset=["POTCONTR_kW"]).copy()
    if df.empty:
        return None

    df["Power_Range"] = pd.cut(
        df["POTCONTR_kW"], bins=POTCONTR_BINS, labels=POTCONTR_BIN_LABELS, right=True
    )
    counts = (
        df.groupby("Power_Range", observed=True)["POD"].nunique()
        .reset_index().rename(columns={"POD": "N_PODs"})
    )
    counts = counts[counts["N_PODs"] > 0]

    total = counts["N_PODs"].sum()
    # Only label slices >= 5% — smaller ones are already in the legend
    custom_text = [
        f"{row['Power_Range']}<br>{row['N_PODs']/total*100:.1f}%"
        if (row["N_PODs"] / total) >= 0.05 else ""
        for _, row in counts.iterrows()
    ]

    fig = go.Figure(go.Pie(
        labels=counts["Power_Range"].astype(str),
        values=counts["N_PODs"],
        hole=0.38,
        marker=dict(
            colors=POTCONTR_COLORS[:len(counts)],
            line=dict(color="#ffffff", width=2),
        ),
        text=custom_text,
        textinfo="text",
        textfont=dict(size=12, color="#e8f4fd"),
        textposition="inside",
        insidetextorientation="horizontal",
        sort=False,
        automargin=False,
    ))
    fig.update_layout(
        title=dict(
            text=f"Contractual Power Distribution<br>"
                 f"<sup>{df['POD'].nunique():,} PODs with power data</sup>",
            font=dict(size=14, color="#e8f4fd"), x=0.5, xanchor="center",
        ),
        legend=dict(
            orientation="v", x=1.02, y=0.5,
            font=dict(size=10, color="#e8f4fd"),
        ),
        plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c", height=480,
        margin=dict(t=80, b=60, l=80, r=180),
        font=dict(family="Arial, sans-serif", color="#e8f4fd"),
    )
    return fig


def plot_potcontr_stacked_bar(df_unique, top_n=10):
    if "POTCONTR_kW" not in df_unique.columns:
        return None
    df = df_unique.dropna(subset=["POTCONTR_kW"]).copy()
    if df.empty:
        return None

    def make_label(row):
        for col in ["ATECO_L3", "ATECO_L2", "ATECO_L1", "CCATETE"]:
            code = row.get(col, None)
            if code and str(code) not in ("", "N/A", "nan"):
                desc = lookup_ateco_description(str(code))
                if desc:
                    return f"{code} | {desc[:40]}"
                return str(code)
        for col in ["D_49DES", "TATE3DES", "FDESC"]:
            val = row.get(col, None)
            if val and str(val) not in ("", "nan"):
                return str(val)[:50]
        return "Unknown"

    df["Label"] = df.apply(make_label, axis=1)
    df["Power_Range"] = pd.cut(
        df["POTCONTR_kW"], bins=POTCONTR_BINS, labels=POTCONTR_BIN_LABELS, right=True
    )

    top_labels = (df.groupby("Label")["POD"].nunique()
                  .sort_values(ascending=False).head(top_n).index.tolist())
    df_top = df[df["Label"].isin(top_labels)]

    pivot = (df_top.groupby(["Label", "Power_Range"], observed=True)["POD"]
             .nunique().unstack(fill_value=0))
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=True).drop(columns="_total")

    fig = go.Figure()
    for idx, power_range in enumerate(POTCONTR_BIN_LABELS):
        if power_range not in pivot.columns:
            continue
        vals = pivot[power_range].values
        row_labels = pivot.index.tolist()
        totals = pivot.sum(axis=1).values
        text_vals = []
        for v, tot in zip(vals, totals):
            if tot > 0 and v / tot > 0.04:
                text_vals.append(str(int(v)))
            else:
                text_vals.append("")
        fig.add_trace(go.Bar(
            y=row_labels, x=vals, name=power_range, orientation="h",
            marker=dict(
                color=POTCONTR_COLORS[idx % len(POTCONTR_COLORS)],
                line=dict(color="#ffffff", width=0.8),
            ),
            text=text_vals, textposition="inside", textangle=0,
            insidetextanchor="middle", textfont=dict(size=9, color="#ffffff"),
        ))

    n_total = df_top["POD"].nunique()
    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"Contractual Power Distribution by Typology — Top {top_n}<br>"
                 f"<sup>{n_total:,} PODs with power data</sup>",
            font=dict(size=14, color="#e8f4fd"), x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Number of PODs",
            title_font=dict(size=11, color="#e8f4fd"),
            tickfont=dict(size=9, color="#e8f4fd"),
            gridcolor="#1e3a6b", showgrid=True,
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=10, color="#e8f4fd"),
            automargin=True, tickangle=0,
        ),
        legend=dict(
            title="Contractual Power",
            font=dict(size=10, color="#e8f4fd"),
            orientation="v", x=1.01, y=0.5,
        ),
        plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c",
        height=max(480, top_n * 50 + 160),
        margin=dict(t=80, b=60, l=320, r=160),
        font=dict(family="Arial, sans-serif", color="#e8f4fd"),
    )
    return fig


# ==============================================================================
# GSE PROFILE LOADING & COMPARISON
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_gse_profiles() -> pd.DataFrame | None:
    """Load profili_GSE_prelievo_2025.xlsx, parse % columns to float."""
    path = DATA_DIR / GSE_FILE_NAME
    if not path.exists():
        return None
    try:
        df = pd.read_excel(path, dtype=str)
        # Normalise column names
        df.columns = [c.strip() for c in df.columns]
        for col in ["Anno", "Mese", "Giorno", "Ora"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Parse percentage columns: "0,10%" → 0.10
        pct_cols = [c for c in df.columns if c in GSE_COLS or c in
                    ["PIRM", "PIRF", "PACM", "PACF", "MDMM", "MDMF", "MAUM", "MAUF"]]
        for col in pct_cols:
            raw = df[col].astype(str).str.strip()
            has_pct_sign = raw.str.contains("%", na=False).any()
            numeric = (
                raw
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
            if has_pct_sign:
                # String like "0,10%" → 0.10 (already a percent, keep as-is)
                df[col] = numeric
            else:
                # Excel percentage format: stored as 0.001 → multiply by 100
                df[col] = numeric * 100
        return df.dropna(subset=["Mese", "Ora"])
    except Exception as e:
        st.error(f"Error loading GSE file: {e}")
        return None


def compute_gse_monthly_hourly(df_gse: pd.DataFrame, col: str) -> dict[int, np.ndarray]:
    """Average GSE column by (Month, Hour-of-day) → dict month→array[24]."""
    result = {}
    for month, grp in df_gse.groupby("Mese"):
        hourly = grp.groupby("Ora")[col].mean().reindex(range(24)).values
        result[int(month)] = hourly
    return result


def get_fascia(dayofweek: int, hour: int) -> str:
    """
    Italian ARERA electricity time-of-use band.
    dayofweek: 0=Mon … 6=Sun.  hour: 0-23.
    F1 = peak (Mon-Fri 08:00-19:00)
    F2 = mid  (Mon-Fri 07:00-08:00 & 19:00-23:00; Sat 07:00-23:00)
    F3 = off-peak (nights, Sundays; public holidays approximated as F3)
    """
    if dayofweek == 6:                       # Sunday → F3
        return "F3"
    if dayofweek == 5:                       # Saturday
        return "F2" if 7 <= hour <= 22 else "F3"
    # Monday-Friday
    if 8 <= hour <= 18:                      # 08:00-19:00
        return "F1"
    if hour == 7 or 19 <= hour <= 22:        # 07:00-08:00 & 19:00-23:00
        return "F2"
    return "F3"                              # 00:00-07:00 & 23:00-24:00


def compute_our_fascia_profiles(
    df_meas: pd.DataFrame,
    pod_set: set,
) -> dict[int, np.ndarray]:
    """
    Fascia-normalised hourly profile: each hour expressed as % of its
    Italian TOU band (F1/F2/F3) monthly total — same normalisation as
    GSE 'F' columns (PDMF, PAUF).  Sum of all 24 hours ≈ 300%.
    Fully vectorised.
    """
    available_q = [c for c in Q_COLS if c in df_meas.columns]
    if not available_q or not pod_set:
        return {}

    df = df_meas[df_meas["POD"].astype(str).isin(pod_set)].copy()
    if df.empty:
        return {}

    df["_month"] = df["DataMisura"].dt.month
    df["_dow"]   = df["DataMisura"].dt.dayofweek

    # Build 24 hourly columns
    h_cols = []
    for h in range(24):
        q_h = [f"Q{h*4+j+1}" for j in range(4) if f"Q{h*4+j+1}" in available_q]
        col = f"_H{h}"
        df[col] = df[q_h].sum(axis=1) if q_h else 0.0
        h_cols.append(col)

    # Per-row daily contribution to each fascia band
    for fascia in ["F1", "F2", "F3"]:
        df[f"_{fascia}_day"] = 0.0

    for dow in range(7):
        mask_dow = df["_dow"] == dow
        if not mask_dow.any():
            continue
        for h in range(24):
            f = get_fascia(dow, h)
            df.loc[mask_dow, f"_{f}_day"] += df.loc[mask_dow, f"_H{h}"]

    # Monthly fascia totals per POD
    pod_month_fascia = (
        df.groupby(["POD", "_month"], observed=True)[["_F1_day", "_F2_day", "_F3_day"]]
        .sum()
        .rename(columns={"_F1_day": "_F1_month",
                         "_F2_day": "_F2_month",
                         "_F3_day": "_F3_month"})
    )
    df = df.join(pod_month_fascia, on=["POD", "_month"])

    # For each hour h: pct_h = H_h / fascia_monthly(fascia(row._dow, h)) * 100
    hpct_cols = []
    for h in range(24):
        fascia_of_h_by_dow = {dow: get_fascia(dow, h) for dow in range(7)}
        fascia_series = df["_dow"].map(fascia_of_h_by_dow)
        denom = np.where(
            fascia_series == "F1", df["_F1_month"].values,
            np.where(fascia_series == "F2", df["_F2_month"].values,
                     df["_F3_month"].values)
        )
        col = f"_Hpct{h}"
        df[col] = np.where(denom > 0, df[f"_H{h}"].values / denom * 100, np.nan)
        hpct_cols.append(col)

    # Average over days → (POD, month), then over PODs → month
    pct_reset = (
        df.groupby(["POD", "_month"], observed=True)[hpct_cols]
        .mean()
        .reset_index()
    )
    result = {}
    for month, grp_m in pct_reset.groupby("_month"):
        result[int(month)] = grp_m[hpct_cols].mean().values

    return result


def compute_our_normalized_profiles(
    df_meas: pd.DataFrame,
    pod_set: set,
) -> dict[int, np.ndarray]:
    """
    For a given set of PODs, compute for each month the average hourly profile
    as % of monthly consumption (same unit as GSE profiles).

    Fully vectorised — no Python loops over PODs.
    Returns dict month (1-12) → np.ndarray[24] of mean % per hour.
    """
    available_q = [c for c in Q_COLS if c in df_meas.columns]
    if not available_q or not pod_set:
        return {}

    df = df_meas[df_meas["POD"].astype(str).isin(pod_set)].copy()
    if df.empty:
        return {}

    df["_month"] = df["DataMisura"].dt.month

    # Build 24 hourly columns by summing 4 quarter-hour columns each
    h_cols = []
    for h in range(24):
        q_h = [f"Q{h * 4 + j + 1}" for j in range(4) if f"Q{h * 4 + j + 1}" in available_q]
        col = f"_H{h}"
        df[col] = df[q_h].sum(axis=1) if q_h else 0.0
        h_cols.append(col)

    # Group by (POD, month): mean over days for each hour col
    grp = df.groupby(["POD", "_month"], observed=True)[h_cols]
    hourly_means = grp.mean()          # avg daily energy per hour (per POD-month)
    n_days = grp.size()                # number of days per POD-month

    # Monthly total = avg_daily_total * n_days
    daily_totals = hourly_means.sum(axis=1)          # avg daily total energy
    monthly_totals = daily_totals * n_days            # actual monthly total

    # Remove POD-months with zero consumption
    valid = monthly_totals > 0
    hourly_means = hourly_means[valid]
    monthly_totals = monthly_totals[valid]

    # Normalise: each hour as % of MONTHLY total → same scale as GSE
    # (GSE: ~0.1–0.5% per hour, 24*n_days values sum to 100%)
    pct = hourly_means.div(monthly_totals, axis=0) * 100

    # Average over PODs per month
    pct_reset = pct.reset_index()
    result = {}
    for month, grp_m in pct_reset.groupby("_month"):
        result[int(month)] = grp_m[h_cols].mean().values

    return result


def plot_gse_comparison_month(
    month: int,
    our_profiles: dict[str, dict[int, np.ndarray]],
    gse_profiles: dict[str, dict[int, np.ndarray]],
    group_label: str,
) -> go.Figure:
    """Single month comparison chart: our profile vs GSE references."""
    hours = list(range(24))
    hour_labels = [f"{h:02d}:00" for h in hours]

    COLORS = {
        "our": "#1f77b4",
        "gse_wd": "#ff7f0e",
        "gse_we": "#2ca02c",
    }

    fig = go.Figure()

    # Our profile
    our = our_profiles.get(month)
    if our is not None:
        fig.add_trace(go.Scatter(
            x=hour_labels, y=our,
            mode="lines", name="PoliTo (avg)",
            line=dict(width=2.5, color=COLORS["our"]),
        ))

    # GSE weekday
    gse_wd = gse_profiles.get("weekday", {}).get(month)
    if gse_wd is not None:
        fig.add_trace(go.Scatter(
            x=hour_labels, y=gse_wd,
            mode="lines", name="GSE Weekday",
            line=dict(width=2, color=COLORS["gse_wd"], dash="dash"),
        ))

    # GSE weekend
    gse_we = gse_profiles.get("weekend", {}).get(month)
    if gse_we is not None:
        fig.add_trace(go.Scatter(
            x=hour_labels, y=gse_we,
            mode="lines", name="GSE Weekend",
            line=dict(width=2, color=COLORS["gse_we"], dash="dot"),
        ))

    fig.update_layout(
        title=dict(text=MONTH_NAMES[month], font=dict(size=12, color="#e8f4fd")),
        xaxis=dict(tickfont=dict(size=8, color="#e8f4fd"), tickangle=-45,
                   dtick=3, gridcolor="#1e3a6b"),
        yaxis=dict(title="% monthly", tickfont=dict(size=8, color="#e8f4fd"),
                   gridcolor="#1e3a6b"),
        height=280,
        margin=dict(t=35, b=55, l=55, r=10),
        legend=dict(font=dict(size=8), orientation="h", y=-0.35, x=0),
        plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c",
        font=dict(family="Arial, sans-serif", color="#e8f4fd"),
    )
    return fig


def compute_comparison_metrics(
    our: np.ndarray | None,
    gse_wd: np.ndarray | None,
    gse_we: np.ndarray | None,
    col_wd: str = "GSE Weekday",
    col_we: str = "GSE Weekend",
) -> dict:
    """Pearson r, RMSE, MAE between our profile and GSE weekday/weekend.
    Profiles are in % units, so RMSE/MAE are in percentage points (pp).
    """
    metrics = {}
    for label, ref in [(f"vs {col_wd}", gse_wd), (f"vs {col_we}", gse_we)]:
        if our is None or ref is None:
            metrics[label] = {"Pearson r": "—", "RMSE (%)": "—", "MAE (%)": "—"}
            continue
        mask = ~(np.isnan(our) | np.isnan(ref))
        if mask.sum() < 2:
            metrics[label] = {"Pearson r": "—", "RMSE (%)": "—", "MAE (%)": "—"}
            continue
        r, _ = pearsonr(our[mask], ref[mask])
        rmse = np.sqrt(np.mean((our[mask] - ref[mask]) ** 2))
        mae = np.mean(np.abs(our[mask] - ref[mask]))
        metrics[label] = {
            "Pearson r": round(float(r), 4),
            "RMSE (%)": round(float(rmse), 4),
            "MAE (%)":  round(float(mae), 4),
        }
    return metrics


def gse_comparison_tab(df_base: pd.DataFrame, df_meas: pd.DataFrame, pods_12m: set):
    """Full GSE comparison tab content."""
    st.subheader("GSE Profile Comparison")
    st.markdown(
        "Comparison of monthly average hourly profiles between PoliTo dataset "
        "and GSE 2025 reference profiles. Values are expressed as **% of monthly "
        "consumption per hour** (same normalisation as GSE). "
        "Only PODs with ≥12 months of data are included."
    )

    # ── Load GSE ──────────────────────────────────────────────────────────────
    gse_path = DATA_DIR / GSE_FILE_NAME
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption(f"Looking for: `{gse_path}`  —  exists: `{gse_path.exists()}`")
    with col_btn:
        if st.button("🔄 Reload GSE cache", key="reload_gse_btn"):
            load_gse_profiles.clear()
            st.session_state.pop("_gse_our_profiles", None)
            st.rerun()

    df_gse = load_gse_profiles()
    if df_gse is None:
        st.error(
            f"GSE file `data/{GSE_FILE_NAME}` not found. "
            "Place it in the `data/` folder and reload."
        )
        return

    gse_domestic_wd = compute_gse_monthly_hourly(df_gse, "PDMM")
    gse_domestic_we = compute_gse_monthly_hourly(df_gse, "PDMF")
    gse_other_wd    = compute_gse_monthly_hourly(df_gse, "PAUM")
    gse_other_we    = compute_gse_monthly_hourly(df_gse, "PAUF")

    # ── POD groups — use df_base which already reflects all global sidebar filters ──
    # df_base is pre-filtered by: tipologia, 12m coverage, power filter, meas_pods
    # No need to re-intersect with pods_12m (already done upstream)
    pods_dor  = set(df_base[df_base["ATECO_L2"] == "DO.R"]["POD"].unique())
    pods_donr = set(df_base[df_base["ATECO_L2"] == "DO.NR"]["POD"].unique())
    pods_other = set(
        df_base[
            ~df_base["ATECO_L1"].str.upper().str.startswith("DO", na=False)
            & ~df_base["ATECO_L1"].str.upper().str.startswith("IL", na=False)
            & (df_base["ATECO_L1"] != "N/A")
        ]["POD"].unique()
    )

    # df_meas is already tipologia-filtered from the sidebar. Keep ap_pods intersection
    # for robustness (in case user selects "All" tipologia but we still want AP for profiles).
    if "Tipologia" in df_meas.columns and df_meas["Tipologia"].notna().any():
        ap_vals = [v for v in df_meas["Tipologia"].dropna().unique()
                   if str(v).strip().upper() == "AP"]
        df_meas_ap = df_meas[df_meas["Tipologia"].isin(ap_vals)].copy() if ap_vals else df_meas.copy()
    else:
        df_meas_ap = df_meas.copy()

    ap_pods = set(df_meas_ap["POD"].astype(str).unique())
    pods_dor   = pods_dor   & ap_pods
    pods_donr  = pods_donr  & ap_pods
    pods_other = pods_other & ap_pods

    st.caption(
        f"Groups (consistent with global filters) — "
        f"DO.R: {len(pods_dor):,} PODs | "
        f"DO.NR: {len(pods_donr):,} PODs | "
        f"Other (excl. IL): {len(pods_other):,} PODs"
    )

    # ── Compute our profiles (cached) ─────────────────────────────────────────
    cache_key = "_gse_our_profiles"
    if cache_key not in st.session_state:
        with st.spinner("Computing normalised hourly profiles (AP only)…"):
            st.session_state[cache_key] = {
                "DO.R": {
                    "avgM": compute_our_normalized_profiles(df_meas_ap, pods_dor),
                    "avgF": compute_our_fascia_profiles(df_meas_ap, pods_dor),
                },
                "DO.NR": {
                    "avgM": compute_our_normalized_profiles(df_meas_ap, pods_donr),
                    "avgF": compute_our_fascia_profiles(df_meas_ap, pods_donr),
                },
                "Other": {
                    "avgM": compute_our_normalized_profiles(df_meas_ap, pods_other),
                    "avgF": compute_our_fascia_profiles(df_meas_ap, pods_other),
                },
            }
    our_profiles = st.session_state[cache_key]

    # ── Group tabs ────────────────────────────────────────────────────────────
    group_tabs = st.tabs([
        "Domestic Resident (DO.R)",
        "Domestic Non-Resident (DO.NR)",
        "Other (excl. Public Lighting)",
    ])

    groups_cfg = [
        ("DO.R",  "DO.R",
         {"weekday": gse_domestic_wd, "weekend": gse_domestic_we},
         {"weekday": "PDMM",          "weekend": "PDMF"}),
        ("DO.NR", "DO.NR",
         {"weekday": gse_domestic_wd, "weekend": gse_domestic_we},
         {"weekday": "PDMM",          "weekend": "PDMF"}),
        ("Other", "Other",
         {"weekday": gse_other_wd,    "weekend": gse_other_we},
         {"weekday": "PAUM",          "weekend": "PAUF"}),
    ]

    gse_export_data: dict[str, dict] = {}

    for tab_widget, (group_key, group_label, gse_ref, gse_col_names) in zip(group_tabs, groups_cfg):
        with tab_widget:
            our = our_profiles.get(group_key, {})
            our_avgM = our.get("avgM", {})
            our_avgF = our.get("avgF", {})
            if not our_avgM and not our_avgF:
                st.warning(f"No data available for group **{group_label}**.")
                continue

            st.markdown("#### Monthly Average Hourly Profiles")
            all_metric_rows = []
            group_figs: dict[int, go.Figure] = {}

            for row_start in range(0, 12, 3):
                cols = st.columns(3)
                for col_idx, month in enumerate(range(row_start + 1, row_start + 4)):
                    with cols[col_idx]:
                        our_m_arr  = our_avgM.get(month)
                        our_f_arr  = our_avgF.get(month)
                        gse_wd_arr = gse_ref["weekday"].get(month)
                        gse_we_arr = gse_ref["weekend"].get(month)
                        hour_labels = [f"{h:02d}:00" for h in range(24)]

                        fig2 = go.Figure()
                        if our_m_arr is not None:
                            fig2.add_trace(go.Scatter(
                                x=hour_labels, y=our_m_arr, mode="lines+markers",
                                name="PoliTo (avgM)",
                                line=dict(width=3, color="#1f77b4"),
                                marker=dict(size=3, color="#1f77b4"),
                            ))
                        if our_f_arr is not None:
                            fig2.add_trace(go.Scatter(
                                x=hour_labels, y=our_f_arr, mode="lines+markers",
                                name="PoliTo (avgF)",
                                line=dict(width=3, color="#9467bd"),
                                marker=dict(size=3, color="#9467bd"),
                            ))
                        if gse_wd_arr is not None:
                            fig2.add_trace(go.Scatter(
                                x=hour_labels, y=gse_wd_arr, mode="lines+markers",
                                name=gse_col_names["weekday"],
                                line=dict(width=3, color="#ff7f0e", dash="dash"),
                                marker=dict(size=3, color="#ff7f0e"),
                            ))
                        if gse_we_arr is not None:
                            fig2.add_trace(go.Scatter(
                                x=hour_labels, y=gse_we_arr, mode="lines+markers",
                                name=gse_col_names["weekend"],
                                line=dict(width=3, color="#2ca02c", dash="dot"),
                                marker=dict(size=3, color="#2ca02c"),
                            ))

                        fig2.update_layout(
                            title=dict(text=MONTH_NAMES[month], font=dict(size=12, color="#e8f4fd")),
                            xaxis=dict(
                                tickfont=dict(size=8, color="#e8f4fd"),
                                title_font=dict(size=8, color="#e8f4fd"),
                                tickangle=-45, dtick=3,
                                gridcolor="#1e3a6b", showgrid=True,
                                zeroline=False,
                            ),
                            yaxis=dict(
                                title="Average Daily Consumption [%]",
                                title_font=dict(size=8, color="#e8f4fd"),
                                tickfont=dict(size=8, color="#e8f4fd"),
                                gridcolor="#1e3a6b", showgrid=True,
                                zeroline=False,
                                rangemode="tozero",
                            ),
                            height=320,
                            margin=dict(t=35, b=100, l=60, r=10),
                            legend=dict(
                                font=dict(size=8, color="#e8f4fd"),
                                orientation="h", y=-0.55, x=0,
                                bgcolor="rgba(0,0,0,0)",
                            ),
                            plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c",
                            font=dict(family="Arial, sans-serif", color="#e8f4fd"),
                            showlegend=True,
                        )
                        st.plotly_chart(fig2, use_container_width=True,
                                        key=f"gse_{group_key}_m{month}",
                                        config={"toImageButtonOptions": {"format": "png", "scale": 4, "width": 1600, "height": 900}})
                        group_figs[month] = fig2

                        # Correct pairing: avgM ↔ M-normalised GSE column (PDMM/PAUM)
                        #                  avgF ↔ F-normalised GSE column (PDMF/PAUF)
                        for polito_label, polito_arr, ref_arr, ref_col in [
                            ("PoliTo(avgM)", our_m_arr, gse_wd_arr, gse_col_names["weekday"]),
                            ("PoliTo(avgF)", our_f_arr, gse_we_arr, gse_col_names["weekend"]),
                        ]:
                            if polito_arr is None or ref_arr is None:
                                continue
                            mask = ~(np.isnan(polito_arr) | np.isnan(ref_arr))
                            if mask.sum() < 2:
                                continue
                            r, _ = pearsonr(polito_arr[mask], ref_arr[mask])
                            rmse = float(np.sqrt(np.mean((polito_arr[mask] - ref_arr[mask]) ** 2)))
                            mae  = float(np.mean(np.abs(polito_arr[mask] - ref_arr[mask])))
                            all_metric_rows.append({
                                "Month":          MONTH_NAMES[month],
                                "PoliTo Profile": polito_label,
                                "GSE Reference":  ref_col,
                                "Pearson r":      round(float(r), 4),
                                "RMSE (pp)":      round(rmse * 100, 4),
                                "MAE (pp)":       round(mae * 100, 4),
                            })

            st.markdown("#### Similarity Metrics")
            st.caption(
                "Correct pairing: **PoliTo avgM ↔ GSE M-column** (PDMM/PAUM) | "
                "**PoliTo avgF ↔ GSE F-column** (PDMF/PAUF). "
                "RMSE and MAE in percentage points (pp)."
            )
            df_metrics = pd.DataFrame(all_metric_rows) if all_metric_rows else pd.DataFrame()
            if not df_metrics.empty:
                st.dataframe(df_metrics, hide_index=True, use_container_width=True)
                # Excel download for metrics
                _xl_met = io.BytesIO()
                with pd.ExcelWriter(_xl_met, engine="openpyxl") as _w:
                    df_metrics.to_excel(_w, index=False, sheet_name="Similarity_Metrics")
                _xl_met.seek(0)
                st.download_button(
                    label="⬇️ Download Metrics Excel",
                    data=_xl_met.getvalue(),
                    file_name=f"GSE_similarity_metrics_{group_label}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"gse_metrics_dl_{group_key}",
                )

            gse_export_data[group_key] = {
                "label": group_label,
                "figs": group_figs,
                "metrics_df": df_metrics,
                "gse_col_names": gse_col_names,
            }

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Export GSE Comparison")
    st.caption("Exports all monthly charts as interactive HTML and metrics as Excel.")

    if st.button("Prepare GSE Export ZIP", type="primary", key="gse_export_btn"):
        gse_buf = io.BytesIO()
        gse_errors = []
        total_gse_steps = max(1, len(gse_export_data) * 13 + len(gse_export_data))
        gse_step = 0
        gse_prog = st.progress(0, text="Building GSE export…")

        with zipfile.ZipFile(gse_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for group_key, exp in gse_export_data.items():
                grp_label  = exp["label"]
                grp_figs   = exp["figs"]
                metrics_df = exp["metrics_df"]
                col_names  = exp["gse_col_names"]
                safe_label = grp_label.replace(".", "_").replace(" ", "_")

                for month, fig in grp_figs.items():
                    gse_step += 1
                    gse_prog.progress(gse_step / total_gse_steps,
                                      text=f"Chart {MONTH_NAMES[month]} — {grp_label}…")
                    try:
                        zf.writestr(f"{safe_label}_{MONTH_NAMES[month]}.html",
                                    fig.to_html(include_plotlyjs="cdn"))
                    except Exception as e:
                        gse_errors.append(f"{grp_label} {MONTH_NAMES[month]}: {e}")

                # Overview HTML
                try:
                    overview_html = (
                        "<html><head><style>"
                        "body{font-family:Arial,sans-serif;padding:20px;background:#fff;}"
                        "h1{color:#333;}.grid{display:grid;"
                        "grid-template-columns:repeat(3,1fr);gap:16px;margin-top:20px;}"
                        "</style></head><body>"
                        f"<h1>GSE Profile Comparison — {grp_label}</h1>"
                        f"<p>PoliTo (avgM) vs PoliTo (avgF) vs "
                        f"{col_names['weekday']} vs {col_names['weekend']}</p>"
                        '<div class="grid">'
                    )
                    for i, (month, fig) in enumerate(grp_figs.items()):
                        overview_html += fig.to_html(
                            include_plotlyjs="cdn" if i == 0 else False,
                            full_html=False,
                        )
                    overview_html += "</div></body></html>"
                    zf.writestr(f"{safe_label}_all_months_overview.html", overview_html)
                except Exception as e:
                    gse_errors.append(f"{grp_label} overview: {e}")

                # Excel
                gse_step += 1
                gse_prog.progress(gse_step / total_gse_steps, text=f"Excel — {grp_label}…")
                try:
                    xl_buf = io.BytesIO()
                    our_data = our_profiles.get(group_key, {})
                    hour_labels = [f"{h:02d}:00" for h in range(24)]
                    # Retrieve the GSE ref for this group from groups_cfg
                    gse_ref_exp = next(
                        (ref for gk, gl, ref, cn in groups_cfg if gk == group_key),
                        {"weekday": {}, "weekend": {}}
                    )
                    gse_cols_exp = next(
                        (cn for gk, gl, ref, cn in groups_cfg if gk == group_key),
                        {"weekday": "GSE_WD", "weekend": "GSE_WE"}
                    )
                    with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                        if not metrics_df.empty:
                            metrics_df.to_excel(writer, index=False, sheet_name="Similarity_Metrics")
                        rows_prof = []
                        for h_idx, hl in enumerate(hour_labels):
                            row = {"Hour": hl}
                            for month in range(1, 13):
                                mn = MONTH_NAMES[month]
                                avgM = our_data.get("avgM", {}).get(month)
                                avgF = our_data.get("avgF", {}).get(month)
                                gse_wd = gse_ref_exp["weekday"].get(month)
                                gse_we = gse_ref_exp["weekend"].get(month)
                                row[f"PoliTo_avgM_{mn}"]                    = float(avgM[h_idx]) if avgM is not None else None
                                row[f"PoliTo_avgF_{mn}"]                    = float(avgF[h_idx]) if avgF is not None else None
                                row[f"{gse_cols_exp['weekday']}_{mn}"]     = float(gse_wd[h_idx]) if gse_wd is not None else None
                                row[f"{gse_cols_exp['weekend']}_{mn}"]     = float(gse_we[h_idx]) if gse_we is not None else None
                            rows_prof.append(row)
                        pd.DataFrame(rows_prof).to_excel(writer, index=False, sheet_name="Profile_Data")
                    zf.writestr(f"{safe_label}_metrics_and_profiles.xlsx", xl_buf.getvalue())
                except Exception as e:
                    gse_errors.append(f"{grp_label} Excel: {e}")

            if gse_errors:
                zf.writestr("_GSE_EXPORT_ISSUES.txt",
                            "GSE EXPORT ISSUES\n" + "=" * 40 + "\n"
                            + "\n".join(f"- {e}" for e in gse_errors))

        gse_buf.seek(0)
        gse_prog.progress(1.0, text="GSE export ready!")
        gse_prog.empty()
        st.session_state["_gse_export_zip"] = gse_buf.getvalue()
        st.session_state["_gse_export_errors"] = gse_errors

    if "_gse_export_zip" in st.session_state:
        if st.session_state.get("_gse_export_errors"):
            st.warning(f"{len(st.session_state['_gse_export_errors'])} issues. "
                       "See _GSE_EXPORT_ISSUES.txt in the ZIP.")
        st.download_button(
            label="⬇️ Download GSE Comparison ZIP",
            data=st.session_state["_gse_export_zip"],
            file_name="gse_comparison_export.zip",
            mime="application/zip",
            type="primary",
            key="gse_download_btn",
        )


# ==============================================================================
# ARERA PROFILE LOADING & COMPARISON
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_arera_profiles(province: str = ARERA_PROVINCE, filename: str = ARERA_FILE_NAME) -> pd.DataFrame | None:
    """Load one ARERA hourly withdrawal file, filter to province."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        if "Residenza " in df.columns:
            df = df.rename(columns={"Residenza ": "Residenza"})
        def parse_anno_mese(v):
            if isinstance(v, int):
                return 0
            if hasattr(v, "month"):
                return v.month
            return 0
        df["_mese"] = df["Anno Mese"].apply(parse_anno_mese)
        df = df[df["Provincia"] == province].copy()
        df["_hour"] = df["Orario"].str.replace("Ora", "", regex=False).astype(int) - 1
        kw_col = "Prelievo medio Orario Provinciale (kWh)"
        df = df[["_mese", "Tipo mercato", "Classe_potenza", "Residenza", "Working Day", "_hour", kw_col]].copy()
        df = df.rename(columns={kw_col: "kWh"})
        return df
    except Exception as e:
        st.error(f"Error loading ARERA file '{filename}': {e}")
        return None


def load_arera_for_power_class(pc: dict, province: str = ARERA_PROVINCE) -> pd.DataFrame | None:
    """Load ARERA data for a specific power class entry."""
    return load_arera_profiles(province=province, filename=pc["file"])


def compute_arera_hourly_kwh(
    df_arera: pd.DataFrame,
    tipo_mercato: str,
    residenza: str,
) -> dict[str, dict[int, np.ndarray]]:
    """
    Return mean hourly kWh profiles for a given mercato × residenza.
    dict: day_type_en → {month (0=annual, 1-12) → array[24] kWh}.
    """
    sub = df_arera[
        (df_arera["Tipo mercato"] == tipo_mercato)
        & (df_arera["Residenza"] == residenza)
    ].copy()
    if sub.empty:
        return {}

    result: dict[str, dict[int, np.ndarray]] = {}
    for wd_it, wd_en in ARERA_DAYTYPE_MAP.items():
        sub_wd = sub[sub["Working Day"] == wd_it]
        month_dict: dict[int, np.ndarray] = {}
        for mese, grp in sub_wd.groupby("_mese"):
            hourly = grp.sort_values("_hour")["kWh"].values.astype(float)
            if len(hourly) == 24:
                month_dict[int(mese)] = hourly
        if month_dict:
            result[wd_en] = month_dict
    return result


def compute_our_hourly_kwh_by_daytype(
    df_meas: pd.DataFrame,
    pod_set: set,
) -> dict[str, dict[int, np.ndarray]]:
    """
    PoliTo mean hourly kWh profiles split by day type (Weekday/Saturday/Sunday).
    Returns dict: day_type_en → {month (0=annual, 1-12) → array[24] kWh}.
    Values are average daily kWh per hour across all PODs.
    """
    available_q = [c for c in Q_COLS if c in df_meas.columns]
    if not available_q or not pod_set:
        return {}

    df = df_meas[df_meas["POD"].astype(str).isin(pod_set)].copy()
    if df.empty:
        return {}

    df["_month"] = df["DataMisura"].dt.month
    df["_dow"]   = df["DataMisura"].dt.dayofweek

    h_cols = []
    for h in range(24):
        q_h = [f"Q{h*4+j+1}" for j in range(4) if f"Q{h*4+j+1}" in available_q]
        col = f"_H{h}"
        # divide by 1000 to get kWh (Q values are in Wh per 15min → sum 4 = Wh/h → /1000 = kWh)
        df[col] = df[q_h].sum(axis=1) / 1000.0 if q_h else 0.0
        h_cols.append(col)

    day_masks = {
        "Weekday":  df["_dow"] < 5,
        "Saturday": df["_dow"] == 5,
        "Sunday":   df["_dow"] == 6,
    }

    result: dict[str, dict[int, np.ndarray]] = {}
    for daytype, mask in day_masks.items():
        sub = df[mask]
        if sub.empty:
            continue
        # Mean per POD per month → mean across PODs
        hmeans = sub.groupby(["POD", "_month"], observed=True)[h_cols].mean()
        month_dict: dict[int, np.ndarray] = {}
        pct_r = hmeans.reset_index()
        for month, grp_m in pct_r.groupby("_month"):
            month_dict[int(month)] = grp_m[h_cols].mean().values
        if month_dict:
            month_dict[0] = np.mean(list(month_dict.values()), axis=0)
        result[daytype] = month_dict
    return result


def _arera_daytype_chart(
    month: int,
    month_label: str,
    daytype_en: str,
    our_kwh: np.ndarray | None,
    arera_mt_kwh: np.ndarray | None,   # first market (MT or Tutti)
    arera_ml_kwh: np.ndarray | None,   # second market (ML) or None
    show_legend: bool = False,
    market_labels: tuple = ("Maggior Tutela", "Mercato Libero"),
) -> go.Figure:
    """One comparison chart for a single (month, day_type) pair."""
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    lbl_mt = market_labels[0] if market_labels else "ARERA"
    lbl_ml = market_labels[1] if len(market_labels) > 1 else None
    fig = go.Figure()

    if our_kwh is not None:
        fig.add_trace(go.Scatter(
            x=hour_labels, y=our_kwh, mode="lines+markers",
            name="PoliTo (avg)",
            line=dict(width=3, color="#1f77b4"),
            marker=dict(size=3, color="#1f77b4"),
        ))
    if arera_mt_kwh is not None:
        fig.add_trace(go.Scatter(
            x=hour_labels, y=arera_mt_kwh, mode="lines+markers",
            name=f"ARERA {lbl_mt}",
            line=dict(width=3, color="#ff7f0e", dash="dash"),
            marker=dict(size=3, color="#ff7f0e"),
        ))
    if arera_ml_kwh is not None and lbl_ml is not None:
        fig.add_trace(go.Scatter(
            x=hour_labels, y=arera_ml_kwh, mode="lines+markers",
            name=f"ARERA {lbl_ml}",
            line=dict(width=3, color="#2ca02c", dash="dot"),
            marker=dict(size=3, color="#2ca02c"),
        ))

    fig.update_layout(
        title=dict(
            text=f"{daytype_en} — {month_label}",
            font=dict(size=11, color="#0d1f3c"),
        ),
        xaxis=dict(
            tickfont=dict(size=8, color="#1a3a6b"), tickangle=-45,
            dtick=3, gridcolor="#d0dff0", showgrid=True,
            zeroline=False,
            title_font=dict(size=8, color="#1a3a6b"),
            linecolor="#1a3a6b",
        ),
        yaxis=dict(
            title="Average kWh / hour",
            title_font=dict(size=8, color="#1a3a6b"),
            tickfont=dict(size=8, color="#1a3a6b"),
            gridcolor="#d0dff0", showgrid=True,
            zeroline=False,
        ),
        height=300,
        margin=dict(t=38, b=65, l=65, r=10),
        legend=dict(font=dict(size=8, color="#0d1f3c"), orientation="h", y=-0.48, x=0),
        plot_bgcolor="#f0f5fc", paper_bgcolor="#ffffff",
        font=dict(family="Arial, sans-serif", color="#0d1f3c"),
        showlegend=show_legend,
    )
    return fig


def arera_comparison_tab(
    df_base: pd.DataFrame,
    df_meas: pd.DataFrame,
    pods_12m: set,
):
    """ARERA profile comparison tab — kWh, two markets, per day type, per power class."""
    st.subheader("ARERA Profile Comparison")
    st.markdown(
        f"Comparison of mean hourly load profiles [**kWh/h**] between PoliTo dataset "
        f"and ARERA 2024 reference data for **{ARERA_PROVINCE}**. "
        f"ARERA profiles are shown for **Maggior Tutela** and **Mercato Libero** separately. "
        f"PoliTo profiles are split by **residenza** and **contractual power class**. "
        f"Only PODs with ≥12 months of AP data are included."
    )

    # ── Reload button ─────────────────────────────────────────────────────────
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        files_status = " | ".join(
            f"`{pc['label']}`: {'✅' if (DATA_DIR / pc['file']).exists() else '❌'}"
            for pc in ARERA_POWER_CLASSES
        )
        st.caption(f"Files: {files_status}")
    with col_btn:
        if st.button("🔄 Reload ARERA cache", key="reload_arera_btn"):
            load_arera_profiles.clear()
            # Clear all per-power-class profile caches
            for pc in ARERA_POWER_CLASSES:
                st.session_state.pop(f"_arera_our_profiles_{pc['label']}", None)
            st.rerun()

    # ── AP filter ─────────────────────────────────────────────────────────────
    if "Tipologia" in df_meas.columns and df_meas["Tipologia"].notna().any():
        ap_vals = [v for v in df_meas["Tipologia"].dropna().unique()
                   if str(v).strip().upper() == "AP"]
        df_meas_ap = df_meas[df_meas["Tipologia"].isin(ap_vals)].copy() if ap_vals else df_meas.copy()
    else:
        df_meas_ap = df_meas.copy()
    ap_pods = set(df_meas_ap["POD"].astype(str).unique())

    # ── Power class dropdown ──────────────────────────────────────────────────
    power_labels = [pc["label"] for pc in ARERA_POWER_CLASSES]
    sel_power_label = st.selectbox(
        "Contractual Power Class",
        power_labels,
        index=0,
        key="arera_power_sel",
        help="Filters both ARERA reference profiles and PoliTo PODs by contractual power.",
    )
    sel_pc = next(pc for pc in ARERA_POWER_CLASSES if pc["label"] == sel_power_label)

    # ── Month selector ────────────────────────────────────────────────────────
    MONTH_KEYS = [0] + list(range(1, 13))
    MONTH_LBLS = ["Annual Average"] + [MONTH_NAMES[m] for m in range(1, 13)]
    sel_month_lbl = st.selectbox(
        "Month", MONTH_LBLS, index=0, key="arera_month_sel"
    )
    sel_month = MONTH_KEYS[MONTH_LBLS.index(sel_month_lbl)]

    # ── Load ARERA for selected power class ───────────────────────────────────
    df_arera = load_arera_for_power_class(sel_pc, ARERA_PROVINCE)
    if df_arera is None:
        st.error(
            f"ARERA file for **{sel_power_label}** (`{sel_pc['file']}`) "
            f"not found in `data/`. Place it there and reload."
        )
        return

    # ── Build POD sets filtered by power class ────────────────────────────────
    # df_base already reflects all global sidebar filters (12m, tipologia, meas_pods)
    has_potcontr = "POTCONTR_kW" in df_base.columns
    if has_potcontr:
        kw_series = df_base["POTCONTR_kW"].fillna(0)
        power_mask = sel_pc["filter"](kw_series)
        pods_power = set(df_base[power_mask]["POD"].unique())
    else:
        pods_power = set(df_base["POD"].unique())
        st.warning("No POTCONTR_kW data — showing all PODs regardless of power class.")

    pods_dor = (
        set(df_base[df_base["ATECO_L2"] == "DO.R"]["POD"].unique())
        & ap_pods & pods_power
    )
    pods_donr = (
        set(df_base[df_base["ATECO_L2"] == "DO.NR"]["POD"].unique())
        & ap_pods & pods_power
    )

    st.caption(
        f"Power class **{sel_power_label}** — "
        f"DO.R: **{len(pods_dor):,}** PODs | DO.NR: **{len(pods_donr):,}** PODs"
    )

    # ── Compute our profiles (cached per power class) ─────────────────────────
    cache_key = f"_arera_our_profiles_{sel_power_label}"
    if cache_key not in st.session_state:
        with st.spinner(f"Computing kWh profiles for power class {sel_power_label}…"):
            st.session_state[cache_key] = {
                "DO.R":  compute_our_hourly_kwh_by_daytype(df_meas_ap, pods_dor),
                "DO.NR": compute_our_hourly_kwh_by_daytype(df_meas_ap, pods_donr),
            }
    our_profiles = st.session_state[cache_key]

    # ── ARERA profiles for markets defined in this power class ───────────────
    MARKETS = sel_pc["markets"]
    DAY_TYPES_EN = ["Weekday", "Saturday", "Sunday"]
    DAY_TYPES_IT = list(ARERA_DAYTYPE_MAP.keys())

    # Colour mapping: up to 2 ARERA markets
    MARKET_COLORS = {
        "Maggior Tutela": ("#ff7f0e", "dash"),
        "Mercato Libero":  ("#2ca02c", "dot"),
        "Tutti":           ("#d62728", "dash"),
    }

    groups_cfg = [
        ("DO.R",  "Residente"),
        ("DO.NR", "Non Residente"),
    ]
    group_tabs = st.tabs([
        "Domestic Resident (DO.R)",
        "Domestic Non-Resident (DO.NR)",
    ])

    arera_export_data: dict[str, dict] = {}

    for tab_widget, (group_key, residenza) in zip(group_tabs, groups_cfg):
        with tab_widget:
            our_dt = our_profiles.get(group_key, {})
            if not our_dt:
                st.warning(f"No PoliTo data for **{group_key}** at power class **{sel_power_label}**.")
                continue

            # Pre-fetch ARERA kWh for both markets
            arera_kwh: dict[str, dict[str, dict[int, np.ndarray]]] = {}
            for mkt in MARKETS:
                arera_kwh[mkt] = compute_arera_hourly_kwh(df_arera, mkt, residenza)

            all_metric_rows = []
            export_figs: dict[str, go.Figure] = {}

            # ── 3 boxes: one per day type ─────────────────────────────────────
            st.markdown(
                f"#### {sel_month_lbl} — {sel_power_label} — kWh/h by day type"
            )
            day_cols = st.columns(3)
            for col_widget, (dtype_en, dtype_it) in zip(
                day_cols, zip(DAY_TYPES_EN, DAY_TYPES_IT)
            ):
                with col_widget:
                    our_arr = our_dt.get(dtype_en, {}).get(sel_month)
                    # Build market arrays dynamically
                    mkt_arrs = {
                        mkt: arera_kwh[mkt].get(dtype_en, {}).get(sel_month)
                        for mkt in MARKETS
                    }
                    mt_arr = mkt_arrs.get("Maggior Tutela") if mkt_arrs.get("Maggior Tutela") is not None else mkt_arrs.get("Tutti")
                    ml_arr = mkt_arrs.get("Mercato Libero")

                    fig = _arera_daytype_chart(
                        sel_month, sel_month_lbl, dtype_en,
                        our_arr, mt_arr, ml_arr,
                        show_legend=True,
                        market_labels=(
                            MARKETS[0] if len(MARKETS) >= 1 else "ARERA",
                            MARKETS[1] if len(MARKETS) >= 2 else None,
                        ),
                    )
                    st.plotly_chart(
                        fig, use_container_width=True,
                        key=f"arera_{group_key}_{dtype_en}_{sel_month}_{sel_power_label}",
                        config={"toImageButtonOptions": {"format": "png", "scale": 4, "width": 1600, "height": 900}})
                    export_figs[f"{dtype_en}_{sel_month_lbl}"] = fig

            # ── Metrics for ALL months × day types × markets ──────────────────
            _day_types_for_metrics = ["Weekday", "Saturday", "Sunday"]
            for mk in MONTH_KEYS:
                ml_lbl = MONTH_LBLS[MONTH_KEYS.index(mk)]
                for _dtype in _day_types_for_metrics:
                    for mkt_label in MARKETS:
                        our_arr_m = our_dt.get(_dtype, {}).get(mk)
                        ref_arr_m = arera_kwh.get(mkt_label, {}).get(_dtype, {}).get(mk)
                        if our_arr_m is None or ref_arr_m is None:
                            continue
                        mask_m = ~(np.isnan(our_arr_m) | np.isnan(ref_arr_m))
                        if mask_m.sum() < 2:
                            continue
                        r_m, _   = pearsonr(our_arr_m[mask_m], ref_arr_m[mask_m])
                        rmse_m   = float(np.sqrt(np.mean((our_arr_m[mask_m] - ref_arr_m[mask_m])**2)))
                        mae_m    = float(np.mean(np.abs(our_arr_m[mask_m] - ref_arr_m[mask_m])))
                        ref_mean = float(np.mean(ref_arr_m[mask_m]))
                        cv_rmse  = (rmse_m / ref_mean * 100) if ref_mean > 0 else 0.0
                        cv_mae   = (mae_m  / ref_mean * 100) if ref_mean > 0 else 0.0
                        all_metric_rows.append({
                            "Period":       ml_lbl,
                            "Day Type":     _dtype,
                            "Power Class":  sel_power_label,
                            "ARERA Market": mkt_label,
                            "Pearson r":    round(float(r_m), 4),
                            "RMSE (%)":     round(cv_rmse, 2),
                            "MAE (%)":      round(cv_mae, 2),
                        })

            # ── Monthly overview ──────────────────────────────────────────────
            st.markdown("#### Full Monthly Overview")
            for dtype_en, dtype_it in zip(DAY_TYPES_EN, DAY_TYPES_IT):
                with st.expander(f"{dtype_en}", expanded=False):
                    for row_start in range(0, 13, 4):
                        mcols = st.columns(4)
                        for ci, mk in enumerate(MONTH_KEYS[row_start:row_start + 4]):
                            ml_lbl = MONTH_LBLS[MONTH_KEYS.index(mk)]
                            with mcols[ci]:
                                mkt_arrs_m = {
                                    mkt: arera_kwh[mkt].get(dtype_en, {}).get(mk)
                                    for mkt in MARKETS
                                }
                                mt_m = mkt_arrs_m.get("Maggior Tutela") if mkt_arrs_m.get("Maggior Tutela") is not None else mkt_arrs_m.get("Tutti")
                                ml_m = mkt_arrs_m.get("Mercato Libero")
                                fig_m = _arera_daytype_chart(
                                    mk, ml_lbl, dtype_en,
                                    our_dt.get(dtype_en, {}).get(mk),
                                    mt_m, ml_m,
                                    show_legend=(mk == 0),
                                    market_labels=(
                                        MARKETS[0] if len(MARKETS) >= 1 else "ARERA",
                                        MARKETS[1] if len(MARKETS) >= 2 else None,
                                    ),
                                )
                                fig_m.update_layout(
                                    height=230,
                                    margin=dict(t=30, b=60, l=55, r=5),
                                    plot_bgcolor="#f0f5fc",
                                    paper_bgcolor="#ffffff",
                                    font=dict(family="Arial, sans-serif", color="#0d1f3c"),
                                    xaxis=dict(tickfont=dict(size=7, color="#1a3a6b"),
                                               gridcolor="#d0dff0", zeroline=False),
                                    yaxis=dict(tickfont=dict(size=7, color="#1a3a6b"),
                                               gridcolor="#d0dff0", zeroline=False),
                                    title=dict(font=dict(size=10, color="#0d1f3c")),
                                    legend=dict(font=dict(size=7, color="#0d1f3c")),
                                )
                                st.plotly_chart(
                                    fig_m, use_container_width=True,
                                    key=f"arera_ov_{group_key}_{dtype_en}_{mk}_{sel_power_label}",
                                    config={"toImageButtonOptions": {"format": "png", "scale": 4, "width": 1600, "height": 900}})
                                export_figs[f"{dtype_en}_{ml_lbl}"] = fig_m

            # ── Metrics table ─────────────────────────────────────────────────
            st.markdown("#### Similarity Metrics")
            st.caption("Pearson r: shape similarity (1=perfect). RMSE/MAE in % (normalised by mean of ARERA reference profile).")
            df_met = pd.DataFrame(all_metric_rows) if all_metric_rows else pd.DataFrame()
            if not df_met.empty:
                # Ensure correct column order
                col_order = ["Period", "Day Type", "Power Class", "ARERA Market", "Pearson r", "RMSE (%)", "MAE (%)"]
                col_order = [c for c in col_order if c in df_met.columns]
                df_met = df_met[col_order]
                st.dataframe(df_met, hide_index=True, use_container_width=True)
                _xl_arera = io.BytesIO()
                with pd.ExcelWriter(_xl_arera, engine="openpyxl") as _wa:
                    df_met.to_excel(_wa, index=False, sheet_name="Similarity_Metrics")
                _xl_arera.seek(0)
                st.download_button(
                    label="⬇️ Download Metrics Excel",
                    data=_xl_arera.getvalue(),
                    file_name=f"ARERA_similarity_metrics_{group_key}_{sel_power_label}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"arera_metrics_dl_{group_key}_{sel_power_label}",
                )

            arera_export_data[group_key] = {
                "residenza":  residenza,
                "figs":       export_figs,
                "metrics_df": df_met,
                "our_dt":     our_dt,
                "arera_kwh":  arera_kwh,
            }

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Export ARERA Comparison")
    safe_pwr = sel_power_label.replace(" ", "_").replace(">", "gt").replace("≤", "le")
    st.caption(
        f"Exports current view: power class **{sel_power_label}**, "
        f"month **{sel_month_lbl}** — HTML charts + Excel."
    )

    if st.button("Prepare ARERA Export ZIP", type="primary", key="arera_export_btn"):
        arera_buf  = io.BytesIO()
        arera_errs = []
        hour_labels = [f"{h:02d}:00" for h in range(24)]
        n_figs = sum(len(v["figs"]) for v in arera_export_data.values())
        total_steps = max(1, n_figs + len(arera_export_data))
        step = 0
        prog = st.progress(0, text="Building ARERA export…")

        with zipfile.ZipFile(arera_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for group_key, exp in arera_export_data.items():
                safe_grp  = group_key.replace(".", "_")
                figs       = exp["figs"]
                metrics_df = exp["metrics_df"]
                our_dt     = exp["our_dt"]
                arera_kwh  = exp["arera_kwh"]

                for fig_key, fig in figs.items():
                    step += 1
                    prog.progress(step / total_steps, text=f"Chart {fig_key}…")
                    try:
                        fname = f"{safe_grp}_{safe_pwr}_{fig_key.replace(' ','_')}.html"
                        zf.writestr(fname, fig.to_html(include_plotlyjs="cdn"))
                    except Exception as e:
                        arera_errs.append(f"{group_key} {fig_key}: {e}")

                step += 1
                prog.progress(step / total_steps, text=f"Excel — {group_key}…")
                try:
                    xl = io.BytesIO()
                    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
                        if not metrics_df.empty:
                            metrics_df.to_excel(writer, index=False, sheet_name="Similarity_Metrics")
                        rows = []
                        for mk in MONTH_KEYS:
                            ml_lbl = MONTH_LBLS[MONTH_KEYS.index(mk)]
                            for dtype_en in DAY_TYPES_EN:
                                o = our_dt.get(dtype_en, {}).get(mk)
                                mkt_vals = {
                                    mkt: arera_kwh[mkt].get(dtype_en, {}).get(mk)
                                    for mkt in MARKETS
                                }
                                for h_idx, hl in enumerate(hour_labels):
                                    row = {
                                        "Power Class": sel_power_label,
                                        "Period":      ml_lbl,
                                        "Day Type":    dtype_en,
                                        "Hour":        hl,
                                        "PoliTo [kWh]": round(float(o[h_idx]), 5) if o is not None else None,
                                    }
                                    for mkt, arr in mkt_vals.items():
                                        row[f"ARERA_{mkt} [kWh]"] = round(float(arr[h_idx]), 5) if arr is not None else None
                                    rows.append(row)
                        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Profile_Data_kWh")
                    zf.writestr(f"{safe_grp}_{safe_pwr}_arera_comparison.xlsx", xl.getvalue())
                except Exception as e:
                    arera_errs.append(f"{group_key} Excel: {e}")

            if arera_errs:
                zf.writestr("_ARERA_EXPORT_ISSUES.txt", "\n".join(f"- {e}" for e in arera_errs))

        arera_buf.seek(0)
        prog.progress(1.0, text="ARERA export ready!")
        prog.empty()
        st.session_state["_arera_export_zip"] = arera_buf.getvalue()
        st.session_state["_arera_export_errors"] = arera_errs

    if "_arera_export_zip" in st.session_state:
        if st.session_state.get("_arera_export_errors"):
            st.warning(f"{len(st.session_state['_arera_export_errors'])} issues.")
        st.download_button(
            label="⬇️ Download ARERA Comparison ZIP",
            data=st.session_state["_arera_export_zip"],
            file_name=f"arera_comparison_{safe_pwr}.zip",
            mime="application/zip",
            type="primary",
            key="arera_download_btn",
        )

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    st.set_page_config(
        page_title="ATECO Hierarchical Clustering",
        page_icon="📊",
        layout="wide",
    )

    # ── Logo ──────────────────────────────────────────────────────────────────
    logo_path = Path(__file__).parent / "images" / "logo_energy_center.png"
    if logo_path.exists():
        st.image(str(logo_path), width=440)

    st.title("Electrical Profiles Data Explorer - PoliTo")
    st.caption("Energy Center Lab, DENERG, Politecnico di Torino — Author: Lorenzo Giannuzzo")

    st.markdown("""
    <style>
    /* ── MAIN APP BACKGROUND ──────────────────────────────────────── */
    [data-testid="stApp"],
    .main,
    [data-testid="stMain"] {
        background-color: #0d1f3c !important;
    }

    /* ── SIDEBAR ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #1a3a6b !important;
    }
    [data-testid="stSidebar"] * {
        color: #e8f4fd !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #e8f4fd !important;
    }

    /* ── TOP HEADER BAR ───────────────────────────────────────────── */
    [data-testid="stHeader"] {
        background-color: #0d1f3c !important;
    }

    /* ── GLOBAL TEXT ──────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6,
    p, span, label, div,
    .stMarkdown, .stCaption {
        color: #e8f4fd !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #e8f4fd !important;
    }

    /* ── TABS ─────────────────────────────────────────────────────── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: #0d1f3c !important;
        border-bottom: 2px solid #ffffff !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: #0d1f3c !important;
        color: #a8c4e0 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background-color: #1a3a6b !important;
        color: #ffffff !important;
        border-bottom: 3px solid #ffffff !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #ffffff !important;
    }

    /* ── EXPANDER HEADERS ─────────────────────────────────────────── */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover,
    details summary,
    details > summary {
        background-color: #1a3a6b !important;
        color: #e8f4fd !important;
        border-radius: 4px !important;
    }
    [data-testid="stExpander"] summary:hover {
        background-color: #2e5ea8 !important;
    }
    /* Arrow/chevron icon in expander */
    [data-testid="stExpander"] summary svg {
        fill: #e8f4fd !important;
        stroke: #e8f4fd !important;
    }
    [data-testid="stExpander"],
    [data-testid="stExpanderDetails"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1a3a6b !important;
        border: 1px solid #2e5ea8 !important;
        border-radius: 6px !important;
    }

    /* ── INFO / WARNING / ERROR BOXES ─────────────────────────────── */
    [data-testid="stInfo"] {
        background-color: #1a3a6b !important;
        border-left: 4px solid #4a9eff !important;
    }
    [data-testid="stWarning"] {
        background-color: #2a3a20 !important;
        border-left: 4px solid #f9a825 !important;
    }
    [data-testid="stError"] {
        background-color: #3a1a1a !important;
        border-left: 4px solid #e53935 !important;
    }
    [data-testid="stSuccess"] {
        background-color: #1a3a20 !important;
        border-left: 4px solid #2e7d32 !important;
    }

    /* ── TOOLTIP / HELP ICONS — keep grey ────────────────────────── */
    button[data-testid="stTooltipIcon"] svg,
    button[data-testid="stTooltipIcon"] svg path,
    button[data-testid="stTooltipIcon"] svg circle {
        fill: #8a9ab5 !important;
        stroke: none !important;
        color: #8a9ab5 !important;
    }
    button[data-testid="stTooltipIcon"]:hover svg,
    button[data-testid="stTooltipIcon"]:hover svg path,
    button[data-testid="stTooltipIcon"]:hover svg circle {
        fill: #b0c4d8 !important;
    }
    /* accent-color for native inputs */
    input[type="radio"],
    input[type="checkbox"] {
        accent-color: #2e7d32 !important;
    }

    /* ── SELECTBOX / DROPDOWNS ────────────────────────────────────── */
    [data-testid="stSelectbox"] > div > div,
    [data-baseweb="select"] {
        background-color: #1a3a6b !important;
        border-color: #2e5ea8 !important;
        color: #e8f4fd !important;
    }
    [data-baseweb="select"] * {
        background-color: #1a3a6b !important;
        color: #e8f4fd !important;
    }
    [data-baseweb="popover"] [role="option"] {
        background-color: #1a3a6b !important;
        color: #e8f4fd !important;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background-color: #2e5ea8 !important;
    }

    /* ── BUTTONS ──────────────────────────────────────────────────── */
    [data-testid="stButton"] > button {
        background-color: #1a3a6b !important;
        color: #e8f4fd !important;
        border: 1px solid #2e5ea8 !important;
        border-radius: 5px !important;
    }
    [data-testid="stButton"] > button:hover {
        background-color: #2e5ea8 !important;
        border-color: #4a9eff !important;
    }
    /* Primary buttons — green */
    [data-testid="stButton"] > button[kind="primary"] {
        background-color: #2e7d32 !important;
        border-color: #43a047 !important;
        color: #ffffff !important;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #43a047 !important;
    }

    /* ── DOWNLOAD BUTTON ──────────────────────────────────────────── */
    [data-testid="stDownloadButton"] > button {
        background-color: #1a3a6b !important;
        color: #e8f4fd !important;
        border: 1px solid #2e5ea8 !important;
    }
    [data-testid="stDownloadButton"] > button[kind="primary"] {
        background-color: #2e7d32 !important;
        border-color: #43a047 !important;
        color: #ffffff !important;
    }

    /* ── DATAFRAME / TABLE ────────────────────────────────────────── */
    /* GlideDataGrid (canvas-based) is themed via config.toml base=dark.
       Only style the outer wrapper border here. */
    [data-testid="stDataFrame"] {
        border: 1px solid #2e5ea8 !important;
        border-radius: 6px !important;
    }
    /* Header row background for GlideDataGrid */
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] thead,
    [data-testid="stDataFrame"] thead tr,
    [data-testid="stDataFrame"] thead th,
    [data-testid="stDataFrame"] .dvn-header,
    [class*="dvn-header"],
    [class*="headerRow"],
    [class*="colHeader"] {
        background-color: #1a3a6b !important;
        color: #e8f4fd !important;
    }

    /* ── PROGRESS BAR ─────────────────────────────────────────────── */
    [data-testid="stProgress"] > div {
        background-color: #2e7d32 !important;
    }

    /* ── METRIC CARDS ─────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background-color: #1a3a6b !important;
        border: 1px solid #2e5ea8 !important;
        border-radius: 6px !important;
        padding: 10px !important;
    }

    /* ── ATECO CHECKBOX NOWRAP (keep existing) ────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCheckbox"] label {
        white-space: nowrap;
        color: #e8f4fd !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow"] {
        overflow-x: auto !important;
        overflow-y: auto !important;
    }

    /* ── PLOTLY CHART CONTAINER ───────────────────────────────────── */
    [data-testid="stPlotlyChart"] {
        background-color: transparent !important;
    }

    /* ── SLIDER ───────────────────────────────────────────────────── */
    [data-testid="stSlider"] [data-testid="stSliderThumb"] {
        background-color: #2e7d32 !important;
    }
    /* ── LOADING OVERLAY via CSS :has() ──────────────────────────── */
    /* Hide Streamlit's default opacity dimming on rerun */
    [data-testid="stApp"] {
        transition: none !important;
        opacity: 1 !important;
    }
    /* Show our overlay whenever Streamlit's status widget is present */
    body:has([data-testid="stStatusWidget"]) #ld-overlay {
        display: flex !important;
    }
    body:has([data-testid="stStatusWidget"]) [data-testid="stMain"],
    body:has([data-testid="stStatusWidget"]) [data-testid="stSidebar"] {
        opacity: 1 !important;
        pointer-events: none !important;
    }
    #ld-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(13, 31, 60, 0.93);
      z-index: 2147483647;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 28px;
      backdrop-filter: blur(3px);
      -webkit-backdrop-filter: blur(3px);
    }
    #ld-logo {
      width: 320px;
      max-width: 80vw;
      opacity: 0.95;
    }
    #ld-text {
      color: #e8f4fd !important;
      font-family: Arial, sans-serif !important;
      font-size: 15px !important;
      letter-spacing: 0.04em;
      margin: 0 !important;
      opacity: 0.75;
    }
    #ld-bar-wrap {
      width: 320px;
      max-width: 80vw;
      height: 5px;
      background: rgba(255,255,255,0.12);
      border-radius: 4px;
      overflow: hidden;
    }
    #ld-bar {
      height: 100%;
      width: 40%;
      background: linear-gradient(90deg, #2e7d32, #4fc3f7, #2e7d32);
      background-size: 200% 100%;
      border-radius: 4px;
      animation: ld-slide 1.4s ease-in-out infinite,
                 ld-gradient 2s linear infinite;
    }
    @keyframes ld-slide {
      0%   { transform: translateX(-100%); }
      100% { transform: translateX(350%); }
    }
    @keyframes ld-gradient {
      0%   { background-position: 0% 50%; }
      100% { background-position: 200% 50%; }
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Custom loading overlay — always in DOM, CSS controls visibility ──────
    import base64
    _logo_path = Path(__file__).parent / "images" / "logo_energy_center.png"
    _logo_b64 = ""
    if _logo_path.exists():
        with open(_logo_path, "rb") as _f:
            _logo_b64 = base64.b64encode(_f.read()).decode()

    _logo_html = (
        f"<img id='ld-logo' src='data:image/png;base64,{_logo_b64}'>"
        if _logo_b64 else ""
    )

    st.markdown(f"""
    <div id="ld-overlay">
      {_logo_html}
      <p id="ld-text">Updating dashboard…</p>
      <div id="ld-bar-wrap"><div id="ld-bar"></div></div>
    </div>
    """, unsafe_allow_html=True)

    if "ateco_loaded" not in st.session_state:
        load_ateco_classification()
        st.session_state["ateco_loaded"] = True
        if len(ATECO_LOOKUP) <= len(DISTRIBUTOR_CODES):
            st.warning(f"ATECO 2025 classification file not found in `data/{ATECO_EXCEL_NAME}`. "
                       "Using distributor codes only.")
    elif not ATECO_LOOKUP:
        load_ateco_classification()

    if "data_loaded" not in st.session_state or "df_potcontr" not in st.session_state:
        for key in ["data_loaded", "df_meta", "df_meas", "df_potcontr", "issues",
                     "profile_norm", "profile_raw", "pods_12m"]:
            st.session_state.pop(key, None)

        df_meta, df_meas, df_potcontr, issues = load_all_data()

        if not df_meas.empty:
            q_present = [c for c in Q_COLS if c in df_meas.columns]
            df_meas[q_present] = df_meas[q_present].apply(pd.to_numeric, errors="coerce").astype(np.float32)
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
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.header("Global Filters")

        if st.button("🔄 Reload Data"):
            prepare_metadata.clear()
            load_gse_profiles.clear()
            load_arera_profiles.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        has_tipologia = ("Tipologia" in df_meas.columns and df_meas["Tipologia"].notna().any())
        if has_tipologia:
            tip_values = sorted(df_meas["Tipologia"].dropna().unique().tolist())
            tip_counts = df_meas.groupby("Tipologia")["POD"].nunique()
            sel_tipologia = st.radio(
                "Data Type",
                ["All"] + tip_values,
                format_func=lambda x: (
                    f"All ({df_meas['POD'].nunique():,} PODs)" if x == "All"
                    else f"{x} ({tip_counts.get(x, 0):,} PODs)"
                ),
                index=0, key="tip_filter",
                help=(
                    "AP = Attiva Prelevata (consumo dalla rete)\n"
                    "AN = Attiva iNiettata (immissione FV)\n"
                    "RLP/RLN/RCP/RCN = Reattiva"
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
            "Data Extension", ["All PODs", "12+ months only"], index=0,
            help="Filter to PODs with ≥12 distinct months of data."
        )
        use_12m_filter = (coverage_opt == "12+ months only")

        profile_period = st.selectbox(
            "Profile period", MONTH_LABELS, index=0,
            help="Select which period to use for clustering.",
        )
        profile_month = MONTH_LABELS.index(profile_period)

        st.divider()
        st.header("Contractual Power")
        has_potcontr = (
            "POTCONTR_kW" in df_unique.columns and df_unique["POTCONTR_kW"].notna().any()
        )

        pot_filter_mask = pd.Series(True, index=df_unique.index)

        if has_potcontr:
            enable_pot_filter = st.checkbox("Enable power filter", value=False, key="pot_en")
            if enable_pot_filter:
                # Build per-bin counts using the global POTCONTR_BINS
                df_pot_tmp = df_unique.dropna(subset=["POTCONTR_kW"]).copy()
                df_pot_tmp["_PRange"] = pd.cut(
                    df_pot_tmp["POTCONTR_kW"],
                    bins=POTCONTR_BINS,
                    labels=POTCONTR_BIN_LABELS,
                    right=True,
                )
                range_counts = (
                    df_pot_tmp.groupby("_PRange", observed=True)["POD"]
                    .nunique()
                    .reindex(POTCONTR_BIN_LABELS, fill_value=0)
                )

                selected_ranges = []
                with st.container(height=220):
                    for range_label in POTCONTR_BIN_LABELS:
                        n_r = int(range_counts.get(range_label, 0))
                        if n_r == 0:
                            continue
                        checked = st.checkbox(
                            f"{range_label} ({n_r:,})",
                            value=True,
                            key=f"pot2_range_{range_label}",
                        )
                        if checked:
                            selected_ranges.append(range_label)

                include_missing = st.checkbox(
                    "Include PODs with missing power data", value=False, key="pot2_na"
                )

                if selected_ranges:
                    df_all_tmp = df_unique.copy()
                    df_all_tmp["_PRange"] = pd.cut(
                        df_all_tmp["POTCONTR_kW"],
                        bins=POTCONTR_BINS,
                        labels=POTCONTR_BIN_LABELS,
                        right=True,
                    )
                    pot_filter_mask = df_all_tmp["_PRange"].isin(selected_ranges)
                else:
                    pot_filter_mask = pd.Series(False, index=df_unique.index)

                if include_missing:
                    pot_filter_mask = pot_filter_mask | df_unique["POTCONTR_kW"].isna()
        else:
            st.info("No contractual power data available.")
            enable_pot_filter = False

        st.divider()
        st.header("Cluster Settings")
        cluster_mode = st.radio(
            "Number of clusters",
            ["Automatic (majority vote)", "Manual"],
            index=0, key="cluster_mode"
        )
        manual_k = None
        if cluster_mode == "Manual":
            manual_k = st.slider("k (number of clusters)", 2, 15, 4, key="manual_k_slider")

        st.divider()
        st.header("Statistics")
        n_pod_total = df_unique["POD"].nunique()
        n_pod_meas = df_meas_filtered["POD"].nunique()
        st.metric("PODs (metadata)", f"{n_pod_total:,}")
        st.metric("PODs (measures)", f"{n_pod_meas:,}")

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
        filter_parts.append(f"Type = {sel_tipologia}")
    if use_12m_filter:
        filter_parts.append("12+ months")
    if has_potcontr and enable_pot_filter:
        filter_parts.append("Power filter ON")
    if profile_month > 0:
        filter_parts.append(f"Month={MONTH_LABELS[profile_month]}")

    if filter_parts:
        st.info(f"**Active filters:** {' | '.join(filter_parts)} → **{n_base:,} PODs** available".replace(",", "\u202f"))
    else:
        st.info(f"No global filters → **{n_base:,} PODs** available".replace(",", "\u202f"))

    if profile_norm.empty:
        st.error("No load profiles available. Check data.")
        return

    l1_counts = (
        df_base.groupby("ATECO_L1")["POD"].nunique()
        .reset_index().rename(columns={"POD": "N"})
        .sort_values("N", ascending=False)
    )

    # =========================================================================
    # MAIN TABS
    # =========================================================================
    tab_cluster, tab_gse, tab_arera = st.tabs([
        "Clustering Explorer",
        "GSE Profile Comparison",
        "ARERA Profile Comparison",
    ])

    with tab_cluster:
        # ── 1. ATECO CLUSTERING SECTION ──────────────────────────────────────
        ateco_clustering_section(df_base, profile_norm, df_unique, manual_k,
                                 profile_month, l1_counts)

        # ── 2. DISTRIBUTION CHARTS ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("Data Distribution Overview")

        period_label = f"Filter: {' | '.join(filter_parts)}" if filter_parts else "All data"

        st.markdown("#### Monthly Consumption Distribution — Top 15 Typologies")
        with st.spinner("Computing consumption distribution..."):
            df_meas_for_dist = df_meas_filtered[df_meas_filtered["POD"].isin(df_base["POD"])]
            fig_dist, summary_dist = plot_consumption_distribution_top15(
                df_meas_for_dist, df_base, title_suffix=period_label
            )
        if fig_dist is not None:
            st.plotly_chart(fig_dist, use_container_width=True, key="main_consumption_dist", config={"toImageButtonOptions": {"format": "png", "scale": 4, "width": 1600, "height": 900}})
            with st.expander("Summary Statistics — Consumption"):
                st.dataframe(summary_dist, hide_index=True, use_container_width=True)
        else:
            st.warning("Not enough data to build consumption distribution chart.")

        if has_potcontr:
            st.markdown("#### Contractual Power Distribution")
            fig_pie = plot_potcontr_pie(df_unique)
            if fig_pie is not None:
                st.plotly_chart(fig_pie, use_container_width=True, key="main_potcontr_pie", config={"toImageButtonOptions": {"format": "png", "scale": 4, "width": 1600, "height": 900}})

        if has_potcontr:
            st.markdown("#### Contractual Power by Typology — Top 10")
            fig_stack = plot_potcontr_stacked_bar(df_unique, top_n=10)
            if fig_stack is not None:
                st.plotly_chart(fig_stack, use_container_width=True, key="main_potcontr_stack", config={"toImageButtonOptions": {"format": "png", "scale": 4, "width": 1600, "height": 900}})

    with tab_gse:
        gse_comparison_tab(df_base, df_meas_filtered, pods_12m)

    with tab_arera:
        arera_comparison_tab(df_base, df_meas_filtered, pods_12m)


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
                    _rendered_l2: set[str] = set()  # guard against duplicate keys
                    for l1_code in selected_l1:
                        l2_of_l1 = l2_data[l2_data["ATECO_L1"] == l1_code].sort_values("N", ascending=False)
                        if l2_of_l1.empty:
                            continue

                        l1_desc = lookup_ateco_description(l1_code)
                        # Only count / show children not yet rendered in this pass
                        child_codes_l2 = [
                            c for c in l2_of_l1["ATECO_L2"].tolist()
                            if c not in _rendered_l2
                        ]
                        if not child_codes_l2:
                            continue
                        l1_total = int(
                            l2_of_l1[l2_of_l1["ATECO_L2"].isin(child_codes_l2)]["N"].sum()
                        )

                        all_kids_on = all(st.session_state.get(f"cb_l2_{c}", False) for c in child_codes_l2)
                        prev_par = st.session_state.get(f"_pv_l2par_{l1_code}", None)

                        if f"cb_l2par_{l1_code}" not in st.session_state:
                            st.session_state[f"cb_l2par_{l1_code}"] = all_kids_on
                        parent_val = st.checkbox(
                            f"▸ {l1_code} — {l1_desc} ({l1_total})", key=f"cb_l2par_{l1_code}")
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
                                if code in _rendered_l2:
                                    # Already rendered under a previous parent; just
                                    # collect its current value without re-rendering.
                                    if st.session_state.get(f"cb_l2_{code}", False):
                                        selected_l2.append(code)
                                    continue
                                _rendered_l2.add(code)
                                n_pods = row["N"]
                                desc = lookup_ateco_description(code)
                                label = f"{code} — {desc}" if desc else str(code)
                                if f"cb_l2_{code}" not in st.session_state:
                                    st.session_state[f"cb_l2_{code}"] = False
                                checked = st.checkbox(f"{label} ({n_pods})", key=f"cb_l2_{code}")
                                if checked:
                                    selected_l2.append(code)
                    # Deduplicate (same code may have been appended via multiple parents)
                    selected_l2 = list(dict.fromkeys(selected_l2))

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
                    _rendered_l3: set[str] = set()  # guard against duplicate keys
                    for l2_code in selected_l2:
                        l3_of_l2 = l3_data[l3_data["ATECO_L2"] == l2_code].sort_values("N", ascending=False)
                        if l3_of_l2.empty:
                            continue

                        l2_desc = lookup_ateco_description(l2_code)
                        child_codes_l3 = [
                            c for c in l3_of_l2["ATECO_L3"].tolist()
                            if c not in _rendered_l3
                        ]
                        if not child_codes_l3:
                            continue
                        l2_total = int(
                            l3_of_l2[l3_of_l2["ATECO_L3"].isin(child_codes_l3)]["N"].sum()
                        )

                        all_kids_on_l3 = all(st.session_state.get(f"cb_l3_{c}", False) for c in child_codes_l3)
                        prev_par_l3 = st.session_state.get(f"_pv_l3par_{l2_code}", None)

                        if f"cb_l3par_{l2_code}" not in st.session_state:
                            st.session_state[f"cb_l3par_{l2_code}"] = all_kids_on_l3
                        parent_val_l3 = st.checkbox(
                            f"▸ {l2_code} — {l2_desc} ({l2_total})", key=f"cb_l3par_{l2_code}")
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
                                if code in _rendered_l3:
                                    if st.session_state.get(f"cb_l3_{code}", False):
                                        selected_l3.append(code)
                                    continue
                                _rendered_l3.add(code)
                                n_pods = row["N"]
                                desc = lookup_ateco_description(code)
                                label = f"{code} — {desc}" if desc else str(code)
                                if f"cb_l3_{code}" not in st.session_state:
                                    st.session_state[f"cb_l3_{code}"] = False
                                checked = st.checkbox(f"{label} ({n_pods})", key=f"cb_l3_{code}")
                                if checked:
                                    selected_l3.append(code)
                    # Deduplicate
                    selected_l3 = list(dict.fromkeys(selected_l3))

    # =========================================================================
    # SELECTION SUMMARY + RUN BUTTON
    # =========================================================================
    if not selected_l1 and not selected_l2 and not selected_l3:
        st.info("Select ATECO codes above, then press **Run Clustering**.")
        return

    pods_l1 = list(df_base[df_base["ATECO_L1"].isin(selected_l1)]["POD"].unique()) if selected_l1 else []
    pods_l2 = list(df_base[df_base["ATECO_L2"].isin(selected_l2)]["POD"].unique()) if selected_l2 else []
    pods_l3 = list(df_base[df_base["ATECO_L3"].isin(selected_l3)]["POD"].unique()) if selected_l3 else []

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

    monthly_breakdown_level = None
    if profile_month == 0 and active_levels:
        available_level_names = [lname for lname, _, _, _ in active_levels]
        opts = ["None"] + available_level_names
        _mbkd_sel = st.selectbox(
            "Monthly breakdown for level", opts, index=0,
            key="monthly_bkd_level",
            help="Run an independent clustering for each calendar month on the selected ATECO level.",
        )
        if _mbkd_sel != "None":
            monthly_breakdown_level = _mbkd_sel

    sel_fingerprint = (
        tuple(sorted(selected_l1)),
        tuple(sorted(selected_l2)),
        tuple(sorted(selected_l3)),
        manual_k,
        profile_month,
    )
    cached_fp = st.session_state.get("_cl_fingerprint", None)
    has_cached = (cached_fp == sel_fingerprint and "_cl_results" in st.session_state)

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        run_clicked = st.button("▶ Run Clustering", type="primary", key="run_cl_btn",
                                use_container_width=True)
    with btn_col2:
        if has_cached:
            st.button("Results cached ✓", disabled=True, key="cached_indicator",
                      use_container_width=True)
        elif cached_fp is not None and cached_fp != sel_fingerprint:
            st.button("Selection changed — click Run", disabled=True,
                      key="changed_indicator", use_container_width=True)

    # =========================================================================
    # COMPUTE OR LOAD CACHED
    # =========================================================================
    if run_clicked:
        results = {}
        for level_name, codes, pods, ateco_col in active_levels:
            pods_with_profile = list(set(profile_norm.index) & set(pods))
            if len(pods_with_profile) < 5:
                results[level_name] = {"error": f"Need ≥5 PODs (have {len(pods_with_profile)})"}
                continue

            with st.spinner(f"Clustering {level_name} ({len(pods_with_profile)} PODs)..."):
                X_df, k, details, Z, err = run_clustering_for_pods(
                    profile_norm, pods_with_profile,
                    n_clusters=manual_k, profile_month=profile_month
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

            # ── Main profile chart (mean ±1σ bands) ──────────────────────
            fig = plot_cluster_profiles(X_df, k, f" | {level_name}")
            st.plotly_chart(fig, use_container_width=True, key=f"frag_prof_{level_name}")

            if len(codes) >= 2:
                with st.expander(f"Cluster Composition by {ateco_col}", expanded=True):
                    fig_comp, comp_table = plot_cluster_composition(
                        X_df, df_base, ateco_col, level_name)
                    st.plotly_chart(fig_comp, use_container_width=True,
                                    key=f"frag_comp_{level_name}")
                    st.dataframe(comp_table, hide_index=True, use_container_width=True,
                                 key=f"frag_comptbl_{level_name}")

            parent_cols = []
            if level_name == "L2" and "ATECO_L1" in df_base.columns:
                parent_cols = [("ATECO_L1", "L1 Section")]
            elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                parent_cols = [("ATECO_L1", "L1 Section"), ("ATECO_L2", "L2 Division")]

            for par_col, par_label in parent_cols:
                with st.expander(f"Cluster Composition by {par_label}", expanded=False):
                    fig_par, tbl_par = plot_cluster_composition(
                        X_df, df_base, par_col, f"{level_name} → {par_label}")
                    st.plotly_chart(fig_par, use_container_width=True,
                                    key=f"frag_comp_{level_name}_{par_col}")
                    st.dataframe(tbl_par, hide_index=True, use_container_width=True,
                                 key=f"frag_comptbl_{level_name}_{par_col}")

            with st.expander("ATECO → Cluster Breakdown (rows = ATECO, cols = Clusters)",
                             expanded=False):
                bkd = build_ateco_cluster_breakdown(X_df, df_base, ateco_col, level_name)
                st.dataframe(bkd, hide_index=True, use_container_width=True,
                             key=f"frag_bkd_{level_name}")
                bkd_parents = []
                if level_name == "L2" and "ATECO_L1" in df_base.columns:
                    bkd_parents = [("ATECO_L1", "L1 Section")]
                elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                    bkd_parents = [("ATECO_L1", "L1 Section"), ("ATECO_L2", "L2 Division")]
                for par_c, par_lbl in bkd_parents:
                    st.markdown(f"**Breakdown by {par_lbl}:**")
                    bkd_par = build_ateco_cluster_breakdown(
                        X_df, df_base, par_c, f"{level_name} → {par_lbl}")
                    st.dataframe(bkd_par, hide_index=True, use_container_width=True,
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
                with st.expander("How to read Pearson r / P-values"):
                    st.markdown(PEARSON_HELP)
                with st.expander("P-values"):
                    st.dataframe(df_pval.style.format("{:.2e}"), use_container_width=True)

    # =========================================================================
    # CROSS-LEVEL COMPARISON
    # =========================================================================
    valid_results = {lname: res for lname, res in results.items() if "error" not in res}

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
                    r, _ = pearsonr(all_centroids[labels_all[i]], all_centroids[labels_all[j]])
                    cross_corr[i, j] = r
                    cross_corr[j, i] = r

            text_matrix = [[f"{cross_corr[i, j]:.3f}" for j in range(n_c)] for i in range(n_c)]
            fig_cross = go.Figure(data=go.Heatmap(
                z=cross_corr.tolist(), x=labels_all, y=labels_all,
                text=text_matrix, texttemplate="%{text}",
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                colorbar=dict(title="Pearson r"),
            ))
            fig_cross.update_layout(
                title="Cross-Level Centroid Correlation",
                height=max(400, n_c * 35 + 100),
                margin=dict(t=40, b=20, l=20, r=20),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#e8f4fd")),
                yaxis=dict(tickfont=dict(size=9, color="#e8f4fd")),
                plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c",
                font=dict(family="Arial, sans-serif", color="#e8f4fd"),
            )
            st.plotly_chart(fig_cross, use_container_width=True, key="frag_cross_corr")
            with st.expander("How to read Pearson r"):
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
                xaxis_title="Time of Day", yaxis_title="Normalized (0-1)",
                height=450,
                yaxis=dict(range=[-0.05, 1.05], gridcolor="#1e3a6b",
                           title_font=dict(color="#e8f4fd"), tickfont=dict(color="#e8f4fd")),
                xaxis=dict(dtick=8, tickangle=-45, gridcolor="#1e3a6b",
                           title_font=dict(color="#e8f4fd"), tickfont=dict(color="#e8f4fd")),
                legend=dict(font=dict(size=9, color="#e8f4fd")),
                plot_bgcolor="#0d2144", paper_bgcolor="#0d1f3c",
                font=dict(family="Arial, sans-serif", color="#e8f4fd"),
            )
            st.plotly_chart(fig_overlay, use_container_width=True, key="frag_overlay")

    # =========================================================================
    # MONTHLY BREAKDOWN
    # =========================================================================
    _mbkd_level = st.session_state.get("monthly_bkd_level", "None")
    if _mbkd_level and _mbkd_level != "None" and _mbkd_level in valid_results:
        st.markdown("---")
        st.subheader(f"Monthly Breakdown — {_mbkd_level}")

        res_bkd = valid_results[_mbkd_level]
        pods_bkd = list(res_bkd["X_df"].index)
        _mbkd_cache_key = f"_monthly_bkd_{_mbkd_level}"

        if st.session_state.get("_cl_fingerprint") != st.session_state.get(f"_monthly_fp_{_mbkd_level}"):
            st.session_state.pop(_mbkd_cache_key, None)

        if _mbkd_cache_key not in st.session_state:
            monthly_results = {}
            month_prog = st.progress(0, text="Computing monthly clusterings...")
            for m in range(1, 13):
                month_name = MONTH_NAMES[m]
                month_prog.progress((m - 1) / 12, text=f"Clustering {month_name} ({m}/12)...")
                X_m, k_m, details_m, Z_m, err_m = run_clustering_for_pods(
                    profile_norm, pods_bkd, n_clusters=None, profile_month=m)
                if err_m:
                    monthly_results[m] = {"error": err_m, "month_name": month_name}
                else:
                    monthly_results[m] = {"X_df": X_m, "k": k_m,
                                           "month_name": month_name, "n_pods": len(X_m)}
            month_prog.progress(1.0, text="Monthly clustering complete!")
            month_prog.empty()
            st.session_state[_mbkd_cache_key] = monthly_results
            st.session_state[f"_monthly_fp_{_mbkd_level}"] = st.session_state.get("_cl_fingerprint")
        else:
            monthly_results = st.session_state[_mbkd_cache_key]

        k_summary = {MONTH_NAMES[m]: (res["k"] if "error" not in res else "—")
                     for m, res in monthly_results.items()}
        st.dataframe(pd.DataFrame([k_summary], index=["k"]), use_container_width=True)

        for m in range(1, 13):
            res_m = monthly_results[m]
            month_name = res_m["month_name"]
            if "error" in res_m:
                with st.expander(f"{month_name} — insufficient data"):
                    st.warning(res_m["error"])
                continue
            X_m = res_m["X_df"]
            k_m = res_m["k"]
            n_m = res_m["n_pods"]
            with st.expander(f"{month_name} — k={k_m} | {n_m} PODs", expanded=False):
                fig_m = plot_cluster_profiles(X_m, k_m, f" | {_mbkd_level} — {month_name}")
                st.plotly_chart(fig_m, use_container_width=True,
                                key=f"frag_monthly_{_mbkd_level}_{m}")

                stats_m = compute_cluster_stats(X_m)
                st.dataframe(stats_m, hide_index=True, use_container_width=True,
                             key=f"frag_monthly_stats_{_mbkd_level}_{m}")
                global_m_vals = compute_global_metrics(X_m)
                mc = st.columns(len(global_m_vals))
                for col_w, (mname, mval) in zip(mc, global_m_vals.items()):
                    col_w.metric(mname, mval, help=METRIC_HELP.get(mname, ""))
                if k_m >= 2:
                    df_corr_m, _, _, _ = compute_centroid_pearson(X_m)
                    fig_corr_m = plot_pearson_heatmap(df_corr_m)
                    st.plotly_chart(fig_corr_m, use_container_width=True,
                                    key=f"frag_monthly_pears_{_mbkd_level}_{m}")

    # =========================================================================
    # EXPORT  — HTML only
    # =========================================================================
    st.markdown("---")
    st.subheader("Export Results")
    st.caption("Charts are exported as interactive HTML files.")

    if st.button("Prepare Export ZIP", type="primary", key="frag_export_btn"):
        buf = io.BytesIO()
        export_errors = []

        n_levels = len(valid_results)
        steps_per_level = 7
        total_steps = max(1, n_levels * steps_per_level)
        step = 0
        export_progress = st.progress(0, text="Preparing export...")

        all_centroids_df = build_all_centroids_df(valid_results)
        pod_cluster_map = build_pod_cluster_map(valid_results)
        ateco_dominant_df = build_ateco_dominant_cluster_df(valid_results, df_base)

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for level_name, res in valid_results.items():
                X_df = res["X_df"]
                k = res["k"]
                ateco_col = res.get("ateco_col", f"ATECO_{level_name}")

                # ── Pre-compute centroid chart HTML div (shared across exports) ──
                centroid_div = ""
                try:
                    fig_centroids_exp = plot_centroids_only(X_df, k, f" | {level_name}")
                    centroid_div = fig_centroids_exp.to_html(
                        include_plotlyjs="cdn", full_html=False
                    )
                except Exception as e:
                    export_errors.append(f"{level_name} centroid div: {e}")

                step += 1
                export_progress.progress(step / total_steps,
                                          text=f"Building Excel for {level_name}...")
                stats_df = compute_cluster_stats(X_df)
                global_m = compute_global_metrics(X_df)
                global_df = pd.DataFrame([global_m])

                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    stats_df.to_excel(writer, index=False, sheet_name="Cluster_Stats")
                    global_df.to_excel(writer, index=False, sheet_name="Global_Metrics")
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
                        on="POD", how="left")
                    pod_cl.to_excel(writer, index=False, sheet_name="POD_List")

                    if len(res.get("codes", [])) >= 2:
                        _, comp_tbl = plot_cluster_composition(X_df, df_base, ateco_col, level_name)
                        comp_tbl.to_excel(writer, index=False, sheet_name="Cluster_Composition")

                    par_cols_exp = []
                    if level_name == "L2" and "ATECO_L1" in df_base.columns:
                        par_cols_exp = [("ATECO_L1", "Composition_by_L1")]
                    elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                        par_cols_exp = [("ATECO_L1", "Composition_by_L1"),
                                        ("ATECO_L2", "Composition_by_L2")]
                    for par_c, sheet_name in par_cols_exp:
                        _, par_tbl = plot_cluster_composition(X_df, df_base, par_c, level_name)
                        par_tbl.to_excel(writer, index=False, sheet_name=sheet_name)

                    profiles_export = X_df.copy()
                    profiles_export.index.name = "POD"
                    profiles_export = profiles_export.reset_index()
                    q_exp = [c for c in profiles_export.columns if c.startswith("Q")]
                    profiles_export = profiles_export.rename(columns={"Cluster": f"Cluster_{level_name}"})
                    for other_level, other_mapping in pod_cluster_map.items():
                        if other_level == level_name:
                            continue
                        col_name = f"Cluster_{other_level}"
                        profiles_export[col_name] = profiles_export["POD"].map(other_mapping)
                        profiles_export[col_name] = pd.array(
                            profiles_export[col_name], dtype=pd.Int64Dtype())
                    cluster_cols_ordered = sorted(
                        [c for c in profiles_export.columns if c.startswith("Cluster_")])
                    meta_merge = ["CCATETE", "D_49DES", "POTCONTR_kW",
                                  "ATECO_L1", "ATECO_L2", "ATECO_L3"]
                    avail_meta = [c for c in meta_merge if c in df_unique.columns]
                    if avail_meta:
                        profiles_export = profiles_export.merge(
                            df_unique[["POD"] + avail_meta].drop_duplicates(subset=["POD"]),
                            on="POD", how="left")
                    final_cols = (["POD"] + cluster_cols_ordered + avail_meta + q_exp)
                    final_cols = [c for c in final_cols if c in profiles_export.columns]
                    profiles_export[final_cols].to_excel(
                        writer, index=False, sheet_name="Profiles_with_Cluster")

                    if not all_centroids_df.empty:
                        all_centroids_df.to_excel(writer, index=False, sheet_name="All_Centroids")

                    if not ateco_dominant_df.empty:
                        ateco_dominant_df.to_excel(writer, index=False,
                                                    sheet_name="ATECO_Dominant_Cluster")

                    bkd_own = build_ateco_cluster_breakdown(X_df, df_base, ateco_col, level_name)
                    bkd_own.to_excel(writer, index=False, sheet_name="ATECO_Breakdown")

                    inv_bkd_exp = build_cluster_ateco_breakdown(X_df, df_base, ateco_col, level_name)
                    inv_bkd_exp.to_excel(writer, index=False, sheet_name="Cluster_ATECO_Breakdown")

                    bkd_par_exp = []
                    if level_name == "L2" and "ATECO_L1" in df_base.columns:
                        bkd_par_exp = [("ATECO_L1", "Breakdown_by_L1")]
                    elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                        bkd_par_exp = [("ATECO_L1", "Breakdown_by_L1"),
                                       ("ATECO_L2", "Breakdown_by_L2")]
                    for bp_col, bp_sheet in bkd_par_exp:
                        bp_df = build_ateco_cluster_breakdown(X_df, df_base, bp_col, level_name)
                        bp_df.to_excel(writer, index=False, sheet_name=bp_sheet)

                zf.writestr(f"{level_name}_k{k}_stats.xlsx", excel_buf.getvalue())

                # Profile chart (HTML)
                step += 1
                export_progress.progress(step / total_steps,
                                          text=f"Profile chart {level_name}...")
                try:
                    fig = plot_cluster_profiles(X_df, k, f" | {level_name}")
                    zf.writestr(f"{level_name}_k{k}_profiles.html",
                                fig.to_html(include_plotlyjs="cdn"))
                except Exception as e:
                    export_errors.append(f"{level_name} profiles: {e}")

                # Centroid overlay chart (HTML)
                step += 1
                export_progress.progress(step / total_steps,
                                          text=f"Centroid overlay chart {level_name}...")
                try:
                    fig_co_exp = plot_centroids_only(X_df, k, f" | {level_name}")
                    zf.writestr(f"{level_name}_k{k}_centroids_overlay.html",
                                fig_co_exp.to_html(include_plotlyjs="cdn"))
                except Exception as e:
                    export_errors.append(f"{level_name} centroid overlay: {e}")

                # Cluster→ATECO breakdown HTML (with centroid chart on the right)
                step += 1
                export_progress.progress(step / total_steps,
                                          text=f"Cluster→ATECO HTML {level_name}...")
                try:
                    inv_bkd_exp = build_cluster_ateco_breakdown(
                        X_df, df_base, ateco_col, level_name)
                    inv_html = build_cluster_ateco_breakdown_html(
                        inv_bkd_exp, level_name, k,
                        centroid_chart_html=centroid_div,
                    )
                    zf.writestr(f"{level_name}_k{k}_cluster_ateco_breakdown.html", inv_html)
                except Exception as e:
                    export_errors.append(f"{level_name} cluster→ATECO HTML: {e}")

                # Pearson chart (HTML)
                step += 1
                export_progress.progress(step / total_steps,
                                          text=f"Pearson heatmap {level_name}...")
                if k >= 2:
                    try:
                        df_corr, _, _, _ = compute_centroid_pearson(X_df)
                        fig_h = plot_pearson_heatmap(df_corr)
                        zf.writestr(f"{level_name}_k{k}_pearson.html",
                                    fig_h.to_html(include_plotlyjs="cdn"))
                    except Exception as e:
                        export_errors.append(f"{level_name} pearson: {e}")

                # Composition chart (HTML)
                step += 1
                export_progress.progress(step / total_steps,
                                          text=f"Composition chart {level_name}...")
                if len(res.get("codes", [])) >= 2:
                    try:
                        fig_c, _ = plot_cluster_composition(X_df, df_base, ateco_col, level_name)
                        zf.writestr(f"{level_name}_k{k}_composition.html",
                                    fig_c.to_html(include_plotlyjs="cdn"))
                    except Exception as e:
                        export_errors.append(f"{level_name} composition: {e}")

                # Stats HTML
                step += 1
                export_progress.progress(min(step / total_steps, 1.0),
                                          text=f"Stats table {level_name}...")
                try:
                    stats_df = compute_cluster_stats(X_df)
                    global_m = compute_global_metrics(X_df)
                    stats_html = (
                        "<html><head><style>"
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
                        stats_html += (f'<div class="metric">'
                                       f'<div class="value">{mval}</div>'
                                       f'<div class="label">{mname}</div></div>')
                    stats_html += "</div>"
                    stats_html += stats_df.to_html(index=False)
                    stats_html += "</body></html>"
                    zf.writestr(f"{level_name}_k{k}_stats_table.html", stats_html)
                except Exception as e:
                    export_errors.append(f"{level_name} stats table: {e}")

                # Parent composition charts (HTML)
                par_cols_chart = []
                if level_name == "L2" and "ATECO_L1" in df_base.columns:
                    par_cols_chart = [("ATECO_L1", "by_L1")]
                elif level_name == "L3" and "ATECO_L1" in df_base.columns:
                    par_cols_chart = [("ATECO_L1", "by_L1"), ("ATECO_L2", "by_L2")]
                for par_c, suffix in par_cols_chart:
                    try:
                        fig_par, _ = plot_cluster_composition(
                            X_df, df_base, par_c, f"{level_name} → {par_c}")
                        zf.writestr(f"{level_name}_k{k}_composition_{suffix}.html",
                                    fig_par.to_html(include_plotlyjs="cdn"))
                    except Exception as e:
                        export_errors.append(f"{level_name} composition {suffix}: {e}")

                # ATECO → Cluster breakdown HTML (with centroid chart below)
                try:
                    bkd_exp = build_ateco_cluster_breakdown(X_df, df_base, ateco_col, level_name)
                    bkd_html = build_ateco_breakdown_html(
                        bkd_exp, level_name, k,
                        centroid_chart_html=centroid_div,
                    )
                    zf.writestr(f"{level_name}_k{k}_ateco_breakdown.html", bkd_html)
                except Exception as e:
                    export_errors.append(f"{level_name} ateco→cluster breakdown: {e}")

            # Distribution charts (HTML)
            try:
                _df_meas_exp = st.session_state.get("df_meas", pd.DataFrame())
                _tip = st.session_state.get("tip_filter", "All")
                if _tip != "All" and "Tipologia" in _df_meas_exp.columns:
                    _df_meas_exp = _df_meas_exp[_df_meas_exp["Tipologia"] == _tip]
                _df_meas_exp = _df_meas_exp[_df_meas_exp["POD"].isin(df_base["POD"])]
                fig_dist_exp, _ = plot_consumption_distribution_top15(
                    _df_meas_exp, df_base, title_suffix="Export")
                if fig_dist_exp is not None:
                    zf.writestr("distribution_monthly_consumption.html",
                                fig_dist_exp.to_html(include_plotlyjs="cdn"))
            except Exception as e:
                export_errors.append(f"consumption dist: {e}")

            try:
                fig_pie_exp = plot_potcontr_pie(df_base)
                if fig_pie_exp is not None:
                    zf.writestr("distribution_potcontr_pie.html",
                                fig_pie_exp.to_html(include_plotlyjs="cdn"))
            except Exception as e:
                export_errors.append(f"potcontr pie: {e}")

            try:
                fig_stack_exp = plot_potcontr_stacked_bar(df_base, top_n=15)
                if fig_stack_exp is not None:
                    zf.writestr("distribution_potcontr_by_typology.html",
                                fig_stack_exp.to_html(include_plotlyjs="cdn"))
            except Exception as e:
                export_errors.append(f"potcontr stacked: {e}")

            # Monthly breakdown export (HTML)
            _mbkd_lv = st.session_state.get("monthly_bkd_level", "None")
            _mbkd_ck = f"_monthly_bkd_{_mbkd_lv}"
            if _mbkd_lv and _mbkd_lv != "None" and _mbkd_ck in st.session_state:
                _monthly_res_exp = st.session_state[_mbkd_ck]
                for m_exp, res_m_exp in _monthly_res_exp.items():
                    if "error" in res_m_exp:
                        continue
                    X_m_exp = res_m_exp["X_df"]
                    k_m_exp = res_m_exp["k"]
                    mname_exp = res_m_exp["month_name"]
                    try:
                        fig_m_exp = plot_cluster_profiles(
                            X_m_exp, k_m_exp, f" | {_mbkd_lv} — {mname_exp}")
                        fname_exp = f"monthly_{_mbkd_lv}_{m_exp:02d}_{mname_exp}_k{k_m_exp}"
                        zf.writestr(f"{fname_exp}.html",
                                    fig_m_exp.to_html(include_plotlyjs="cdn"))
                    except Exception as e:
                        export_errors.append(f"monthly {mname_exp}: {e}")

            if export_errors:
                err_txt = "EXPORT ISSUES\n" + "=" * 40 + "\n"
                err_txt += "\n".join(f"- {e}" for e in export_errors)
                zf.writestr("_EXPORT_ISSUES.txt", err_txt)

        buf.seek(0)
        export_progress.progress(1.0, text="Export ready!")
        st.session_state["_export_zip"] = buf.getvalue()
        st.session_state["_export_errors"] = export_errors
        export_progress.empty()

    if "_export_zip" in st.session_state:
        _exp_errors = st.session_state.get("_export_errors", [])
        if _exp_errors:
            st.warning(f"{len(_exp_errors)} export issues. See _EXPORT_ISSUES.txt in the ZIP.")
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