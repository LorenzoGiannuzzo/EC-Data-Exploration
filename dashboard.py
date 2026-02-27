"""
================================================================================
DATA DASHBOARD - Hierarchical Exploration of Electrical Consumption
================================================================================
Author: Lorenzo Giannuzzo - Energy Center Lab, DENERG, Politecnico di Torino

Launch:  streamlit run dashboard_ateco.py
================================================================================
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

COLORS_PLOTLY = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1

# ==============================================================================
# ATECO LEGEND
# ==============================================================================

# Distributor-specific codes (not in official ATECO classification)
DISTRIBUTOR_CODES = {
    "DO": "Domestic / Residential",
    "DO.01": "Domestic - Resident (primary residence)",
    "DO.02": "Domestic - Non-Resident (secondary/vacation home)",
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

# Global lookup dict: filled at startup from ATECO 2025 Excel + distributor codes
ATECO_LOOKUP: dict[str, str] = {}

ATECO_EXCEL_NAME = "Note-esplicative-ATECO-2025-italiano-inglese.xlsx"


def load_ateco_classification():
    """Load ATECO 2025 classification from Excel file in data/ folder.
    Returns dict {code: description} using Italian titles."""
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
        except Exception as e:
            pass  # fallback to distributor codes only

    # Distributor codes override / supplement
    for code, desc in DISTRIBUTOR_CODES.items():
        if code not in lookup:
            lookup[code] = desc

    ATECO_LOOKUP = lookup
    return lookup


# ==============================================================================
# UTILITY
# ==============================================================================

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


def lookup_ateco_description(code: str) -> str:
    """Lookup description for any ATECO code using the loaded classification."""
    if not code or code == "N/A":
        return ""
    code = str(code).strip()

    # Direct match (works for all levels: A, 01, 01.1, 01.11, DO, DO.01, etc.)
    if code in ATECO_LOOKUP:
        return ATECO_LOOKUP[code]

    # Try without trailing zeros: 01.10.00 -> 01.10 -> 01.1
    parts = code.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in ATECO_LOOKUP:
            return ATECO_LOOKUP[candidate]

    # Try just the first part (division code)
    if parts[0] in ATECO_LOOKUP:
        return ATECO_LOOKUP[parts[0]]

    return ""


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_all_data():
    if not DATA_DIR.exists():
        return pd.DataFrame(), pd.DataFrame(), ["data/ folder not found"]

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
        return pd.DataFrame(), pd.DataFrame(), ["No valid directories found"]

    meta_frames, meas_frames = [], []
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
                cols = ["DataMisura", "POD"] + available_q
                df_ms = df_ms[cols].copy()
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

    progress_bar.progress(1.0, text="Loading complete!")
    status_text.empty()
    progress_bar.empty()

    return df_meta, df_meas, issues


@st.cache_data(show_spinner=False)
def prepare_metadata(df_meta):
    df_unique = (
        df_meta.sort_values("Periodo")
        .drop_duplicates(subset=["POD"], keep="last").copy()
    )
    if "CCATETE" not in df_unique.columns:
        return df_unique, False
    ateco_parsed = df_unique["CCATETE"].apply(parse_ateco).apply(pd.Series)
    df_unique = pd.concat([df_unique, ateco_parsed], axis=1)
    for col in ["ATECO_L1", "ATECO_L2", "ATECO_L3"]:
        df_unique[col] = df_unique[col].fillna("N/A")
    return df_unique, True


@st.cache_data(show_spinner=False)
def compute_pods_with_12_months(_df_meas_periodi):
    """Return set of PODs having 12+ unique months of data."""
    pod_months = _df_meas_periodi.groupby("POD")["Periodo"].nunique()
    return set(pod_months[pod_months >= 12].index)


# ==============================================================================
# DAILY LOAD PROFILE COMPUTATION
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

    # Memory-efficient: add Month column in-place, no full copy
    month_col = df_meas["DataMisura"].dt.month

    if _prog:
        _prog.progress(0.15, text="Averaging daily profiles per POD per month (15%)...")

    # Groupby directly on original data — no 6 GiB copy needed
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

    min_non_nan = 6 * n_q
    unstacked = unstacked.dropna(thresh=min_non_nan)

    if _prog:
        _prog.progress(0.60, text="Interpolating missing values (60%)...")

    profile_raw = unstacked.interpolate(axis=1, limit_direction="both")

    if _prog:
        _prog.progress(0.80, text="Normalizing profiles (min-max per month) (80%)...")

    # Min-Max normalization per POD per MONTH:
    # For each month's 96 Q-values, independently map min->0, max->1.
    # This highlights intra-day pattern within each month,
    # removing cross-month seasonal amplitude differences.
    profile_norm = profile_raw.copy()
    for m in range(1, 13):
        prefix = f"M{m:02d}_"
        m_cols = [c for c in profile_norm.columns if c.startswith(prefix)]
        if not m_cols:
            continue
        m_data = profile_norm[m_cols]
        m_min = m_data.min(axis=1)
        m_max = m_data.max(axis=1)
        m_range = (m_max - m_min).replace(0, np.nan)
        profile_norm[m_cols] = m_data.sub(m_min, axis=0).div(m_range, axis=0)
    profile_norm = profile_norm.dropna()

    if _prog:
        _prog.progress(1.0, text="Profiles complete!")
        _prog.empty()

    return profile_norm, profile_raw


def extract_month_profile(profile_df, month: int):
    prefix = f"M{month:02d}_"
    cols = [c for c in profile_df.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame()
    sub = profile_df[cols].copy()
    sub.columns = [c.replace(prefix, "") for c in cols]
    return sub


def get_overall_avg_profile(profile_df):
    all_months_data = []
    for m in range(1, 13):
        mdf = extract_month_profile(profile_df, m)
        if not mdf.empty:
            all_months_data.append(mdf)
    if not all_months_data:
        return pd.DataFrame()
    stacked = pd.concat(all_months_data, keys=range(len(all_months_data)))
    return stacked.groupby(level=1).mean()


# ==============================================================================
# CLUSTERING
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


def run_clustering(profile_norm, pods_subset, cluster_month=None):
    available = profile_norm.index.intersection(pods_subset)
    sub = profile_norm.loc[available]

    if cluster_month is not None:
        X_df = extract_month_profile(sub, cluster_month)
    else:
        X_df = get_overall_avg_profile(sub)

    X_df = X_df.dropna()
    if len(X_df) < 5:
        return None, None, None, None, f"Not enough PODs with profile data ({len(X_df)}). Need at least 5."

    X = X_df.values
    Z = linkage(X, method="average", metric="euclidean")
    max_k = min(10, max(3, int(np.sqrt(len(X)))))
    k_range = range(3, max_k + 1)
    optimal_k, details = find_optimal_k(X, Z, k_range)
    labels = fcluster(Z, t=optimal_k, criterion="maxclust")
    X_df = X_df.copy()
    X_df["Cluster"] = labels
    return X_df, Z, optimal_k, details, None


# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def show_bar_chart(df_unique, level_col, title):
    """Horizontal bar chart -- avoids overlapping labels."""
    counts = (
        df_unique.groupby(level_col)["POD"].nunique().reset_index()
        .rename(columns={"POD": "N_POD"})
        .sort_values("N_POD", ascending=True)
    )
    counts["Pct"] = (counts["N_POD"] / counts["N_POD"].sum() * 100).round(1)
    counts["Label"] = counts.apply(
        lambda r: f"{r['N_POD']:,} ({r['Pct']:.1f}%)", axis=1
    )
    counts["Desc"] = counts[level_col].apply(lookup_ateco_description)
    counts["DisplayLabel"] = counts.apply(
        lambda r: f"{r[level_col]} - {r['Desc'][:30]}" if r["Desc"] else str(r[level_col]),
        axis=1
    )

    n_cats = len(counts)
    bar_height = max(350, n_cats * 22)

    fig = px.bar(
        counts, y="DisplayLabel", x="N_POD",
        text="Label",
        title=title,
        color=level_col,
        color_discrete_sequence=COLORS_PLOTLY,
        orientation="h",
    )
    fig.update_layout(
        xaxis_title="Number of PODs",
        yaxis_title="",
        showlegend=False,
        height=bar_height,
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=10),
    )
    fig.update_traces(textposition="outside")
    return fig, counts


def show_dendrogram(Z, labels, optimal_k, max_display=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    truncate_mode = "lastp" if n > max_display else None
    p = min(max_display, n)

    fig_mpl, ax = plt.subplots(figsize=(14, 6))
    cut_height = Z[-(optimal_k - 1), 2] if optimal_k > 1 and len(Z) >= optimal_k else None
    dendrogram(
        Z, ax=ax, truncate_mode=truncate_mode, p=p,
        color_threshold=cut_height,
        leaf_rotation=90, leaf_font_size=7, no_labels=(n > max_display),
    )
    ax.set_title(f"Dendrogram (Average Linkage) - {n} PODs, k={optimal_k}")
    ax.set_xlabel("POD" if n <= max_display else f"Clusters ({p} leaf nodes)")
    ax.set_ylabel("Distance")
    if cut_height is not None:
        ax.axhline(y=cut_height, color="red", linestyle="--", alpha=0.7,
                    label=f"Cut k={optimal_k}")
        ax.legend()
    fig_mpl.tight_layout()
    return fig_mpl


def show_cluster_profiles(X_df, optimal_k, q_labels):
    q_cols = [c for c in X_df.columns if c.startswith("Q")]
    fig = go.Figure()
    for cl in sorted(X_df["Cluster"].unique()):
        cluster_data = X_df[X_df["Cluster"] == cl][q_cols]
        mean_profile = cluster_data.mean()
        std_profile = cluster_data.std()
        n_members = len(cluster_data)
        x_labels = q_labels[:len(q_cols)]

        fig.add_trace(go.Scatter(
            x=x_labels, y=mean_profile, mode="lines",
            name=f"Cluster {cl} (n={n_members})", line=dict(width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=x_labels + x_labels[::-1],
            y=list((mean_profile + std_profile).clip(0, 1)) +
              list((mean_profile - std_profile).clip(0, 1))[::-1],
            fill="toself", opacity=0.12, line=dict(width=0),
            showlegend=False, name=f"+/-1s Cluster {cl}",
        ))
    fig.update_layout(
        title=f"Average Daily Load Profile per Cluster (k={optimal_k})",
        xaxis_title="Time of Day",
        yaxis_title="Normalized Consumption (0-1, min-max)",
        height=500,
        yaxis=dict(range=[-0.05, 1.05]),
        xaxis=dict(dtick=4, tickangle=-45),
    )
    return fig


def show_voting_chart(details):
    votes = details["votes"]
    method_picks = details["method_picks"]
    vote_df = pd.DataFrame([
        {"k": k, "Votes": v} for k, v in votes.items() if v > 0
    ]).sort_values("k")
    if vote_df.empty:
        return None, None
    fig = px.bar(
        vote_df, x="k", y="Votes", text="Votes",
        title="Votes for Number of Clusters (k)",
        color="Votes", color_continuous_scale="Blues",
    )
    fig.update_layout(
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Votes (out of 5 methods)",
        height=350, xaxis=dict(dtick=1),
    )
    fig.update_traces(textposition="outside")
    method_df = pd.DataFrame([
        {"Method": m, "k chosen": k} for m, k in method_picks.items()
    ])
    return fig, method_df


# ==============================================================================
# ATECO LEGEND TABLE
# ==============================================================================

def build_ateco_legend(df_unique):
    """Build a comprehensive ATECO legend from the actual data + known descriptions."""
    rows = []
    for _, row in df_unique.iterrows():
        for lcol in ["ATECO_L1", "ATECO_L2", "ATECO_L3"]:
            code = row.get(lcol, "N/A")
            if code and code != "N/A":
                rows.append({"Code": code, "Level": lcol.replace("ATECO_", "")})

    if not rows:
        return pd.DataFrame()

    legend = pd.DataFrame(rows).drop_duplicates(subset=["Code"]).sort_values("Code")
    legend["Description"] = legend["Code"].apply(lookup_ateco_description)

    pod_counts = {}
    for lcol in ["ATECO_L1", "ATECO_L2", "ATECO_L3"]:
        counts = df_unique.groupby(lcol)["POD"].nunique()
        for code, n in counts.items():
            if code != "N/A":
                pod_counts[code] = n

    legend["N_POD"] = legend["Code"].map(pod_counts).fillna(0).astype(int)
    legend = legend.sort_values(["Level", "Code"]).reset_index(drop=True)
    return legend


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Data Dashboard - Electrical Consumption",
        page_icon="zap",
        layout="wide",
    )

    st.title("Data Dashboard - Electrical Consumption Data Exploration")
    st.caption("Energy Center Lab, DENERG, Politecnico di Torino")

    # --- Load ATECO 2025 classification ---
    if "ateco_loaded" not in st.session_state:
        load_ateco_classification()
        st.session_state["ateco_loaded"] = True
        if len(ATECO_LOOKUP) <= len(DISTRIBUTOR_CODES):
            st.warning(
                f"ATECO 2025 classification file not found in `data/{ATECO_EXCEL_NAME}`. "
                "Using distributor codes only. Place the Excel file in the data/ folder "
                "and reload."
            )
    elif not ATECO_LOOKUP:
        load_ateco_classification()

    # --- Data Loading ---
    if "data_loaded" not in st.session_state:
        df_meta, df_meas, issues = load_all_data()
        st.session_state["df_meta"] = df_meta
        st.session_state["df_meas"] = df_meas
        st.session_state["issues"] = issues
        st.session_state["data_loaded"] = True
    else:
        df_meta = st.session_state["df_meta"]
        df_meas = st.session_state["df_meas"]
        issues = st.session_state["issues"]

    if df_meta.empty:
        st.error("No data found in data/ folder.")
        if issues:
            with st.expander("Error details"):
                for iss in issues:
                    st.text(iss)
        return

    df_unique, has_ateco = prepare_metadata(df_meta)

    if not has_ateco:
        st.error("CCATETE column not found in metadata.")
        st.info(f"Columns found: {list(df_unique.columns)}")
        return

    # --- Profile computation ---
    if "profile_norm" not in st.session_state:
        prog = st.progress(0, text="Computing daily load profiles (0%)...")
        profile_norm, profile_raw = compute_daily_profiles(df_meas, prog)
        st.session_state["profile_norm"] = profile_norm
        st.session_state["profile_raw"] = profile_raw
    else:
        profile_norm = st.session_state["profile_norm"]
        profile_raw = st.session_state["profile_raw"]

    # --- Compute 12+ month PODs ---
    if "pods_12m" not in st.session_state:
        st.session_state["pods_12m"] = compute_pods_with_12_months(
            df_meas[["POD", "Periodo"]].drop_duplicates()
        )
    pods_12m = st.session_state["pods_12m"]

    # --- Sidebar ---
    with st.sidebar:
        st.header("Global Statistics")

        if st.button("Reload Data"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        n_pod_total = df_unique["POD"].nunique()
        n_pod_meas = df_meas["POD"].nunique()
        n_periodi = df_meas["Periodo"].nunique()
        n_with_profile = len(profile_norm)
        n_12m = len(pods_12m)

        st.metric("Unique PODs (metadata)", f"{n_pod_total:,}")
        st.metric("Unique PODs (measures)", f"{n_pod_meas:,}")
        st.metric("Months in dataset", f"{n_periodi}")
        st.metric("PODs with 12+ months", f"{n_12m:,}")
        st.metric("PODs with load profile", f"{n_with_profile:,}")
        st.metric("ATECO codes loaded", f"{len(ATECO_LOOKUP):,}")

        st.divider()
        st.header("Filters")

        coverage_opt = st.radio(
            "Data coverage",
            ["All PODs", "12+ months only"],
            index=0,
            help="Filter to PODs that have measurements in at least 12 distinct months."
        )
        use_12m_filter = (coverage_opt == "12+ months only")

        st.divider()
        st.header("ATECO Filters")

        if use_12m_filter:
            df_base = df_unique[df_unique["POD"].isin(pods_12m)]
        else:
            df_base = df_unique

        l1_values = sorted(df_base["ATECO_L1"].unique())
        l1_options = ["All"] + [v for v in l1_values if v != "N/A"] + \
                     (["N/A"] if "N/A" in l1_values else [])
        sel_l1 = st.selectbox("Level 1 (Section)", l1_options, key="l1")

        df_l1 = df_base if sel_l1 == "All" else df_base[df_base["ATECO_L1"] == sel_l1]
        l2_values = sorted(df_l1["ATECO_L2"].unique())
        l2_options = ["All"] + [v for v in l2_values if v != "N/A"] + \
                     (["N/A"] if "N/A" in l2_values else [])
        sel_l2 = st.selectbox("Level 2 (Division)", l2_options, key="l2")

        df_l2 = df_l1 if sel_l2 == "All" else df_l1[df_l1["ATECO_L2"] == sel_l2]
        l3_values = sorted(df_l2["ATECO_L3"].unique())
        l3_options = ["All"] + [v for v in l3_values if v != "N/A"] + \
                     (["N/A"] if "N/A" in l3_values else [])
        sel_l3 = st.selectbox("Level 3 (Class)", l3_options, key="l3")

        df_filtered = df_base.copy()
        if sel_l1 != "All":
            df_filtered = df_filtered[df_filtered["ATECO_L1"] == sel_l1]
        if sel_l2 != "All":
            df_filtered = df_filtered[df_filtered["ATECO_L2"] == sel_l2]
        if sel_l3 != "All":
            df_filtered = df_filtered[df_filtered["ATECO_L3"] == sel_l3]

        n_filtered = df_filtered["POD"].nunique()
        st.divider()
        st.metric("PODs in selection", f"{n_filtered:,}")

        if issues:
            with st.expander(f"{len(issues)} warnings"):
                for iss in issues[:20]:
                    st.text(iss)

    # ==========================================================================
    # MAIN AREA
    # ==========================================================================

    tab1, tab2, tab3 = st.tabs([
        "POD Counts by ATECO Level",
        "Load Profile Clustering",
        "ATECO Code Legend",
    ])

    # ==========================================================================
    # TAB 1: Counts
    # ==========================================================================
    with tab1:
        st.subheader("POD Distribution by ATECO Level")

        filter_parts = []
        if use_12m_filter:
            filter_parts.append("12+ months")
        if sel_l1 != "All":
            filter_parts.append(f"L1={sel_l1}")
        if sel_l2 != "All":
            filter_parts.append(f"L2={sel_l2}")
        if sel_l3 != "All":
            filter_parts.append(f"L3={sel_l3}")

        if filter_parts:
            st.info(f"Active filters: {' > '.join(filter_parts)}  |  "
                    f"**{n_filtered:,} PODs**")
        else:
            st.info(f"No filter active  |  **{n_filtered:,} PODs**")

        CHART_HEIGHT = 600

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Level 1 (Section)")
            fig_l1, counts_l1 = show_bar_chart(
                df_filtered, "ATECO_L1", "PODs by ATECO Section"
            )
            with st.container(height=CHART_HEIGHT):
                st.plotly_chart(fig_l1, use_container_width=True)

        with col2:
            st.markdown("#### Level 2 (Division)")
            fig_l2, counts_l2 = show_bar_chart(
                df_filtered, "ATECO_L2", "PODs by ATECO Division"
            )
            with st.container(height=CHART_HEIGHT):
                st.plotly_chart(fig_l2, use_container_width=True)

        with col3:
            st.markdown("#### Level 3 (Class)")
            fig_l3, counts_l3 = show_bar_chart(
                df_filtered, "ATECO_L3", "PODs by ATECO Class"
            )
            with st.container(height=CHART_HEIGHT):
                st.plotly_chart(fig_l3, use_container_width=True)

        # Data tables in separate row below
        st.markdown("---")
        st.markdown("#### Data Tables")
        tcol1, tcol2, tcol3 = st.columns(3)

        with tcol1:
            st.markdown("**Level 1 (Section)**")
            with st.container(height=300):
                st.dataframe(
                    counts_l1[["ATECO_L1", "N_POD", "Pct", "Desc"]]
                    .sort_values("N_POD", ascending=False),
                    hide_index=True, use_container_width=True,
                )

        with tcol2:
            st.markdown("**Level 2 (Division)**")
            with st.container(height=300):
                st.dataframe(
                    counts_l2[["ATECO_L2", "N_POD", "Pct", "Desc"]]
                    .sort_values("N_POD", ascending=False),
                    hide_index=True, use_container_width=True,
                )

        with tcol3:
            st.markdown("**Level 3 (Class)**")
            with st.container(height=300):
                st.dataframe(
                    counts_l3[["ATECO_L3", "N_POD", "Pct", "Desc"]]
                    .sort_values("N_POD", ascending=False),
                    hide_index=True, use_container_width=True,
                )

    # ==========================================================================
    # TAB 2: Clustering
    # ==========================================================================
    with tab2:
        st.subheader("Hierarchical Clustering of Daily Load Profiles")
        st.markdown(
            "Hierarchical clustering with **average linkage** on daily load profiles "
            "(Q1-Q96, 00:00-23:45 in 15-min intervals), averaged per calendar month "
            "and **min-max normalized per month** (each month independently: 0 = lowest, 1 = peak). "
            "Optimal k (min 3) by **majority vote** among 5 methods, "
            "with **Elbow as tie-breaker**."
        )

        pods_in_selection = set(df_filtered["POD"].unique())

        if profile_norm.empty:
            st.warning("No load profiles available.")
        else:
            pods_with_profile = set(profile_norm.index)
            pods_available = pods_in_selection & pods_with_profile
            st.caption(f"PODs in selection: {len(pods_in_selection):,}  |  "
                       f"With load profile: {len(pods_available):,}")

            month_options = ["Overall Average"] + \
                [f"{m:02d} - {MONTH_NAMES[m]}" for m in range(1, 13)]
            sel_month = st.selectbox(
                "Cluster on which month's daily profile?", month_options
            )
            cluster_month = None if sel_month == "Overall Average" \
                else int(sel_month.split(" - ")[0])

            if len(pods_available) < 5:
                st.warning(
                    f"Need at least 5 PODs with load profiles. "
                    f"Currently: {len(pods_available)}."
                )
            else:
                if st.button("Run Clustering", type="primary"):
                    with st.spinner("Clustering in progress..."):
                        X_df, Z, optimal_k, details, error = run_clustering(
                            profile_norm, list(pods_available), cluster_month
                        )

                    if error:
                        st.error(error)
                    else:
                        q_cols = [c for c in X_df.columns if c.startswith("Q")]
                        month_label = "Overall Average" if cluster_month is None \
                            else MONTH_NAMES[cluster_month]

                        st.success(
                            f"Clustering complete! Optimal k = {optimal_k} | "
                            f"{len(X_df)} PODs | Profile: {month_label}"
                        )

                        # --- Voting ---
                        st.markdown("---")
                        st.markdown("### Optimal k Determination")
                        vote_fig, method_df = show_voting_chart(details)
                        col_v1, col_v2 = st.columns([2, 1])
                        with col_v1:
                            if vote_fig:
                                st.plotly_chart(vote_fig, use_container_width=True)
                        with col_v2:
                            st.markdown("**Method choices:**")
                            if method_df is not None and not method_df.empty:
                                st.dataframe(method_df, hide_index=True)
                            st.metric("Optimal k (majority vote)", optimal_k)

                        # --- Cluster profiles (FIRST) ---
                        st.markdown("---")
                        st.markdown("### Average Daily Load Profile per Cluster")
                        fig_profiles = show_cluster_profiles(
                            X_df, optimal_k, Q_TIME_LABELS
                        )
                        st.plotly_chart(fig_profiles, use_container_width=True)

                        # --- Monthly breakdown (only for Overall) ---
                        if cluster_month is None:
                            st.markdown("---")
                            st.markdown("### Monthly Profile Breakdown by Cluster")
                            cluster_labels = X_df[["Cluster"]].copy()
                            months_to_show = [
                                m for m in range(1, 13)
                                if any(c.startswith(f"M{m:02d}_")
                                       for c in profile_norm.columns)
                            ]
                            cols_per_row = 4
                            for row_start in range(0, len(months_to_show), cols_per_row):
                                row_months = months_to_show[row_start:row_start + cols_per_row]
                                cols = st.columns(len(row_months))
                                for col, m in zip(cols, row_months):
                                    with col:
                                        m_df = extract_month_profile(
                                            profile_norm.loc[X_df.index], m
                                        )
                                        if m_df.empty:
                                            continue
                                        m_df["Cluster"] = cluster_labels["Cluster"]
                                        q_cols_m = [c for c in m_df.columns
                                                    if c.startswith("Q")]
                                        fig_m = go.Figure()
                                        for cl in sorted(m_df["Cluster"].unique()):
                                            cl_data = m_df[m_df["Cluster"] == cl][q_cols_m]
                                            mean_p = cl_data.mean()
                                            x_labels = Q_TIME_LABELS[:len(q_cols_m)]
                                            fig_m.add_trace(go.Scatter(
                                                x=x_labels, y=mean_p, mode="lines",
                                                name=f"Cl.{cl} (n={len(cl_data)})",
                                                line=dict(width=2),
                                            ))
                                        fig_m.update_layout(
                                            title=f"{MONTH_NAMES[m]}",
                                            height=250,
                                            margin=dict(t=30, b=20, l=30, r=10),
                                            showlegend=(m == months_to_show[0]),
                                            xaxis=dict(dtick=8, tickangle=-45,
                                                       tickfont=dict(size=7)),
                                            yaxis=dict(range=[-0.05, 1.05],
                                                       tickfont=dict(size=8)),
                                        )
                                        st.plotly_chart(fig_m,
                                                        use_container_width=True)

                        # --- Summary ---
                        st.markdown("---")
                        st.markdown("### Cluster Summary")
                        summary_rows = []
                        for cl in sorted(X_df["Cluster"].unique()):
                            cl_data = X_df[X_df["Cluster"] == cl]
                            profile_mean = cl_data[q_cols].mean()
                            peak_idx = profile_mean.idxmax()
                            trough_idx = profile_mean.idxmin()
                            peak_q = int(peak_idx.replace("Q", "")) - 1
                            trough_q = int(trough_idx.replace("Q", "")) - 1
                            peak_time = Q_TIME_LABELS[peak_q] \
                                if peak_q < len(Q_TIME_LABELS) else "?"
                            trough_time = Q_TIME_LABELS[trough_q] \
                                if trough_q < len(Q_TIME_LABELS) else "?"
                            summary_rows.append({
                                "Cluster": cl,
                                "N. PODs": len(cl_data),
                                "% of total": f"{len(cl_data)/len(X_df)*100:.1f}%",
                                "Peak Time": peak_time,
                                "Trough Time": trough_time,
                                "Max (norm)": f"{profile_mean.max():.3f}",
                                "Min (norm)": f"{profile_mean.min():.3f}",
                            })
                        st.dataframe(pd.DataFrame(summary_rows),
                                     hide_index=True, use_container_width=True)

                        # --- Typology per cluster ---
                        if "D_49DES" in df_unique.columns:
                            st.markdown("### Typology Composition per Cluster")
                            cl_meta = (
                                X_df[["Cluster"]].reset_index()
                                .merge(df_unique[["POD", "D_49DES"]],
                                       on="POD", how="left")
                            )
                            cl_meta["D_49DES"] = cl_meta["D_49DES"].fillna("Unknown")
                            for cl in sorted(X_df["Cluster"].unique()):
                                cl_sub = cl_meta[cl_meta["Cluster"] == cl]
                                tipo_count = (
                                    cl_sub.groupby("D_49DES")["POD"].count()
                                    .reset_index().rename(columns={"POD": "N"})
                                    .sort_values("N", ascending=False)
                                )
                                tipo_count["%"] = (
                                    tipo_count["N"] / tipo_count["N"].sum() * 100
                                ).round(1)
                                with st.expander(f"Cluster {cl} ({len(cl_sub)} PODs)"):
                                    st.dataframe(tipo_count, hide_index=True)

                        # --- Dendrogram (LAST) ---
                        st.markdown("---")
                        st.markdown("### Dendrogram")
                        fig_dendro = show_dendrogram(Z, X_df.index, optimal_k)
                        st.pyplot(fig_dendro)

    # ==========================================================================
    # TAB 3: ATECO Legend
    # ==========================================================================
    with tab3:
        st.subheader("ATECO Code Reference")
        st.markdown(
            "Comprehensive lookup of all ATECO codes found in the dataset, "
            "with their official description (ATECO 2025 / ISTAT classification, "
            "loaded from `Note-esplicative-ATECO-2025-italiano-inglese.xlsx`) "
            "and the number of PODs associated with each code. "
            "Non-ATECO distributor codes (DO, CO, IL) are included separately."
        )

        legend_df = build_ateco_legend(df_unique)

        if legend_df.empty:
            st.warning("No ATECO codes found in the data.")
        else:
            search = st.text_input(
                "Search by code or description",
                placeholder="e.g. 84, DO, manufacturing..."
            )
            if search:
                mask = (
                    legend_df["Code"].str.contains(search, case=False, na=False) |
                    legend_df["Description"].str.contains(search, case=False, na=False)
                )
                display_df = legend_df[mask]
            else:
                display_df = legend_df

            lev_tabs = st.tabs(["All Levels", "L1 - Sections", "L2 - Divisions", "L3 - Classes"])

            with lev_tabs[0]:
                st.dataframe(
                    display_df[["Level", "Code", "Description", "N_POD"]]
                    .sort_values(["Level", "N_POD"], ascending=[True, False])
                    .rename(columns={"N_POD": "PODs"}),
                    hide_index=True,
                    use_container_width=True,
                    height=min(600, 35 * len(display_df) + 40),
                )

            with lev_tabs[1]:
                l1_df = display_df[display_df["Level"] == "L1"].sort_values("N_POD", ascending=False)
                st.dataframe(
                    l1_df[["Code", "Description", "N_POD"]].rename(columns={"N_POD": "PODs"}),
                    hide_index=True, use_container_width=True,
                )

            with lev_tabs[2]:
                l2_df = display_df[display_df["Level"] == "L2"].sort_values("N_POD", ascending=False)
                st.dataframe(
                    l2_df[["Code", "Description", "N_POD"]].rename(columns={"N_POD": "PODs"}),
                    hide_index=True, use_container_width=True,
                )

            with lev_tabs[3]:
                l3_df = display_df[display_df["Level"] == "L3"].sort_values("N_POD", ascending=False)
                st.dataframe(
                    l3_df[["Code", "Description", "N_POD"]].rename(columns={"N_POD": "PODs"}),
                    hide_index=True, use_container_width=True,
                )

            st.caption(
                f"Total unique codes in dataset: {len(legend_df)} "
                f"(L1: {(legend_df['Level']=='L1').sum()}, "
                f"L2: {(legend_df['Level']=='L2').sum()}, "
                f"L3: {(legend_df['Level']=='L3').sum()})"
            )


if __name__ == "__main__":
    main()