"""Hierarchical clustering on quarter-hourly load profiles.

Mirrors the algorithms used in the legacy `data_dashboard.py`:
    - scipy.cluster.hierarchy linkage + fcluster
    - cluster centroids
    - cluster-quality metrics (silhouette, calinski-harabasz, davies-bouldin)
    - centroid pearson correlation matrix
    - ATECO ↔ cluster cross-tabulation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import pearsonr
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)

Q_COLS = [f"q{i}" for i in range(1, 97)]


# ── Core clustering call ─────────────────────────────────────────────────────
def cluster_profiles(
    profiles_normalised: pd.DataFrame,
    n_clusters: int,
    method:     str = "ward",
    metric:     str = "euclidean",
) -> pd.Series:
    """Hierarchical clustering on a normalised profile DataFrame.

    Returns a Series mapping ``pod → cluster_label`` (labels are 1-indexed,
    consistent with scipy's fcluster output).
    """
    if profiles_normalised.empty:
        return pd.Series(dtype=int, name="cluster")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be ≥ 1, got {n_clusters}")
    n_pods = len(profiles_normalised)
    if n_clusters > n_pods:
        raise ValueError(
            f"n_clusters={n_clusters} > n_pods={n_pods}; cannot form that many clusters"
        )

    X = profiles_normalised[Q_COLS].to_numpy(dtype=float)
    # Ward requires Euclidean — guard upfront for clarity
    if method == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage only supports the Euclidean metric")
    Z = linkage(X, method=method, metric=metric)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return pd.Series(labels, index=profiles_normalised.index, name="cluster")


# ── Auto-k selection (legacy `find_optimal_k`) ───────────────────────────────
def find_optimal_k(
    profiles_normalised: pd.DataFrame,
    method:  str = "average",
    metric:  str = "euclidean",
    k_range: range | None = None,
) -> tuple[int, dict]:
    """Pick the number of clusters via the legacy dashboard's voting scheme.

    Scans ``k_range`` (default: 3 .. min(10, max(3, √n))) on a single linkage
    matrix and lets four selectors vote: max silhouette, max Calinski-Harabasz,
    min Davies-Bouldin, and the elbow on within-cluster inertia. Ties are
    broken in favour of the elbow pick, then the smallest k. The legacy Gap
    Statistic vote is intentionally omitted server-side (it requires ~10
    reference re-clusterings per k and would dominate API response time).

    Returns ``(optimal_k, details)`` where details holds per-k metrics,
    the votes and each selector's pick.
    """
    X = profiles_normalised[Q_COLS].to_numpy(dtype=float)
    n = len(X)
    if n < 4:
        return min(3, max(1, n)), {}
    if k_range is None:
        max_k = min(10, max(3, int(np.sqrt(n))))
        k_range = range(3, max_k + 1)

    Z = linkage(X, method=method, metric=metric)
    results: dict[int, dict] = {}
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        if not 2 <= len(set(labels)) <= n - 1:
            continue
        entry: dict = {"sil": None, "ch": None, "db": None, "inertia": None}
        try:
            entry["sil"] = float(silhouette_score(X, labels))
        except Exception:
            pass
        try:
            entry["ch"] = float(calinski_harabasz_score(X, labels))
        except Exception:
            pass
        try:
            entry["db"] = float(davies_bouldin_score(X, labels))
        except Exception:
            pass
        inertia = 0.0
        for c in set(labels):
            members  = X[labels == c]
            centroid = members.mean(axis=0)
            inertia += float(((members - centroid) ** 2).sum())
        entry["inertia"] = inertia
        results[k] = entry

    votes: dict[int, int] = {k: 0 for k in results}
    picks: dict[str, int] = {}
    sil = {k: v["sil"] for k, v in results.items() if v["sil"] is not None}
    if sil:
        best = max(sil, key=sil.get); votes[best] += 1
        picks["Silhouette"] = best
    ch = {k: v["ch"] for k, v in results.items() if v["ch"] is not None}
    if ch:
        best = max(ch, key=ch.get); votes[best] += 1
        picks["Calinski-Harabasz"] = best
    db = {k: v["db"] for k, v in results.items() if v["db"] is not None}
    if db:
        best = min(db, key=db.get); votes[best] += 1
        picks["Davies-Bouldin"] = best

    elbow_k: int | None = None
    inert = {k: v["inertia"] for k, v in results.items() if v["inertia"] is not None}
    if len(inert) >= 3:
        ks = sorted(inert)
        drops = {
            ks[i]: (inert[ks[i - 1]] - inert[ks[i]]) / inert[ks[i - 1]]
            for i in range(1, len(ks)) if inert[ks[i - 1]] > 0
        }
        if drops:
            avg_drop = float(np.mean(list(drops.values())))
            elbow_k = next((k for k in sorted(drops) if drops[k] < avg_drop),
                           sorted(drops)[0])
            votes[elbow_k] += 1
            picks["Elbow"] = elbow_k

    if votes and max(votes.values()) > 0:
        top = max(votes.values())
        candidates = [k for k, v in votes.items() if v == top]
        if len(candidates) == 1:
            optimal_k = candidates[0]
        elif elbow_k is not None and elbow_k in candidates:
            optimal_k = elbow_k
        else:
            optimal_k = min(candidates)
    else:
        optimal_k = 3

    return optimal_k, {"votes": votes, "method_picks": picks,
                       "metrics": results}


# ── Centroids ─────────────────────────────────────────────────────────────────
def _smooth_quarter_anomalies(
    centroids: pd.DataFrame,
    rel_threshold: float = 0.30,
) -> pd.DataFrame:
    """Interpolate sub-quarter values that drop far below their neighbours.

    Italian DSO measurements store the daily profile in 96 quarter-hour
    columns (q1..q96). On the spring DST switch the 02:00-02:59 hour does
    not exist, so q9..q12 are often filed as 0 for that day; on the autumn
    switch the same hour happens twice and gets summed, producing an
    inflated value. Once averaged across the year the artefact is small,
    but for some datasets it shows up as a sharp V-shaped dip around 02:00
    in every centroid — physically implausible.

    Detection — for each cluster centroid and each quarter q:
        v_q < ``rel_threshold`` × mean(8 nearest non-self neighbours)
    where the neighbours are the 4 quarters before and 4 after q (clipped
    to [0, 95]). When the test fires we replace v_q with the local mean.
    The threshold is conservative (30 %) and only catches genuine dips —
    it leaves quiet-night values from real low-consumption clusters alone
    when the surrounding hours are also quiet.
    """
    if centroids.empty:
        return centroids
    out = centroids.copy()
    arr = out[Q_COLS].to_numpy(dtype=float, copy=True)
    n_clusters, n_q = arr.shape  # n_q == 96
    smoothed_any = False
    for ci in range(n_clusters):
        row = arr[ci]
        for q in range(n_q):
            lo, hi = max(0, q - 4), min(n_q, q + 5)
            neighbours = np.concatenate((row[lo:q], row[q + 1:hi]))
            if neighbours.size == 0:
                continue
            local_mean = float(neighbours.mean())
            if local_mean <= 0:
                continue
            if row[q] < rel_threshold * local_mean:
                arr[ci, q] = local_mean
                smoothed_any = True
    if smoothed_any:
        out[Q_COLS] = arr
    return out


def compute_centroids(
    profiles_normalised: pd.DataFrame,
    clusters:            pd.Series,
) -> pd.DataFrame:
    """Mean profile of each cluster (one row per cluster, q1..q96 columns).

    Applies :func:`_smooth_quarter_anomalies` to the centroids before
    returning, so DST-related spikes/drops around 02:00 (and any other
    quarter where the value is implausibly low compared to its
    8-neighbour mean) are interpolated away.
    """
    if profiles_normalised.empty or clusters.empty:
        return pd.DataFrame(columns=Q_COLS)
    df = profiles_normalised[Q_COLS].copy()
    df["_cluster"] = clusters
    centroids = df.groupby("_cluster", observed=True)[Q_COLS].mean()
    centroids.index.name = "cluster"
    return _smooth_quarter_anomalies(centroids)


# ── Cluster metrics ───────────────────────────────────────────────────────────
def cluster_metrics(
    profiles_normalised: pd.DataFrame,
    clusters:            pd.Series,
) -> dict:
    """Silhouette, Calinski-Harabasz, Davies-Bouldin, sizes — used to judge
    whether a given ``k`` is sensible. All three quality scores are computed
    only when 2 ≤ n_clusters < n_pods."""
    if profiles_normalised.empty or clusters.empty:
        return {"n_pods": 0, "n_clusters": 0}

    sizes = clusters.value_counts().sort_index().to_dict()
    out = {
        "n_pods":     int(len(clusters)),
        "n_clusters": int(clusters.nunique()),
        "sizes":      {int(k): int(v) for k, v in sizes.items()},
        "silhouette":         None,
        "calinski_harabasz":  None,
        "davies_bouldin":     None,
    }

    if 2 <= out["n_clusters"] < out["n_pods"]:
        X = profiles_normalised[Q_COLS].to_numpy(dtype=float)
        lbl = clusters.to_numpy()
        try:
            out["silhouette"]        = float(silhouette_score(X, lbl))
        except Exception:
            pass
        try:
            out["calinski_harabasz"] = float(calinski_harabasz_score(X, lbl))
        except Exception:
            pass
        try:
            out["davies_bouldin"]    = float(davies_bouldin_score(X, lbl))
        except Exception:
            pass

    return out


# ── Pearson correlation matrix between centroids ─────────────────────────────
def centroid_pearson_matrix(centroids: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson r between every pair of cluster centroids."""
    if centroids.empty:
        return pd.DataFrame()
    cls = list(centroids.index)
    n   = len(cls)
    mat = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                r, _ = pearsonr(centroids.iloc[i].values, centroids.iloc[j].values)
            except Exception:
                r = np.nan
            mat[i, j] = mat[j, i] = r
    return pd.DataFrame(mat, index=cls, columns=cls)


# ── ATECO ↔ cluster cross-tab ─────────────────────────────────────────────────
def cluster_ateco_breakdown(
    clusters:    pd.Series,
    metadata:    pd.DataFrame,
    ateco_level: int = 1,
) -> pd.DataFrame:
    """How is each cluster composed in terms of ATECO codes?

    Returns a long DataFrame with one row per (cluster, ateco_code) combination
    and columns: cluster, ateco, n_pods, pct_of_cluster.
    """
    if clusters.empty or metadata.empty:
        return pd.DataFrame(columns=["cluster", "ateco", "n_pods", "pct_of_cluster"])
    ateco_col = f"ateco_l{ateco_level}"
    if ateco_col not in metadata.columns:
        raise ValueError(f"Metadata has no column {ateco_col!r}")

    df = (
        clusters.rename("cluster").rename_axis("pod").reset_index()
        .merge(metadata[["pod", ateco_col]], on="pod", how="left")
        .rename(columns={ateco_col: "ateco"})
    )
    cross = (
        df.groupby(["cluster", "ateco"], observed=True)["pod"]
        .nunique().reset_index().rename(columns={"pod": "n_pods"})
    )
    totals = (
        df.groupby("cluster", observed=True)["pod"]
        .nunique().rename("cluster_total")
    )
    cross = cross.merge(totals, on="cluster", how="left")
    cross["pct_of_cluster"] = (cross["n_pods"] / cross["cluster_total"] * 100).round(2)
    cross = cross.drop(columns=["cluster_total"]).sort_values(
        ["cluster", "n_pods"], ascending=[True, False]
    ).reset_index(drop=True)
    return cross


def dominant_ateco_per_cluster(
    breakdown: pd.DataFrame,
    top_n:     int = 3,
) -> pd.DataFrame:
    """For each cluster keep only the top-N ATECO codes by membership."""
    if breakdown.empty:
        return breakdown
    return (
        breakdown.groupby("cluster", as_index=False, group_keys=False)
        .head(top_n).reset_index(drop=True)
    )
