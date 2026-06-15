"""Outlier detection on quarter-hourly load profiles.

Reproduces the legacy `data_dashboard.outliers_detection_tab` algorithm:

    1. Take the per-POD daily profile (96 quarters)
    2. Cluster the PODs with hierarchical single-linkage (Euclidean)
    3. Pick a k between 10 and 30
    4. Cluster sizes below ``threshold`` are flagged as outliers
    5. The PODs in those small clusters are the outlier POD set

Single linkage is intentional: it tends to form one or two large compact
clusters plus several tiny "chains" — the tiny ones are anomalous profiles.
"""

from __future__ import annotations

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

Q_COLS = [f"q{i}" for i in range(1, 97)]


def detect_outliers(
    profiles_normalised: pd.DataFrame,
    k:                   int = 10,
    threshold:           int = 5,
) -> dict:
    """Single-linkage outlier detection.

    Args:
        profiles_normalised: DataFrame indexed by POD, with q1..q96 columns,
            already row-normalised (max or min/max).
        k:        number of clusters to form (10-30 typical).
        threshold: cluster sizes < threshold are flagged as outliers.

    Returns:
        dict with keys:
            cluster_assignments: {pod: cluster_id}
            cluster_sizes:       {cluster_id: n_pods}
            outlier_clusters:    list of cluster ids
            normal_clusters:     list of cluster ids
            outlier_pods:        list of POD identifiers
            outlier_centroids:   {cluster_id: [96-value mean profile]}
            outlier_stds:        {cluster_id: [96-value std profile]}
            n_outliers:          number of outlier PODs
    """
    if profiles_normalised.empty:
        return {
            "cluster_assignments": {}, "cluster_sizes": {},
            "outlier_clusters": [], "normal_clusters": [],
            "outlier_pods": [], "outlier_centroids": {}, "outlier_stds": {},
            "n_outliers": 0,
        }
    n_pods = len(profiles_normalised)
    if n_pods < k:
        raise ValueError(f"Need ≥{k} PODs for outlier detection, got {n_pods}.")

    X = profiles_normalised[Q_COLS].to_numpy(dtype=float)
    Z = linkage(X, method="single", metric="euclidean")
    labels = fcluster(Z, t=k, criterion="maxclust")

    series = pd.Series(labels, index=profiles_normalised.index, name="cluster")
    sizes  = series.value_counts().sort_index().to_dict()
    outlier_clusters = sorted(c for c, n in sizes.items() if n < threshold)
    normal_clusters  = sorted(c for c, n in sizes.items() if n >= threshold)
    outlier_pods     = series[series.isin(outlier_clusters)].index.tolist()

    # Per outlier-cluster: mean + std profile across its PODs
    outlier_centroids: dict[int, list[float]] = {}
    outlier_stds:      dict[int, list[float]] = {}
    if outlier_clusters:
        df = profiles_normalised[Q_COLS].copy()
        df["__cluster"] = series.values
        for cl in outlier_clusters:
            sub = df[df["__cluster"] == cl][Q_COLS]
            if sub.empty:
                continue
            outlier_centroids[int(cl)] = [float(v) for v in sub.mean().values]
            if len(sub) > 1:
                outlier_stds[int(cl)] = [float(v) for v in sub.std().values]
            else:
                outlier_stds[int(cl)] = [0.0] * 96

    return {
        "cluster_assignments": {str(p): int(c) for p, c in series.items()},
        "cluster_sizes":       {int(c): int(n) for c, n in sizes.items()},
        "outlier_clusters":    [int(c) for c in outlier_clusters],
        "normal_clusters":     [int(c) for c in normal_clusters],
        "outlier_pods":        [str(p) for p in outlier_pods],
        "outlier_centroids":   outlier_centroids,
        "outlier_stds":        outlier_stds,
        "n_outliers":          len(outlier_pods),
    }
