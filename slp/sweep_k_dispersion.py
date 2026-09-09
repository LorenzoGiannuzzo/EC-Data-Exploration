"""
Selection of the number of groups on representativeness rather than on internal
validity indices.

For each K the second clustering stage is rerun, the profiles are generated and
the dispersion of the members from their own curve is collected. The sweep
reports the dispersion weighted by the population behind each cell alongside the
unweighted figure, because an unweighted median rewards the small tight groups
that appear as K grows and would place the knee arbitrarily far to the right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

def run_second_stage(features: pd.DataFrame, k: int, n_replicas: int, seed: int):
    #Lorenzo Giannuzzo: replace the body with the call already used in
    #clustering.py. Must return a Series of group labels indexed by point.
    raise NotImplementedError("wire to the second stage in clustering.py")


def compute_dispersion(labels: pd.Series, n_min: int) -> pd.DataFrame:
    #Lorenzo Giannuzzo: replace the body with the call that writes dispersion.csv.
    #Must return the same columns that file already carries, that is group, cell,
    #n_days and the nrmsd quantiles.
    raise NotImplementedError("wire to the dispersion computation")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

@dataclass
class SweepRow:
    k: int
    n_profiles: int
    pods_covered: int
    pods_below_n_min: int
    nrmsd_unweighted: float
    nrmsd_pod_weighted: float
    nrmsd_day_weighted: float
    nrmsd_p95_pod_weighted: float
    worst_cell_p95: float


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    #Lorenzo Giannuzzo: the weighted median is the value at which the cumulative
    #weight crosses one half. It is used instead of the weighted mean because the
    #dispersion distribution is skewed by a handful of very tight cells and a mean
    #would be pulled by them in the same way the unweighted median is.
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float("nan")
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def sweep(
    features: pd.DataFrame,
    k_values: range | list[int],
    n_min: int = 30,
    n_replicas: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    rows: list[SweepRow] = []

    for k in k_values:
        labels = run_second_stage(features, k=k, n_replicas=n_replicas, seed=seed)
        dispersion = compute_dispersion(labels, n_min=n_min)

        sizes = labels.value_counts()
        kept = sizes[sizes >= n_min]

        #Lorenzo Giannuzzo: cells are attributed the population of their group, so
        #that a group of forty points cannot weigh as much as a group of two
        #thousand simply because both are split over the same nine cells.
        pods_per_cell = dispersion["group"].map(sizes).to_numpy(dtype=float)
        days_per_cell = dispersion["n_days"].to_numpy(dtype=float)
        p50 = dispersion["nrmsd_p50"].to_numpy(dtype=float)
        p95 = dispersion["nrmsd_p95"].to_numpy(dtype=float)

        rows.append(
            SweepRow(
                k=k,
                n_profiles=int(len(kept)),
                pods_covered=int(kept.sum()),
                pods_below_n_min=int(sizes.sum() - kept.sum()),
                nrmsd_unweighted=float(np.median(p50)),
                nrmsd_pod_weighted=_weighted_median(p50, pods_per_cell),
                nrmsd_day_weighted=_weighted_median(p50, days_per_cell),
                nrmsd_p95_pod_weighted=_weighted_median(p95, pods_per_cell),
                worst_cell_p95=float(np.max(p95)),
            )
        )

        row = rows[-1]
        print(
            f"K={k:>3}  profiles={row.n_profiles:>3}  "
            f"covered={row.pods_covered:>6,}  "
            f"lost={row.pods_below_n_min:>4}  "
            f"nRMSD unweighted={row.nrmsd_unweighted:.3f}  "
            f"pod-weighted={row.nrmsd_pod_weighted:.3f}"
        )

    return pd.DataFrame([vars(r) for r in rows])


def report_knee(table: pd.DataFrame, column: str = "nrmsd_pod_weighted") -> None:
    #Lorenzo Giannuzzo: the gain per added group. The knee is where this quantity
    #stops being worth the extra profile, and reading it as a marginal gain is more
    #honest than fitting a curvature, since the sweep has fewer than twenty points
    #and any curvature estimate on that grid is dominated by noise.
    values = table[column].to_numpy()
    ks = table["k"].to_numpy()
    gains = -np.diff(values)

    print(f"\nmarginal gain in {column} per added group")
    for i, gain in enumerate(gains):
        share = 100.0 * gain / values[i] if values[i] else float("nan")
        print(f"  {ks[i]:>3} -> {ks[i + 1]:<3}  {gain:+.4f}  ({share:+.1f}%)")

    print(
        "\nunweighted and pod-weighted columns must be read together: a K where "
        "the unweighted figure improves and the weighted one does not is adding "
        "small tight groups without representing more of the population."
    )
