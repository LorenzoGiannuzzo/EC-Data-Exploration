"""Choosing D and K from the data rather than by hand (Section 2.3).

The two stages are answering different questions, so they are not selected on the
same quantity.

    D   the dictionary is a vocabulary, and what is asked of a vocabulary is that
        it represent the days: the criterion is the quantization error, the share
        of the variance of the shapes that survives the substitution of each day
        by its codeword. Internal separation indices are reported alongside but
        do not decide, since on a population where one form holds half the days
        they reward a tree that only shaves outliers off the bulk.

    K   the groups are a partition of the users, and what is asked of a partition
        is that it be a property of the population rather than of the sample:
        the criterion is stability, the agreement between the partitions obtained
        on two independent subsamples. A partition reading sampling noise does not
        reproduce itself, which is exactly what the silhouette cannot see on
        sparse compositional vectors, where it rewards the pattern of empty
        coordinates.

A third question is asked after the profiles exist, in Section 2.4: whether two
profiles are far enough apart to be told apart at all, given how far their own
members lie from them. That is what merge_map answers, and it is the one that
decides how many standard profiles are actually being delivered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score


# ── stage one, the dictionary ────────────────────────────────────────────────
def quantization(points: np.ndarray, weights: np.ndarray,
                 labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted within-codeword inertia over the total, and the share of the
    days each codeword holds.

    The weights are the number of shapes standing behind each micro-cluster, so
    the error is expressed per day and not per micro-cluster, which would let a
    codeword holding forty days count as much as one holding a million.
    """
    x = np.asarray(points, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    grand = np.average(x, axis=0, weights=w)
    ss_tot = (w[:, None] * (x - grand) ** 2).sum()

    ss_within = 0.0
    shares = []
    for k in np.unique(labels):
        m = labels == k
        c = np.average(x[m], axis=0, weights=w[m])
        ss_within += (w[m, None] * (x[m] - c) ** 2).sum()
        shares.append(w[m].sum())

    err = ss_within / ss_tot if ss_tot > 0 else np.nan
    return float(err), np.asarray(shares) / w.sum()


def select_D(val: pd.DataFrame, tol: float = 0.01,
             min_share: float = 0.005) -> tuple[int, str]:
    """Smallest D whose marginal reduction of the quantization error falls below
    tol, among those leaving no codeword under min_share of the days.

    Returns the value and the sentence that justifies it, which goes in the run
    summary so that the choice is reportable and not merely made.
    """
    v = val.sort_values("D").reset_index(drop=True)
    ok = v["smallest_share"] >= min_share
    if not ok.any():
        d = int(v.loc[v["quantization_error"].idxmin(), "D"])
        return d, (f"no D leaves every codeword above {min_share:.1%} of the days; "
                   f"D = {d} taken as the smallest error")

    gain = v["quantization_error"].shift(1).sub(v["quantization_error"]).div(
        v["quantization_error"].shift(1))
    hit = v[(gain < tol) & ok]
    if len(hit):
        row = hit.iloc[0]
        return int(row["D"]), (
            f"D = {int(row['D'])}: the marginal gain of one more codeword falls "
            f"to {gain.loc[row.name]:.1%}, below the {tol:.0%} threshold, and the "
            f"smallest codeword still holds {row['smallest_share']:.1%} of the days")
    row = v[ok].iloc[-1]
    return int(row["D"]), (
        f"D = {int(row['D'])}: the error is still falling by more than {tol:.0%} at "
        f"the end of the range, so the top of the range is taken")


# ── stage two, the users ─────────────────────────────────────────────────────
def stability_table(X: np.ndarray, k_range: tuple[int, int], linkage_fn,
                    n_boot: int = 10, frac: float = 0.8,
                    seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Agreement between the partitions of two independent subsamples, per K.

    The tree of each subsample is built once and cut at every K, so the cost is
    two linkages per replica and not two per replica per K.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    size = int(round(frac * n))
    ks = list(range(k_range[0], k_range[1] + 1))
    scores = {k: [] for k in ks}

    for b in range(n_boot):
        ia = np.sort(rng.choice(n, size=size, replace=False))
        ib = np.sort(rng.choice(n, size=size, replace=False))
        shared = np.intersect1d(ia, ib)
        if len(shared) < 50:
            continue
        za, zb = linkage_fn(X[ia]), linkage_fn(X[ib])
        pa = {u: j for j, u in enumerate(ia)}
        pb = {u: j for j, u in enumerate(ib)}
        ja = np.array([pa[u] for u in shared])
        jb = np.array([pb[u] for u in shared])
        for k in ks:
            la = fcluster(za, k, criterion="maxclust")[ja]
            lb = fcluster(zb, k, criterion="maxclust")[jb]
            scores[k].append(adjusted_rand_score(la, lb))
        if verbose:
            print(f"      replica {b + 1}/{n_boot}")

    return pd.DataFrame({
        "K": ks,
        "stability_mean": [float(np.mean(scores[k])) if scores[k] else np.nan for k in ks],
        "stability_min": [float(np.min(scores[k])) if scores[k] else np.nan for k in ks],
    })


def select_K(stab: pd.DataFrame, sizes_at_K: dict[int, int],
             threshold: float = 0.75, n_min: int = 30,
             max_small_share: float = 0.01) -> tuple[int, str]:
    """Largest K that stays at or above the stability threshold without
    fragmenting the population.

    The guard is the share of users falling in groups below n_min, not the size
    of the smallest group. A handful of points with a genuinely singular pattern
    separate at every K, and the pipeline already reports them and excludes them
    from the metrics: requiring every group to clear n_min would let six users
    out of seven thousand veto the whole range, which is what happened on the
    first run of this criterion.
    """
    v = stab.dropna(subset=["stability_mean"]).sort_values("K")
    if "small_share" in v.columns:
        v = v[v["small_share"] <= max_small_share]
    else:
        v = v[v["K"].map(lambda k: sizes_at_K.get(int(k), 0)) >= n_min]
    ok = v[v["stability_mean"] >= threshold]
    if len(ok):
        row = ok.iloc[-1]
        return int(row["K"]), (
            f"K = {int(row['K'])}: the largest number of groups whose partition "
            f"reproduces itself across subsamples at ARI {row['stability_mean']:.2f}, "
            f"at or above the {threshold:.2f} threshold, and with no more than "
            f"{max_small_share:.0%} of the users left in groups below n_min")
    if len(v):
        row = v.loc[v["stability_mean"].idxmax()]
        return int(row["K"]), (
            f"K = {int(row['K'])}: no K reaches the {threshold:.2f} stability "
            f"threshold; the most stable of the range is taken, at "
            f"ARI {row['stability_mean']:.2f}, and the weakness is reported")
    return int(stab["K"].min()), "no K satisfies n_min; the bottom of the range is taken"


# ── usable K, the profiles a reader can actually tell apart ──────────────────
def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.sum() <= 0 or b.sum() <= 0:
        return np.nan
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def profile_distance(curves_a: dict, curves_b: dict, cell_weights: dict) -> float:
    """Distance between two delivered profiles, averaged over the cells of the
    grid and weighted by the share of the year each cell carries, so that a cell
    holding three percent of the year cannot drive the comparison."""
    cells = [c for c in cell_weights if c in curves_a and c in curves_b]
    tot = sum(cell_weights[c] for c in cells)
    if not cells or tot <= 0:
        return np.nan
    return float(sum(cell_weights[c] * total_variation(curves_a[c], curves_b[c])
                     for c in cells) / tot)


def merge_map(profile_curves: dict, cell_weights: dict, intra_tv: dict,
              ratio: float = 1.0) -> tuple[list[tuple], pd.DataFrame]:
    """Pairs of profiles closer to each other than to their own members.

    Two profiles are kept apart only when the distance between them exceeds
    `ratio` times the smaller of their two internal dispersions, measured in the
    same metric. Below that, the difference between them is smaller than the
    error each of them already makes, and delivering both claims a distinction
    the data does not support.
    """
    ids = sorted(profile_curves)
    pairs, rows = [], []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = profile_distance(profile_curves[a], profile_curves[b], cell_weights)
            floor = ratio * min(intra_tv.get(a, np.nan), intra_tv.get(b, np.nan))
            merge = bool(np.isfinite(d) and np.isfinite(floor) and d < floor)
            rows.append({"a": a, "b": b, "distance": d, "floor": floor, "merge": merge})
            if merge:
                pairs.append((a, b))
    return pairs, pd.DataFrame(rows)