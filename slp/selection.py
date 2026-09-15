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

-------------------------------------------------------------------------------
Author:        Lorenzo Giannuzzo
Affiliation:   Politecnico di Torino, Department of Energy (DENERG)
               Energy Center Lab
Contact:       lorenzo.giannuzzo@polito.it

Developed in collaboration with ENEA within the Italian Research on the Electric
System programme (Ricerca di Sistema Elettrico).
-------------------------------------------------------------------------------
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
    """Smallest D beyond which no further codeword reduces the quantization error
    by tol or more, among the D leaving no codeword under min_share of the days.

    The quantization error of nested Ward cuts can only fall as D grows, but its
    relative gains are not monotone: a split of a small form can gain little and
    the next split of a large one gain a lot. Taking the first D whose gain dips
    below tol would stop on such a plateau, so the rule asks that every later gain
    in the range stay below tol as well, which makes the choice independent of
    where a single small step happens to fall.

    Returns the value and the sentence that justifies it, which goes in the run
    summary so that the choice is reportable and not merely made.
    """
    v = val.sort_values("D").reset_index(drop=True)
    ok = v["smallest_share"] >= min_share
    if not ok.any():
        d = int(v.loc[v["quantization_error"].idxmin(), "D"])
        return d, (f"no D leaves every codeword above {min_share:.1%} of the days; "
                   f"D = {d} taken as the smallest error")

    err = v["quantization_error"].to_numpy(dtype=float)
    #Lorenzo Giannuzzo: gain[i] is what the step from D[i-1] to D[i] buys, relative to the
    # error at D[i-1]; the gain of adding one codeword to D[i] is therefore gain[i+1]
    gain = np.full(len(v), np.nan)
    gain[1:] = (err[:-1] - err[1:]) / np.where(err[:-1] > 0, err[:-1], np.nan)
    v["gain_next"] = np.append(gain[1:], np.nan)
    for i in range(len(v) - 1):
        if not ok.iloc[i]:
            continue
        later = gain[i + 1:]
        later = later[np.isfinite(later)]
        if len(later) and np.all(later < tol):
            row = v.iloc[i]
            return int(row["D"]), (
                f"D = {int(row['D'])}: from here to the end of the range no further "
                f"codeword reduces the quantization error by {tol:.0%} or more (largest "
                f"later gain {later.max():.2%}), and the smallest codeword holds "
                f"{row['smallest_share']:.1%} of the days")
    row = v[ok].iloc[-1]
    return int(row["D"]), (
        f"D = {int(row['D'])}: a codeword still buys {tol:.0%} or more near the end of "
        f"the range, so the top of the range is taken and the range should be widened")


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
             threshold: float = 0.65, n_min: int = 30,
             max_small_share: float = 0.01) -> tuple[int, str]:
    """Largest K up to which every partition of the range reproduces itself.

    Stability is the mean ARI between the partitions of two independent subsamples.
    The threshold follows the reading of the ARI proposed by Steinley (2004), where
    0.65 separates moderate from poor recovery. K is the largest value such that the
    partitions at that K and at every smaller K of the range all reach it: a single
    larger K that happens to clear the threshold after coarser ones have failed is a
    fluctuation of the estimate rather than a stable structure, and taking the largest
    such value would reward it.

    The guard is the share of users falling in groups below n_min, not the size of the
    smallest group. A handful of points with a singular pattern separate at every K,
    and the pipeline reports them and excludes them from the metrics: requiring every
    group to clear n_min would let six users out of seven thousand veto the range.
    """
    v = stab.dropna(subset=["stability_mean"]).sort_values("K").reset_index(drop=True)
    if "small_share" in v.columns:
        admissible = v["small_share"] <= max_small_share
    else:
        admissible = v["K"].map(lambda k: sizes_at_K.get(int(k), 0)) >= n_min
    passed = (v["stability_mean"] >= threshold) & admissible
    run = 0
    while run < len(v) and bool(passed.iloc[run]):
        run += 1
    if run:
        row = v.iloc[run - 1]
        if run < len(v):
            stop = v.iloc[run]
            why = (f"{stop['stability_mean']:.2f}" if stop["stability_mean"] < threshold
                   else f"{stop['stability_mean']:.2f}, but more than {max_small_share:.0%} "
                        f"of the users fall in groups below n_min")
            nxt = f"; at K = {int(stop['K'])} the ARI is {why}"
        else:
            nxt = "; the whole range passes, so the range should be widened"
        return int(row["K"]), (
            f"K = {int(row['K'])}: the largest K up to which every partition of the range "
            f"reproduces itself across subsamples at mean ARI of at least {threshold:.2f} "
            f"(ARI {row['stability_mean']:.2f} at the selected K{nxt}), with no more than "
            f"{max_small_share:.0%} of the users in groups below n_min")
    cand = v[admissible]
    if len(cand):
        row = cand.loc[cand["stability_mean"].idxmax()]
        return int(row["K"]), (
            f"K = {int(row['K'])}: no K of the admissible range {int(v['K'].min())}-"
            f"{int(v['K'].max())} reaches the {threshold:.2f} threshold; the most stable "
            f"admissible K is taken, at mean ARI {row['stability_mean']:.2f}, and the value "
            f"is reported against the threshold")
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