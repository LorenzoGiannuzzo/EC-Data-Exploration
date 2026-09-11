"""Stage 2 — Clustering (Section 2.3).

Two clusterings, on two different units of observation.

    Stage 1, the days.  Every POD-day is one observation: a vector of 96 values
    summing to one. Ward on a stratified sample yields D recurring forms, and
    every remaining day is assigned to the nearest of them.

    Stage 2, the users. Every POD is one observation: a vector of D
    energy-weighted frequencies (Eq. 4) plus scale features, CLR-transformed
    (Eq. 5). Ward yields K groups.

What comes out are groups of users, not profiles: a centroid here is a vector of
frequencies, not a curve. The curves are built in Section 2.4.

Outputs
    cache/dictionary.npy            (D, 96) the codewords
    cache/day_codeword.npy          (n_days,) int, -1 where the day has no shape
    cache/user_vectors.parquet      pod, f_0..f_D-1, the scale features
    cache/groups.parquet            pod, group
    paper_results/clustering_results/
        dictionary.csv              the D codewords, with the share of days and energy
        dictionary.png
        validity_D.csv / .png       how D was chosen
        validity_K.csv / .png       how K was chosen
        groups.csv                  the K groups, size and composition
        stability.csv               ARI across replicas, and against BIRCH
        ablation.csv                the two-stage against mean curves
        summary.txt                 [D], [K], [M], [R], [n_min]

Run
    python clustering.py

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

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config

from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

from common.cache import signature as cache_signature
from common.cache import write_manifest
from common.ward import ward_weighted, weighted_centroids
from selection import quantization, select_D, select_K, stability_table


# ── stage 1: the dictionary ──────────────────────────────────────────────────
def to_hourly(shapes, unit_integral: bool = True,
              batch: int = 200_000) -> np.ndarray:
    """Sum the four quarters of each hour, then renormalise.

    A quarter-hourly shape normalised to unit integral is dominated by the
    largest single reading of the day, and at daily resolution that reading is
    very often one appliance rather than an operating pattern: a water heater
    starting at nine in the evening moves the curve further than any difference
    of behaviour, and Euclidean distance over 96 components reads it as the
    principal feature. The dictionary then spends its codewords on the hour at
    which the spike falls. Summing to the hour suppresses that without touching
    anything else, since only the vocabulary is built in this space: the profiles
    of Section 2.4 average the metered quarter-hourly days as before, and the
    codewords themselves are reported at 96 components by the function below.

    Done in batches because the pool is a memory-mapped array of several
    gigabytes and materialising it whole is unnecessary.
    """
    n = len(shapes)
    out = np.empty((n, 24), dtype="float32")
    for i in range(0, n, batch):
        blk = np.asarray(shapes[i:i + batch], dtype="float32")
        h = blk.reshape(len(blk), 24, 4).sum(axis=2)
        if unit_integral:
            t = h.sum(axis=1, keepdims=True)
            h = np.divide(h, t, out=np.zeros_like(h), where=t > 0)
        out[i:i + batch] = h
    return out


def full_resolution_centroids(shapes, code: np.ndarray, D: int,
                              unit_integral: bool = True,
                              batch: int = 200_000) -> np.ndarray:
    """The codewords at 96 components, as the mean of the shapes assigned to them.

    The partition is decided in the hourly space; what is reported, plotted and
    compared against the national profiles is the quarter-hourly mean of the days
    each codeword gathers, so that nothing downstream has to know at which
    resolution the vocabulary was built.
    """
    sums = np.zeros((D, 96), dtype="float64")
    cnt = np.zeros(D, dtype="float64")
    for i in range(0, len(shapes), batch):
        blk = np.asarray(shapes[i:i + batch], dtype="float64")
        c = code[i:i + batch]
        ok = c >= 0
        np.add.at(sums, c[ok], blk[ok])
        np.add.at(cnt, c[ok], 1.0)
    cent = np.divide(sums, cnt[:, None], out=np.zeros_like(sums), where=cnt[:, None] > 0)
    if unit_integral:
        t = cent.sum(axis=1, keepdims=True)
        cent = np.divide(cent, t, out=np.zeros_like(cent), where=t > 0)
    return cent.astype("float32")


def to_monthly(shapes: np.ndarray, days: pd.DataFrame,
               unit_integral: bool) -> tuple[np.ndarray, pd.DataFrame]:
    """Collapse the days of each POD-month into one shape.

    The month shape is the energy-weighted mean of its days, which is the same
    thing as the shape of the month's total curve: a day carrying more energy
    must weigh more, exactly as in Eq. 4. Weekday and weekend end up averaged
    together, which is what this unit costs.
    """
    d = days[days["has_shape"]].copy()
    d["ym"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    key = d["pod"].astype(str) + "|" + d["ym"]
    codes, uniq = pd.factorize(key)

    e = d["energy"].to_numpy()
    sums = np.zeros((len(uniq), 96))
    np.add.at(sums, codes, np.asarray(shapes[d["shape_idx"].to_numpy()]) * e[:, None])
    tot = np.zeros(len(uniq))
    np.add.at(tot, codes, e)
    mshapes = np.divide(sums, tot[:, None], out=np.zeros_like(sums), where=tot[:, None] > 0)
    if unit_integral:
        s = mshapes.sum(axis=1, keepdims=True)
        mshapes = np.divide(mshapes, s, out=np.zeros_like(mshapes), where=s > 0)

    first = d.groupby(codes).first()
    out = pd.DataFrame({
        "pod": [u.split("|")[0] for u in uniq],
        "date": pd.to_datetime([u.split("|")[1] + "-01" for u in uniq]),
        "season": first["season"].to_numpy(),
        "daytype": "month",
        "energy": tot,
        "has_shape": tot > 0,
        "shape_idx": np.arange(len(uniq)),
    })
    #Lorenzo Giannuzzo: the weight of a month is its share of the user's observed energy
    out["w"] = out["energy"] / out["pod"].map(out.groupby("pod")["energy"].sum())
    return mshapes.astype("float32"), out


def stratified_sample(days: pd.DataFrame, users: pd.DataFrame, n: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Row indices of a sample balanced across the strata.

    Stratification decides which days enter the sample, not how they are grouped
    once there, so the clustering stays blind to every label. The step is needed
    because the population is unbalanced: drawn in proportion, the sample would
    be domestic days almost throughout and the dictionary would spend its
    resolution on distinctions among them.

    Balanced rather than proportional makes the dictionary a vocabulary, a set of
    forms the population realises, and not an estimate of how often it realises
    them. Nothing downstream inherits the balancing: the frequencies of Eq. 4 are
    computed over each user's own year, as observed.
    """
    d = days[days["has_shape"]].copy()
    keys = ["season", "daytype"]
    for extra in ("tariff_class", "is_domestic"):
        if extra in d.columns:
            keys.append(extra)
    strata = d.groupby(keys, observed=True).indices
    per = max(1, n // max(len(strata), 1))
    picks = []
    for _, idx in strata.items():
        take = min(per, len(idx))
        picks.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(picks)
    if len(out) > n:
        out = rng.choice(out, size=n, replace=False)
    return d["shape_idx"].to_numpy()[out]


def _validity_D(sub: np.ndarray, w: np.ndarray, Z: np.ndarray,
                d_range: tuple[int, int]) -> pd.DataFrame:
    """The table on which D is chosen, and on which the choice is defended.

    The quantization error is what the dictionary is asked to minimise, and the
    two share columns say what the tree is actually doing at each cut: while the
    smallest codeword holds a handful of micro-clusters, the tree is shaving
    outliers off the population rather than dividing it, and the separation
    indices reward exactly that. They are reported all the same, since the paper
    compares them, but they are not what decides.
    """
    rows = []
    for D in range(d_range[0], d_range[1] + 1):
        lab = fcluster(Z, D, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        err, shares = quantization(sub, w, lab)
        rows.append({
            "D": D,
            "n_clusters": int(len(np.unique(lab))),
            "quantization_error": err,
            "largest_share": float(shares.max()),
            "smallest_share": float(shares.min()),
            "silhouette": silhouette_score(sub, lab),
            "davies_bouldin": davies_bouldin_score(sub, lab),
            "calinski_harabasz": calinski_harabasz_score(sub, lab),
        })
    return pd.DataFrame(rows)


def _resolve_D(val: pd.DataFrame, d_fixed, tol: float,
               min_share: float) -> tuple[int, str]:
    #Lorenzo Giannuzzo: null or auto in the configuration hands the choice to
    # the quantization criterion; an integer forces it and is reported as such,
    # so that a run always states which of the two produced the D it used.
    if d_fixed not in (None, "auto", "null"):
        return int(d_fixed), f"D = {int(d_fixed)}: fixed in the configuration"
    return select_D(val, tol=tol, min_share=min_share)


def summarised_ward_dictionary(shapes: np.ndarray, d_range: tuple[int, int],
                               d_fixed: int | None, out_dir: Path,
                               n_micro: int, batch: int,
                               rng: np.random.Generator,
                               unit_integral: bool = True,
                               d_tol: float = 0.01,
                               d_min_share: float = 0.005) -> tuple[np.ndarray, pd.DataFrame, int, str]:
    """Ward on every daily shape, by way of a summary of fixed size.

    Ward needs the distance between every pair, which is quadratic in memory: on
    2.5 million shapes that is tens of terabytes and no machine holds it. The
    obstacle is the number of points handed to it, not the criterion.

    The pool is therefore summarised first, in one linear pass, into `n_micro`
    micro-clusters, each standing for the shapes that fell in it; Ward then runs
    on those, weighted by cardinality, which leaves the linkage criterion
    untouched. Every shape takes part in the summary and none is sampled away.

    The summary is built by mini-batch k-means rather than by BIRCH for one
    reason that matters in practice: the number of micro-clusters is fixed in
    advance, so the memory Ward will need is known before starting. BIRCH sizes
    its summary from a distance threshold, and on shapes normalised to unit
    integral the typical distance is of the same order as any sensible
    threshold, so the count is impossible to anticipate and can run to six
    figures, at which point Ward is back to being impossible.
    """
    #Lorenzo Giannuzzo: The summary is only there because Ward cannot hold the distances between
    # millions of shapes. Below that, it is pointless and, if n_micro exceeds the
    # number of shapes, impossible: Ward then runs on the shapes themselves.
    if len(shapes) <= max(n_micro, 25_000):
        print(f"    only {len(shapes):,} shapes: Ward runs on all of them, "
              f"no summary needed")
        sub = np.asarray(shapes, dtype="float64")
        w = np.ones(len(sub))
        Z = ward_weighted(sub, w)
        val = _validity_D(sub, w, Z, d_range)
        val.to_csv(out_dir / "validity_D.csv", index=False)
        D, reason = _resolve_D(val, d_fixed, d_tol, d_min_share)
        lab = fcluster(Z, D, criterion="maxclust")
        cent = weighted_centroids(sub, w, lab)
        if unit_integral:
            cent = cent / cent.sum(axis=1, keepdims=True)
        return cent.astype("float32"), val, 0, reason

    print(f"    summarising all {len(shapes):,} shapes into {n_micro:,} "
          f"micro-clusters (one linear pass)...")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=n_micro, batch_size=batch, n_init=3,
                         random_state=42, max_no_improvement=50)
    for i in range(0, len(shapes), batch):
        km.partial_fit(np.asarray(shapes[i:i + batch], dtype="float64"))
    print(f"      done in {time.time()-t0:.0f}s")

    print("    counting the shapes behind each micro-cluster...")
    w = np.zeros(n_micro, dtype="float64")
    for i in range(0, len(shapes), 200_000):
        lab = km.predict(np.asarray(shapes[i:i + 200_000], dtype="float64"))
        w += np.bincount(lab, minlength=n_micro)

    keep = w > 0
    sub, w = km.cluster_centers_[keep], w[keep]
    mem = len(sub) ** 2 * 8 / 1e9
    print(f"      Ward on {len(sub):,} micro-clusters weighted by the "
          f"{int(w.sum()):,} shapes behind them (~{mem:.2f} GB)...")
    t0 = time.time()
    Z = ward_weighted(sub, w)
    print(f"      done in {time.time()-t0:.0f}s")

    val = _validity_D(sub, w, Z, d_range)
    val.to_csv(out_dir / "validity_D.csv", index=False)

    D, reason = _resolve_D(val, d_fixed, d_tol, d_min_share)
    lab = fcluster(Z, D, criterion="maxclust")
    cent = weighted_centroids(sub, w, lab)
    if unit_integral:
        cent = cent / cent.sum(axis=1, keepdims=True)
    return cent.astype("float32"), val, len(sub), reason


def ward_dictionary(sample: np.ndarray, d_range: tuple[int, int],
                    d_fixed: int | None, out_dir: Path,
                    rng: np.random.Generator,
                    d_tol: float = 0.01,
                    d_min_share: float = 0.005) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Ward on a stratified sample; the fallback when BIRCH is switched off.

    Ward merges the pair of groups minimising the increase in within-group
    inertia (Eq. 3). The criterion is defined on Euclidean inertia, so the
    distance is Euclidean by construction, which on shapes already normalised to
    unit integral and aligned on the clock is the right choice.
    """
    print(f"    Ward on {len(sample):,} sampled shapes...")
    t0 = time.time()
    Z = linkage(sample, method="ward")
    print(f"      linkage done in {time.time()-t0:.0f}s")

    w = np.ones(len(sample))
    val = _validity_D(sample, w, Z, d_range)
    val.to_csv(out_dir / "validity_D.csv", index=False)

    D, reason = _resolve_D(val, d_fixed, d_tol, d_min_share)
    lab = fcluster(Z, D, criterion="maxclust")
    cent = np.vstack([sample[lab == k].mean(axis=0) for k in np.unique(lab)])
    cent = cent / cent.sum(axis=1, keepdims=True)      # stay on unit integral
    return cent.astype("float32"), val, reason


def assign_nearest(shapes: np.ndarray, cent: np.ndarray,
                   chunk: int = 200_000) -> np.ndarray:
    """Nearest centroid in Euclidean distance, in chunks to bound memory."""
    out = np.empty(len(shapes), dtype="int16")
    cn = (cent ** 2).sum(axis=1)
    for i in range(0, len(shapes), chunk):
        blk = shapes[i:i + chunk]
        d = (blk ** 2).sum(axis=1)[:, None] - 2 * blk @ cent.T + cn[None, :]
        out[i:i + chunk] = d.argmin(axis=1)
    return out


# ── stage 2: the users ───────────────────────────────────────────────────────
def frequencies(days: pd.DataFrame, code: np.ndarray, D: int) -> pd.DataFrame:
    """Eq. 4 — the share of each user's annual energy falling on each codeword.

    Weighted by the day weight, not counted: a day carrying two percent of the
    year must outweigh one carrying a tenth of a percent, which a plain count
    would efface.
    """
    d = days.copy()
    d["code"] = -1
    has = d["has_shape"].to_numpy()
    d.loc[has, "code"] = code[d.loc[has, "shape_idx"].to_numpy()]
    f = (d[d["code"] >= 0]
         .pivot_table(index="pod", columns="code", values="w",
                      aggfunc="sum", fill_value=0.0))
    for k in range(D):
        if k not in f.columns:
            f[k] = 0.0
    f = f[sorted(f.columns)]
    f.columns = [f"f_{k+1}" for k in f.columns]
    #Lorenzo Giannuzzo: renormalise: days at zero carry w=0 and are absent, so the row may fall
    # short of one by the share of the year spent at zero
    s = f.sum(axis=1).replace(0, np.nan)
    return f.div(s, axis=0).fillna(0.0)


def clr(f: np.ndarray, delta: float) -> np.ndarray:
    """Eq. 5 — centred log-ratio.

    The frequency vector is compositional: non-negative, summing to one, so it
    lies on a simplex where Euclidean geometry does not apply in the ordinary
    sense and Ward would carry no warrant. The transform is undefined where a
    component vanishes, as it does whenever a user never realises a codeword, so
    zeros are replaced first, the closure being preserved.
    """
    x = f.astype("float64").copy()
    zero = x <= 0
    x[zero] = delta
    x = x / x.sum(axis=1, keepdims=True)
    g = np.exp(np.log(x).mean(axis=1, keepdims=True))
    return np.log(x / g)


def scale_features(days: pd.DataFrame, shapes: np.ndarray,
                   users: pd.DataFrame) -> pd.DataFrame:
    """The five features the dictionary does not carry, plus the zero fraction.

    load factor        mean power over the year / peak power        -> intermittency across the year
    peak-to-average    mean over days of (peak of day / mean of day) -> peakedness within the day
    weekday/weekend    mean daily energy of working over non-working days
    seasonal amplitude (winter - summer) / (winter + summer), in [-1, 1]
    """
    d = days[days["has_shape"]].copy()
    idx = d["shape_idx"].to_numpy()
    smax = shapes[idx].max(axis=1)
    d["day_peak_kW"] = d["energy"].to_numpy() * smax * 4.0     # kWh/quarter -> kW
    d["par_day"] = smax * 96.0                                  # peak / mean, within the day

    g = d.groupby("pod")
    feat = pd.DataFrame({
        "peak_kW": g["day_peak_kW"].max(),
        "par": g["par_day"].mean(),
    })
    feat = feat.join(users.set_index("pod")[["E", "zero_day_fraction"]])
    feat["load_factor"] = (feat["E"] / 8760.0) / feat["peak_kW"].replace(0, np.nan)

    wk = d[d["daytype"] == "weekday"].groupby("pod")["energy"].mean()
    we = d[d["daytype"] != "weekday"].groupby("pod")["energy"].mean()
    feat["weekday_weekend"] = (wk / we.replace(0, np.nan)).reindex(feat.index)

    win = d[d["season"] == "winter"].groupby("pod")["energy"].mean().reindex(feat.index)
    summ = d[d["season"] == "summer"].groupby("pod")["energy"].mean().reindex(feat.index)
    feat["seasonal_amplitude"] = (win - summ) / (win + summ).replace(0, np.nan)

    return feat[["E", "load_factor", "par", "weekday_weekend",
                 "seasonal_amplitude", "zero_day_fraction"]].fillna(0.0)


def zscore(a: np.ndarray) -> np.ndarray:
    m, s = a.mean(axis=0), a.std(axis=0)
    s[s == 0] = 1.0
    return (a - m) / s


def ward_users(X: np.ndarray, k_range: tuple[int, int], k_fixed: int | None,
               out_dir: Path, n_min: int = 30, threshold: float = 0.75,
               n_boot: int = 10, frac: float = 0.8, seed: int = 42,
               max_small_share: float = 0.01
               ) -> tuple[np.ndarray, pd.DataFrame, str, np.ndarray]:
    """The K groups, and the table on which K was chosen.

    The separation indices are computed and reported because Section 3 compares
    them, but they do not decide: on these vectors the silhouette rises almost
    monotonically with K and would return the top of the range whatever the
    population looked like. What decides is whether a partition reproduces
    itself on an independent subsample, which is a property of the users rather
    than of the sample drawn from them.
    """
    Z = linkage(X, method="ward")
    rows, sizes_at_K = [], {}
    for K in range(k_range[0], k_range[1] + 1):
        lab = fcluster(Z, K, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        counts = np.bincount(lab)[1:]
        counts = counts[counts > 0]
        sizes_at_K[K] = int(counts.min())
        rows.append({
            "K": K,
            "smallest_group": sizes_at_K[K],
            #Lorenzo Giannuzzo: the users a partition strands in groups too small to carry a
            # profile, which is what fragmentation costs and what the smallest
            # group on its own does not say
            "small_share": float(counts[counts < n_min].sum() / counts.sum()),
            "silhouette": silhouette_score(X, lab),
            "davies_bouldin": davies_bouldin_score(X, lab),
            "calinski_harabasz": calinski_harabasz_score(X, lab),
        })
    val = pd.DataFrame(rows)

    if k_fixed not in (None, "auto", "null"):
        K = int(k_fixed)
        reason = f"K = {K}: fixed in the configuration"
        val["stability_mean"] = np.nan
        val["stability_min"] = np.nan
    else:
        print(f"    stability of the partition over {n_boot} pairs of subsamples...")
        stab = stability_table(X, k_range, lambda a: linkage(a, method="ward"),
                               n_boot=n_boot, frac=frac, seed=seed)
        val = val.merge(stab, on="K", how="left")
        K, reason = select_K(val, sizes_at_K, threshold=threshold, n_min=n_min,
                             max_small_share=max_small_share)

    val.to_csv(out_dir / "validity_K.csv", index=False)
    #Lorenzo Giannuzzo: the linkage is returned so that the sweep on
    #representativeness can cut the same tree at every K instead of rebuilding
    #it. Rebuilding would also re-derive the same dendrogram, but reusing this
    #one guarantees that the partition the sweep scores at the selected K is
    #the very partition the run goes on to use.
    return fcluster(Z, K, criterion="maxclust"), val, reason, Z


# ── representativeness ───────────────────────────────────────────────────────
def pod_cell_sums(days_daily: pd.DataFrame, shapes, pod_index
                  ) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Per point and per cell, the sum of the daily shapes and the day count.

    Both quantities are independent of K, so they are accumulated once and every
    partition the sweep tries is scored by aggregating them rather than by
    walking the shapes again. The day-level view is returned with them because
    the quantiles of the dispersion cannot be recovered from the sums.
    """
    piv = (days_daily[days_daily["has_shape"]]
           .assign(cell=lambda x: x["season"].astype(str) + "|" + x["daytype"].astype(str)))
    cells = sorted(piv["cell"].unique())
    cell_id = piv["cell"].map({c: i for i, c in enumerate(cells)}).to_numpy()
    pod_id = pd.Categorical(piv["pod"], categories=list(pod_index)).codes
    ok = pod_id >= 0                       # PODs absent from the vectors carry -1
    shape_idx = piv["shape_idx"].to_numpy()[ok]
    key = (pod_id[ok].astype("int64") * len(cells) + cell_id[ok]).astype("int64")

    #Lorenzo Giannuzzo: one pass with np.add.at instead of a Python loop over the PODs
    sums = np.zeros((len(pod_index) * len(cells), 96), dtype="float64")
    np.add.at(sums, key, np.asarray(shapes[shape_idx], dtype="float64"))
    cnt = np.bincount(key, minlength=len(pod_index) * len(cells)).astype("float64")

    day_view = pd.DataFrame({"pod_id": pod_id[ok].astype("int64"),
                             "cell_id": cell_id[ok].astype("int64"),
                             "shape_idx": shape_idx})
    return sums, cnt, cells, day_view


def group_cell_curves(lab: np.ndarray, sums: np.ndarray, cnt: np.ndarray,
                      n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #Lorenzo Giannuzzo: the curve of a group in a cell is the mean of the days
    #its members spend in that cell, which is the sum of the member sums over the
    #sum of the member counts and not the mean of the member means. The two differ
    #whenever the points contribute unequal numbers of days, and the first is what
    #the profile claims to be.
    ids = np.unique(lab)
    n_pods = len(lab)
    rows = np.arange(n_pods, dtype="int64")[:, None] * n_cells + np.arange(n_cells)
    curves = np.zeros((len(ids), n_cells, 96), dtype="float64")
    counts = np.zeros((len(ids), n_cells), dtype="float64")
    for i, g in enumerate(ids):
        sel = rows[lab == g].ravel()
        s = sums[sel].reshape(-1, n_cells, 96).sum(axis=0)
        c = cnt[sel].reshape(-1, n_cells).sum(axis=0)
        counts[i] = c
        np.divide(s, c[:, None], out=curves[i], where=c[:, None] > 0)
    return curves, counts, ids


def dispersion_by_cell(lab: np.ndarray, sums: np.ndarray, cnt: np.ndarray,
                       cells: list[str], day_view: pd.DataFrame, shapes,
                       n_min: int, rng, sample: int = 200_000) -> pd.DataFrame:
    """How far the members sit from the curve that is meant to stand for them.

    The curves are exact, built on every day. The quantiles are read on a random
    sample of the days, because the sweep asks the question fifteen times and a
    quantile on two hundred thousand days is already stable to the third decimal.
    The value reported is the root mean squared deviation of a day from its own
    curve, divided by the mean level of that curve, so that it is a share of the
    signal and points of any size can be put on the same axis.
    """
    n_cells = len(cells)
    curves, counts, ids = group_cell_curves(lab, sums, cnt, n_cells)

    sizes = pd.Series(lab).value_counts()
    keep = set(sizes[sizes >= n_min].index)
    row_of_group = {g: i for i, g in enumerate(ids)}

    pod_id = day_view["pod_id"].to_numpy()
    group_of_day = lab[pod_id]
    live = np.isin(group_of_day, list(keep))
    idx = np.flatnonzero(live)
    if len(idx) > sample:
        idx = np.sort(rng.choice(idx, size=sample, replace=False))

    si = day_view["shape_idx"].to_numpy()[idx]
    x = np.asarray(shapes[si], dtype="float64")
    g_row = np.array([row_of_group[g] for g in group_of_day[idx]])
    c_row = day_view["cell_id"].to_numpy()[idx]

    ref = curves[g_row, c_row]
    level = ref.mean(axis=1)
    rmsd = np.sqrt(((x - ref) ** 2).mean(axis=1))
    nrmsd = np.divide(rmsd, level, out=np.full_like(rmsd, np.nan), where=level > 0)

    frame = pd.DataFrame({"group": group_of_day[idx],
                          "cell": np.asarray(cells)[c_row],
                          "nrmsd": nrmsd}).dropna()
    out = (frame.groupby(["group", "cell"])["nrmsd"]
           .agg(n_days_sampled="size",
                nrmsd_p50=lambda s: s.quantile(0.50),
                nrmsd_p90=lambda s: s.quantile(0.90),
                nrmsd_p95=lambda s: s.quantile(0.95))
           .reset_index())
    out["n_pods"] = out["group"].map(sizes)
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    #Lorenzo Giannuzzo: the value at which the cumulative weight crosses one half.
    #A weighted mean would be pulled by the handful of very tight cells that the
    #partition produces at high K, which is the same distortion the weighting is
    #there to remove.
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    total = np.cumsum(weights)
    if total[-1] <= 0:
        return float("nan")
    return float(values[np.searchsorted(total, 0.5 * total[-1])])


def sweep_K_dispersion(Z: np.ndarray, sums: np.ndarray, cnt: np.ndarray,
                       cells: list[str], day_view: pd.DataFrame, shapes,
                       k_range: tuple[int, int], out_dir: Path, rng,
                       n_min: int = 30, sample: int = 200_000) -> pd.DataFrame:
    """K read on what the profiles are for, rather than on how compact they are.

    The separation indices are computed in the very space the partition was built
    in, and the ablation shows the same partition scoring below zero once it is
    carried into the space of the mean curves. Dispersion is outside that circle:
    it asks how far a member sits from the curve published in its name, which is
    the claim a standard load profile makes and the claim Section 3 tests.
    """
    n_cells = len(cells)
    rows = []
    print(f"    dispersion over K = {k_range[0]}..{k_range[1]} "
          f"(quantiles on up to {sample:,} days per K)")

    for K in range(k_range[0], k_range[1] + 1):
        lab = fcluster(Z, K, criterion="maxclust")
        sizes = pd.Series(lab).value_counts()
        kept = sizes[sizes >= n_min]
        if kept.empty:
            continue

        disp = dispersion_by_cell(lab, sums, cnt, cells, day_view, shapes,
                                  n_min=n_min, rng=rng, sample=sample)
        p50 = disp["nrmsd_p50"].to_numpy()
        p95 = disp["nrmsd_p95"].to_numpy()
        w_pod = disp["n_pods"].to_numpy(dtype="float64")
        w_day = disp["n_days_sampled"].to_numpy(dtype="float64")

        rows.append({
            "K": K,
            "n_profiles": int(len(kept)),
            "pods_covered": int(kept.sum()),
            "pods_below_n_min": int(sizes.sum() - kept.sum()),
            #Lorenzo Giannuzzo: the cell median as the summary is written today, where a group of
            # forty points weighs as much as a group of two thousand because both
            # are spread over the same nine cells
            "nrmsd_p50_unweighted": float(np.median(p50)),
            #Lorenzo Giannuzzo: and the same median with each cell carrying its population, which is
            # the figure the knee has to be read on: without it K improves simply
            # by shedding small tight groups off the main body of users
            "nrmsd_p50_pod_weighted": _weighted_median(p50, w_pod),
            "nrmsd_p50_day_weighted": _weighted_median(p50, w_day),
            "nrmsd_p95_pod_weighted": _weighted_median(p95, w_pod),
            "worst_cell_p95": float(np.max(p95)),
        })
        r = rows[-1]
        print(f"      K={K:>3}  profiles={r['n_profiles']:>3}  "
              f"covered={r['pods_covered']:>6,}  "
              f"stranded={r['pods_below_n_min']:>4}  "
              f"nRMSD unweighted={r['nrmsd_p50_unweighted']:.3f}  "
              f"weighted={r['nrmsd_p50_pod_weighted']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "validity_K_dispersion.csv", index=False)

    if len(table) > 1:
        v = table["nrmsd_p50_pod_weighted"].to_numpy()
        u = table["nrmsd_p50_unweighted"].to_numpy()
        ks = table["K"].to_numpy()
        print("\n      gain per added group, on the weighted median")
        for i in range(len(ks) - 1):
            gain = v[i] - v[i + 1]
            share = 100.0 * gain / v[i] if v[i] else float("nan")
            drift = (u[i] - u[i + 1]) - gain
            flag = "   <- unweighted only" if drift > 0.01 else ""
            print(f"        {ks[i]:>3} -> {ks[i+1]:<3}  {gain:+.4f}  "
                  f"({share:+.1f}%){flag}")
        print("\n      a step marked unweighted only is buying its improvement "
              "from small groups")

    return table


def plot_K_dispersion(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].plot(table["K"], table["nrmsd_p50_unweighted"], "o--", ms=4,
                 color="#9aa5b1", label="unweighted")
    axes[0].plot(table["K"], table["nrmsd_p50_pod_weighted"], "o-", ms=4,
                 color="#0d1f3c", label="weighted by PODs")
    axes[0].set_title("median nRMSD  (lower is better)", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)

    axes[1].plot(table["K"], table["nrmsd_p95_pod_weighted"], "o-", ms=4,
                 color="#0d1f3c")
    axes[1].set_title("p95 nRMSD, weighted  (lower is better)", fontsize=9)

    axes[2].plot(table["K"], table["pods_below_n_min"], "o-", ms=4, color="#8c2f2f")
    axes[2].set_title("PODs left without a profile  (lower is better)", fontsize=9)

    for ax in axes:
        ax.set_xlabel("K")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── plots ────────────────────────────────────────────────────────────────────
def plot_dictionary(cent: np.ndarray, share: np.ndarray, path: Path,
                    shapes: np.ndarray | None = None,
                    code: np.ndarray | None = None,
                    n_show: int = 400,
                    rng: np.random.Generator | None = None,
                    unit_integral: bool = True) -> None:
    """One panel per codeword: the member curves, their spread, the centroid.

    A centroid on its own says nothing about whether it describes anything. A
    genuine form has members that look like it; a residual cluster has members
    with peaks at different hours that cancel in the mean, and its centroid is
    flat while none of its members is. The two are indistinguishable until the
    members are drawn, which is what this does.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = rng or np.random.default_rng(0)
    scale = 100.0 if unit_integral else 1.0
    D = len(cent)
    ncol = min(4, D)
    nrow = int(np.ceil(D / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow),
                             sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(96) / 4.0

    for k in range(D):
        ax = axes[k]
        top = cent[k].max() * scale

        if shapes is not None and code is not None:
            idx = np.where(code == k)[0]
            if len(idx):
                pick = rng.choice(idx, size=min(n_show, len(idx)), replace=False)
                mem = np.asarray(shapes[np.sort(pick)], dtype="float64") * scale
                #Lorenzo Giannuzzo: the members themselves, faint enough that density reads as shade
                ax.plot(x, mem.T, color="#0d1f3c", alpha=0.02, lw=0.6)
                lo, hi = np.percentile(mem, [10, 90], axis=0)
                ax.fill_between(x, lo, hi, color="#1565c0", alpha=0.18, lw=0)
                top = max(top, np.percentile(mem, 97))

        ax.plot(x, cent[k] * scale, color="#c62828", lw=1.8)
        if unit_integral:
            ax.axhline(100 / 96, color="#64748b", lw=0.7, ls=":")   # a flat day
        n = int(round(share[k] * (len(code) if code is not None else 0)))
        sub = f"{share[k]*100:.1f}% of days" + (f"  ·  {n}" if n else "")
        ax.set_title(f"form {k+1}  ·  {sub}", fontsize=9)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(0, top * 1.1)
        ax.tick_params(labelsize=7)

    for k in range(D, len(axes)):
        axes[k].axis("off")
    fig.supxlabel("hour of day")
    fig.supylabel("% of the daily energy" if unit_integral else "share of the daily peak")
    tail = ("   ·   dotted: a perfectly flat day (1/96)" if unit_integral
            else "   ·   min-max: the peak is 1 whatever it is")
    fig.suptitle("red: centroid   ·   blue band: 10th-90th percentile of the members"
                 + tail, fontsize=8, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_groups(f: pd.DataFrame, lab: np.ndarray, path: Path) -> None:
    """What each group is made of, in the terms in which it was built.

    A group is not a curve: it is a set of users spending their year on the same
    mixture of daily forms. The mixture is therefore what describes it, and the
    heatmap reads as a sentence: the users of group 3 spend 78% of their energy
    on days of form 5 and 19% on days of form 9.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fcols = [c for c in f.columns if c.startswith("f_")]
    g = pd.DataFrame(f[fcols].to_numpy(), columns=fcols)
    g["group"] = lab
    mix = g.groupby("group")[fcols].mean()
    sizes = g.groupby("group").size()
    K, D = mix.shape

    fig = plt.figure(figsize=(3.0 + 0.55 * D, 1.8 + 0.5 * K))
    gs = fig.add_gridspec(1, 2, width_ratios=[D, 3.0], wspace=0.06)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(mix.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(D), [c.replace("f_", "") for c in fcols], fontsize=8)
    ax.set_yticks(range(K), [f"group {i}" for i in mix.index], fontsize=8)
    ax.set_xlabel("daily form", fontsize=9)
    for i in range(K):
        for j in range(D):
            v = mix.iat[i, j]
            if v >= 0.04:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.5 else "#0d1f3c")
    ax.set_title("share of the annual energy spent on each form  (%)",
                 fontsize=9, loc="left")

    ax2 = fig.add_subplot(gs[1])
    y = np.arange(K)
    ax2.barh(y, sizes.to_numpy(), color="#0d1f3c", height=0.6)
    ax2.set_yticks(y, [])
    ax2.invert_yaxis()          # imshow counts rows from the top, barh from the bottom
    ax2.set_xlabel("PODs", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)
    for i, n in enumerate(sizes.to_numpy()):
        ax2.text(n, i, f" {n}", va="center", fontsize=7)
    ax2.set_xlim(0, sizes.max() * 1.3)

    fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_validity(val: pd.DataFrame, col: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, m, better in zip(axes,
                             ["silhouette", "davies_bouldin", "calinski_harabasz"],
                             ["higher", "lower", "higher"]):
        ax.plot(val[col], val[m], "o-", color="#0d1f3c", ms=4)
        ax.set_xlabel(col)
        ax.set_title(f"{m}  ({better} is better)", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── pipeline ─────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    cfg = load_config()
    cl = cfg["clustering"]
    out = cfg.results_dir("clustering")
    rng = np.random.default_rng(int(cl.get("random_state", 42)))

    print(f"\n{'='*78}\nSTAGE 2 — CLUSTERING (Section 2.3)\n{'='*78}\n")

    norm = str(cfg.get("preprocessing.shape_normalisation", "unit_integral")).lower()
    unit_integral = norm == "unit_integral"
    unit = str(cl.get("shape_unit", "day")).lower()
    shapes = np.load(cfg.cache_dir / "shapes.npy", mmap_mode="r")
    days = pd.read_parquet(cfg.cache_dir / "days.parquet")
    users = pd.read_parquet(cfg.cache_dir / "users.parquet")
    days_daily = days                      # generation and the ablation want the days

    if unit == "month":
        shapes, days = to_monthly(np.asarray(shapes), days, unit_integral)
        print(f"  first stage on POD-months: {len(shapes):,} shapes "
              f"(from {len(days_daily):,} days)")
    elif unit != "day":
        raise ValueError(f"shape_unit must be day | month, got {unit!r}")
    print(f"  {len(shapes):,} daily shapes, {len(users):,} PODs   "
          f"(shapes normalised: {norm})\n")

    #Lorenzo Giannuzzo: The resolution at which the vocabulary is built, which is not the
    # resolution of anything else: the shapes stay at 96 components everywhere
    # they are reported or compared.
    res = str(cl.get("dictionary_resolution", "quarter_hourly")).lower()
    shapes_full = shapes
    if res in ("hourly", "hour", "1h"):
        shapes = to_hourly(shapes, unit_integral)
        print(f"  dictionary built on hourly shapes: {shapes.shape[1]} components "
              f"per day instead of {shapes_full.shape[1]}\n")
    elif res not in ("quarter_hourly", "15min", "qh"):
        raise ValueError(f"dictionary_resolution must be hourly | quarter_hourly, "
                         f"got {res!r}")

    #Lorenzo Giannuzzo: strata that exist in the data
    tcol = next((c for c in users.columns if c.lower() in ("d_tipta", "d_49des")), None)
    if tcol:
        days = days.merge(users[["pod", tcol]].rename(columns={tcol: "tariff_class"}),
                          on="pod", how="left")
    if "ateco_l1" in users:
        dom = users[["pod", "ateco_l1"]].copy()
        dom["is_domestic"] = dom["ateco_l1"].astype(str).str.startswith(("DO", "CO", "IL"))
        days = days.merge(dom[["pod", "is_domestic"]], on="pod", how="left")

    # ── the dictionary ───────────────────────────────────────────────────────
    print("  Stage 1 — the dictionary")
    method = str(cl.get("dictionary_method", "birch_ward")).lower()
    scope = str(cl.get("dictionary_scope", "global")).lower()
    n_sub = 0
    d_reasons: list[str] = []
    d_reason = ""

    if scope == "per_season":
        #Lorenzo Giannuzzo: One vocabulary per season. Shapes are normalised to unit integral, so a
        # flat winter day and a flat summer day are the same curve and a global
        # dictionary merges them, taking the seasonality out of the frequency
        # vector. Split by season, it becomes a coordinate.
        seasons = [s for s in cfg.get("preprocessing.seasons")
                   if (days["season"] == s).any()]
        print(f"    per_season: one dictionary for each of {seasons}")
        cent_parts, code = [], np.full(len(shapes), -1, dtype="int32")
        val_D, offset = [], 0
        for s in seasons:
            sel = days.loc[days["has_shape"] & (days["season"] == s),
                           "shape_idx"].to_numpy()
            print(f"\n    [{s}] {len(sel):,} shapes")
            sub_shapes = np.asarray(shapes[np.sort(sel)], dtype="float32")
            c_s, v_s, n_s, r_s = summarised_ward_dictionary(
                sub_shapes, tuple(cl["codewords_range"]), cl.get("n_codewords"),
                out, int(cl.get("n_micro", 4000)),
                int(cl.get("batch_size", 50_000)), rng, unit_integral=unit_integral,
                d_tol=float(cl.get("d_tol", 0.01)),
                d_min_share=float(cl.get("d_min_share", 0.005)))
            d_reasons.append(f"[{s}] {r_s}")
            n_sub += n_s
            code[np.sort(sel)] = assign_nearest(sub_shapes, c_s) + offset
            offset += len(c_s)
            cent_parts.append(c_s)
            v_s = v_s.assign(season=s)
            val_D.append(v_s)
        cent = np.vstack(cent_parts)
        val_D = pd.concat(val_D, ignore_index=True) if val_D else pd.DataFrame()
        #Lorenzo Giannuzzo: which season each codeword belongs to, for the report
        cw_season = np.concatenate([[s] * len(c_) for s, c_ in zip(seasons, cent_parts)])
        D = len(cent)
        d_reason = " ; ".join(d_reasons)
        print(f"\n    D = {D} codewords in all "
              f"({', '.join(f'{s}:{len(c_)}' for s, c_ in zip(seasons, cent_parts))})")
    elif method in ("summarised_ward", "birch_ward"):
        #Lorenzo Giannuzzo: Ward on every shape, through a summary of fixed size. Nothing is
        # sampled away, and the memory Ward will need is known in advance.
        cent, val_D, n_sub, d_reason = summarised_ward_dictionary(
            shapes, tuple(cl["codewords_range"]), cl.get("n_codewords"), out,
            int(cl.get("n_micro", 4000)), int(cl.get("batch_size", 50_000)), rng,
            unit_integral=unit_integral,
            d_tol=float(cl.get("d_tol", 0.01)),
            d_min_share=float(cl.get("d_min_share", 0.005)))
    elif method == "sample_ward":
        M = int(cl["sample_size"])
        if M > 25_000:
            print(f"    NOTE: Ward is quadratic in memory; {M:,} needs "
                  f"~{M*M*8/2/1e9:.1f} GB for the condensed distances")
        samp_idx = stratified_sample(days, users, M, rng)
        sample = np.asarray(shapes[np.sort(samp_idx)], dtype="float64")
        print(f"    stratified sample: {len(sample):,} of {len(shapes):,} shapes")
        cent, val_D, d_reason = ward_dictionary(
            sample, tuple(cl["codewords_range"]), cl.get("n_codewords"), out, rng,
            d_tol=float(cl.get("d_tol", 0.01)),
            d_min_share=float(cl.get("d_min_share", 0.005)))
    elif method == "minibatch":
        print("    MiniBatchKMeans (exploratory only; the paper declares Ward)")
        D = int(cl.get("n_codewords") or 8)
        km = MiniBatchKMeans(n_clusters=D, random_state=42, n_init=3)
        for i in range(0, len(shapes), 100_000):
            km.partial_fit(np.asarray(shapes[i:i + 100_000], dtype="float64"))
        cent = km.cluster_centers_.astype("float32")
        cent = cent / cent.sum(axis=1, keepdims=True)
        val_D = pd.DataFrame()
        d_reason = f"D = {D}: fixed in the configuration (exploratory method)"
    else:
        raise ValueError(f"dictionary_method must be summarised_ward | "
                         f"sample_ward | minibatch, got {method!r}")
    if scope != "per_season":
        D = len(cent)
        cw_season = np.array(["all"] * D)
        print(f"    D = {D} codewords")
        if d_reason:
            print(f"      {d_reason}")
        print(f"    assigning all {len(shapes):,} shapes to the nearest codeword...")
        code = assign_nearest(np.asarray(shapes), cent)

    if shapes_full is not shapes:
        cent = full_resolution_centroids(shapes_full, code, D, unit_integral)

    # ── stability and bias ───────────────────────────────────────────────────
    stab = []
    R = int(cl.get("replicas", 0) or 0)
    if R > 1 and method == "sample_ward" and scope != "per_season":
        print(f"    stability over {R} replicas...")
        hold = rng.choice(len(shapes), size=min(20_000, len(shapes)), replace=False)
        hold_x = np.asarray(shapes[np.sort(hold)], dtype="float64")
        base = assign_nearest(hold_x.astype("float32"), cent)
        for r in range(R - 1):
            s2 = stratified_sample(days, users, M, np.random.default_rng(1000 + r))
            x2 = np.asarray(shapes[np.sort(s2)], dtype="float64")
            Z2 = linkage(x2, method="ward")
            lab2 = fcluster(Z2, D, criterion="maxclust")
            c2 = np.vstack([x2[lab2 == k].mean(axis=0) for k in np.unique(lab2)])
            c2 = (c2 / c2.sum(axis=1, keepdims=True)).astype("float32")
            stab.append({"comparison": f"replica {r+1}",
                         "ARI": adjusted_rand_score(base, assign_nearest(hold_x.astype("float32"), c2))})
        #Lorenzo Giannuzzo: bias against the full pool, which no replica can reveal
        print("    BIRCH on the whole pool, to bound the sampling bias...")
        br = Birch(n_clusters=None, threshold=0.02).fit(
            np.asarray(shapes[::5], dtype="float64"))
        sub = br.subcluster_centers_
        wgt = np.bincount(br.subcluster_labels_, minlength=len(sub))
        Zb = linkage(sub, method="ward")
        lb = fcluster(Zb, D, criterion="maxclust")
        cb = np.vstack([np.average(sub[lb == k], axis=0, weights=wgt[lb == k])
                        for k in np.unique(lb)])
        cb = (cb / cb.sum(axis=1, keepdims=True)).astype("float32")
        stab.append({"comparison": "BIRCH on full pool",
                     "ARI": adjusted_rand_score(base, assign_nearest(hold_x.astype("float32"), cb))})
        pd.DataFrame(stab).to_csv(out / "stability.csv", index=False)
        for s in stab:
            print(f"      {s['comparison']:22s} ARI = {s['ARI']:.3f}")

    # ── stage 2: the users ───────────────────────────────────────────────────
    print("\n  Stage 2 — the users")
    f = frequencies(days, code, D)
    print(f"    frequency vectors: {f.shape[0]:,} x {f.shape[1]}")

    feat = scale_features(days_daily, np.load(cfg.cache_dir / "shapes.npy", mmap_mode="r"),
                          users).reindex(f.index)
    Xf = clr(f.to_numpy(), float(cl["zero_replacement"]))
    lam = float(cl["scale_weight"])
    Xs_raw = zscore(feat.to_numpy())

    #Lorenzo Giannuzzo: Ward reads one Euclidean distance over the concatenated vector and has no
    # notion of which coordinates describe behaviour and which describe size, so
    # what settles the balance between the two blocks is the total variance each
    # carries and not the count of coordinates. Left as they are, the CLR block
    # carries a variance that grows with D and with the sparsity of the
    # frequency vectors, while the standardised scale block always carries the
    # number of its features: lambda then means something different at every D,
    # which is how a declared weight of one half came to drive three per cent of
    # the distance. Dividing each block by its own total variance before
    # applying lambda makes the declared weight the realised one, at any D.
    mode = str(cl.get("block_normalisation", "variance")).lower()
    if mode == "variance":
        vf, vs = Xf.var(axis=0).sum(), Xs_raw.var(axis=0).sum()
        Xf = Xf / np.sqrt(vf) if vf > 0 else Xf
        Xs_raw = Xs_raw / np.sqrt(vs) if vs > 0 else Xs_raw
        Xf = np.sqrt(1.0 - lam) * Xf
        Xs = np.sqrt(lam) * Xs_raw
    elif mode in ("none", "raw"):
        Xs = lam * Xs_raw
    else:
        raise ValueError(f"block_normalisation must be variance | none, got {mode!r}")

    X = np.hstack([Xf, Xs])
    print(f"    vector: {D} CLR coordinates + {feat.shape[1]} scale features "
          f"weighted by lambda = {lam}   (block normalisation: {mode})")

    v_form, v_scale = Xf.var(axis=0).sum(), Xs.var(axis=0).sum()
    share = v_scale / (v_form + v_scale) if (v_form + v_scale) else 0
    print(f"      variance carried: forms {v_form:8.4f}   scale {v_scale:8.4f}   "
          f"-> scale drives {share*100:.0f}% of the distance "
          f"(declared lambda {lam:.2f})")
    if mode == "variance" and abs(share - lam) > 0.05:
        print(f"      WARNING: realised weight {share:.2f} differs from the "
              f"declared {lam:.2f}; the blocks did not normalise as expected")
    if share > 0.5:
        print("      WARNING: the partition is driven by size, not by behaviour.")
        print("               Section 2.1 claims the residential divide is an")
        print("               output; at this weight it would be an artefact.")

    lab, val_K, k_reason, Z_users = ward_users(
        X, tuple(cl["profiles_range"]), cl.get("n_profiles"), out,
        n_min=int(cl.get("min_group_size") or 30),
        threshold=float(cl.get("k_stability", 0.75)),
        n_boot=int(cl.get("k_n_boot", 10)),
        frac=float(cl.get("k_frac", 0.8)),
        seed=int(cl.get("random_state", 42)),
        max_small_share=float(cl.get("k_max_small_share", 0.01)))
    K = len(np.unique(lab))
    print(f"    K = {K} groups")
    print(f"      {k_reason}")

    # ── small groups ─────────────────────────────────────────────────────────
    n_min = cl.get("min_group_size") or max(30, 3 * K)
    sizes = pd.Series(lab).value_counts()
    small = set(sizes[sizes < n_min].index)
    if small:
        print(f"    {len(small)} group(s) below n_min = {n_min}: reported, "
              f"excluded from the metrics")

    groups = pd.DataFrame({"pod": f.index, "group": lab})
    groups["below_n_min"] = groups["group"].isin(small)

    # ── ablation: the mixture against the mean it replaces ───────────────────
    print("\n  Ablation — mean curves on the regulatory grid")
    #Lorenzo Giannuzzo: the baseline is always the mean curves of the regulatory grid, whatever
    # unit the dictionary was built on
    shp_daily = np.load(cfg.cache_dir / "shapes.npy", mmap_mode="r")
    #Lorenzo Giannuzzo: the same accumulator serves the baseline here and the sweep below, so the
    # shapes are walked once for both
    sums, cnt, cells, day_view = pod_cell_sums(days_daily, shp_daily, f.index)
    means = np.divide(sums, cnt[:, None], out=np.zeros_like(sums), where=cnt[:, None] > 0)
    Xm = zscore(means.reshape(len(f), len(cells) * 96))

    lab_m = fcluster(linkage(Xm, method="ward"), K, criterion="maxclust")
    ari = adjusted_rand_score(lab, lab_m)

    #Lorenzo Giannuzzo: The ARI alone cannot tell an informative divergence from noise: two
    # partitions disagree completely both when one sees a structure the other
    # misses and when one of them has found nothing at all. The silhouette of
    # each says which of the two is the case, and it is the question the paper
    # actually asks of this test.
    sub = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    sil_two = silhouette_score(X[sub], lab[sub])
    sil_mean = silhouette_score(Xm[sub], lab_m[sub])

    #Lorenzo Giannuzzo: the two silhouettes are computed in two different
    # spaces, so each says how well its own partition holds in its own
    # representation and neither is comparable with the other as a number. The
    # cross terms are what make the comparison legitimate: each partition is
    # scored in both representations, and a partition that survives the other's
    # space is carrying structure rather than an artefact of its own coordinates.
    # No sentence is written here on the strength of these numbers. The same rule
    # once printed two opposite conclusions on two configurations of the same
    # data, and a reading that changes with a threshold belongs to the author of
    # the paper and not to the run that produced the figures.
    sil_two_in_mean = silhouette_score(Xm[sub], lab[sub])
    sil_mean_in_two = silhouette_score(X[sub], lab_m[sub])

    pd.DataFrame([{"comparison": "two-stage vs mean curves", "K": K, "ARI": ari,
                   "silhouette_two_stage": sil_two,
                   "silhouette_mean_curves": sil_mean,
                   "silhouette_two_stage_in_mean_space": sil_two_in_mean,
                   "silhouette_mean_curves_in_two_stage_space": sil_mean_in_two,
                   }]).to_csv(out / "ablation.csv", index=False)
    print(f"    ARI(two-stage, mean curves) = {ari:.3f}")
    print(f"    silhouette in its own space:   two-stage {sil_two:.3f}   "
          f"mean curves {sil_mean:.3f}")
    print(f"    silhouette in the other space: two-stage {sil_two_in_mean:.3f}   "
          f"mean curves {sil_mean_in_two:.3f}")

    # ── K on representativeness ──────────────────────────────────────────────
    disp_K = pd.DataFrame()
    if bool(cl.get("k_sweep_dispersion", True)):
        print("\n  Selection of K — dispersion of the members from their curve")
        disp_K = sweep_K_dispersion(
            Z_users, sums, cnt, cells, day_view, shp_daily,
            tuple(cl["profiles_range"]), out, rng,
            n_min=int(cl.get("min_group_size") or 30),
            sample=int(cl.get("k_sweep_sample", 200_000)))
        if not disp_K.empty:
            plot_K_dispersion(disp_K, out / "validity_K_dispersion.png")

    # ── write ────────────────────────────────────────────────────────────────
    write_manifest(cfg.cache_dir, cache_signature(
        cl, {"n_codewords": D, "n_profiles": K,
             "d_reason": d_reason, "k_reason": k_reason}))
    np.save(cfg.cache_dir / "dictionary.npy", cent)
    np.save(cfg.cache_dir / "day_codeword.npy", code)
    f.join(feat).reset_index().to_parquet(cfg.cache_dir / "user_vectors.parquet", index=False)
    groups.to_parquet(cfg.cache_dir / "groups.parquet", index=False)

    share_days = np.bincount(code, minlength=D) / len(code)
    e = days.loc[days["has_shape"], "energy"].to_numpy()
    share_energy = np.bincount(code, weights=e, minlength=D) / e.sum()
    pd.DataFrame({"codeword": range(1, D + 1),
                  "season": cw_season,
                  "share_of_days": share_days,
                  "share_of_energy": share_energy,
                  "peak_hour": cent.argmax(axis=1) / 4.0,
                  **{f"q{i+1}": cent[:, i] for i in range(96)}}
                 ).to_csv(out / "dictionary.csv", index=False)
    #Lorenzo Giannuzzo: the members are drawn at the resolution the codewords are reported at,
    # which is 96 components whatever resolution the vocabulary was built in
    plot_dictionary(cent, share_days, out / "dictionary.png",
                    shapes=shapes_full, code=code, rng=rng,
                    unit_integral=unit_integral)
    if not val_D.empty:
        plot_validity(val_D, "D", out / "validity_D.png")
    plot_validity(val_K, "K", out / "validity_K.png")

    plot_groups(f.reset_index(drop=True), lab, out / "groups.png")

    keep = ["pod"] + [c for c in ("ateco_l1", "ateco_l2", "prosumer") if c in users]
    comp = groups.merge(users[keep], on="pod", how="left")
    agg = comp.groupby("group").agg(n_pods=("pod", "size"))
    agg["share_of_pods"] = agg["n_pods"] / agg["n_pods"].sum()

    #Lorenzo Giannuzzo: the mixture that defines the group, which is how it was built
    fcols = [c for c in f.columns if c.startswith("f_")]
    mix = pd.DataFrame(f[fcols].to_numpy(), columns=fcols)
    mix["group"] = lab
    agg = agg.join(mix.groupby("group")[fcols].mean().round(3))
    #Lorenzo Giannuzzo: the form the group lives on, and how much of its year that form takes
    m = mix.groupby("group")[fcols].mean()
    agg["dominant_form"] = [int(c.replace("f_", "")) for c in m.idxmax(axis=1)]
    agg["dominant_share"] = m.max(axis=1).round(3)

    #Lorenzo Giannuzzo: the scale features, averaged
    fe = feat.copy()
    fe["group"] = lab
    agg = agg.join(fe.groupby("group").mean().round(3),
                   rsuffix="_mean")

    if "prosumer" in comp:
        agg["prosumer_share"] = comp.groupby("group")["prosumer"].mean().round(3)
    for lvl in ("ateco_l1", "ateco_l2"):
        if lvl in comp:
            agg[f"n_{lvl}"] = comp.groupby("group")[lvl].nunique()
            agg[f"top3_{lvl}"] = comp.groupby("group")[lvl].agg(
                lambda s: " | ".join(f"{k} {v*100:.0f}%" for k, v in
                                     s.value_counts(normalize=True).head(3).items())
                if s.notna().any() else "")
    agg["below_n_min"] = agg.index.isin(small)
    agg.to_csv(out / "groups.csv")

    #Lorenzo Giannuzzo: who is in each group, one row per POD: the table Section 3 needs
    comp.merge(f.reset_index()[["pod"] + fcols], on="pod", how="left").to_csv(
        out / "group_members.csv", index=False)

    with open(out / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"first-stage unit             {unit}\n")
        fh.write(f"shape normalisation          {norm}\n")
        fh.write(f"dictionary scope             {scope}\n")
        fh.write(f"dictionary method            {method}\n")
        if n_sub:
            fh.write(f"micro-clusters               {n_sub:,} "
                     f"(standing for all {len(shapes):,} shapes)\n")
        fh.write(f"[D] codewords                {D}\n")
        fh.write(f"    chosen because           {d_reason}\n")
        fh.write(f"[R] replicas                 {R}\n")
        fh.write(f"[K] groups                   {K}\n")
        fh.write(f"    chosen because           {k_reason}\n")
        fh.write(f"[n_min] minimum group        {n_min}\n")
        fh.write(f"lambda, scale weight         {lam}\n")
        fh.write(f"block normalisation          {mode}\n")
        fh.write(f"variance: forms / scale      {v_form:.4f} / {v_scale:.4f} "
                 f"(scale drives {share*100:.0f}% of the distance)\n")
        fh.write(f"delta, zero replacement      {cl['zero_replacement']}\n")
        fh.write(f"\nARI two-stage vs mean curves {ari:.3f}\n")
        for s in stab:
            fh.write(f"ARI {s['comparison']:24s} {s['ARI']:.3f}\n")
        fh.write(f"\nGroups below n_min           {len(small)}\n")

        if not disp_K.empty:
            here = disp_K[disp_K["K"] == K]
            fh.write("\nDispersion of the members from their own curve, "
                     "median nRMSD over the cells:\n")
            if not here.empty:
                r = here.iloc[0]
                fh.write(f"  at the selected K            "
                         f"{r['nrmsd_p50_pod_weighted']:.3f} weighted by PODs, "
                         f"{r['nrmsd_p50_unweighted']:.3f} unweighted\n")
                fh.write(f"  PODs left without a profile  "
                         f"{int(r['pods_below_n_min'])}\n")
            fh.write("  the sweep over the whole range is in "
                     "validity_K_dispersion.csv\n")

    print(f"\n{'='*78}")
    print(f"  [D] = {D} codewords   [K] = {K} groups   ({time.time()-t0:.0f}s)")
    print(f"\n  cache/   dictionary.npy, day_codeword.npy, user_vectors.parquet, groups.parquet")
    print(f"  results/ {out.name}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()