"""Stage 3 — Estimation.

Builds, for one typology, the object the generator walks on, and saves it so that
generation never has to re-estimate.

The model in one paragraph. The state is the level of power in the quarter-hour,
discretised on the quantiles of what that typology actually draws, with the zero
kept apart as a state of its own rather than as a low bin: nineteen percent of the
days in this archive are at zero and they are closures, not small consumption.
The state carries a momentum, the direction of the previous step, because a chain
that only knows its level underestimates persistence and produces curves far more
jagged than any metered day. Above the quarter-hourly chain sits a daily one over
four regimes, closed, low, medium and high, and that second layer is what makes
one generated day differ from the next: without it every day of a synthetic year
converges to the same average shape, which is exactly the diversity the project
asks for. The chain is not homogeneous in time; its transitions are conditioned on
the regime of the day, the season, the day type and the block of hours.

Sparsity is handled by backoff. A cell whose row holds few transitions is
interpolated toward its parent, the same cell with one conditioning dropped, with
a weight n / (n + k) that lets a well observed row stand on its own and a thin one
lean on the level above. No row is ever left degenerate.

Returning from a bin to a value is done by inverting the empirical distribution of
the readings observed in that bin, not by drawing uniformly inside it, and the
position within the bin follows an AR(1) so that a point sitting high in its bin
tends to stay high. That is what keeps the output from looking like a staircase.

Outputs
    results/models/<key>.npz        the estimated arrays
    results/models/manifest.csv     one row per model, with what it rests on

Run
    python -m synthgen.estimate --level 1 --all
    python -m synthgen.estimate --level 1 --typology 47
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .taxonomy import census, level_column, normalise_typology

ZERO_EPS = 1e-9
#Lorenzo Giannuzzo: Context of a step: where it came from and, when it did not move, for how long
# it has been standing still. A chain that knows only the direction leaves a level
# as readily after four hours as after fifteen minutes, which is what makes its
# curves spikier than any metered day. Splitting the standing still case by dwell
# is the cheapest way to give the level a memory of its own duration without
# going to a full second order chain, which would need the cube of the states.
N_CTX = 4                         #Lorenzo Giannuzzo: down, up, flat and recent, flat and settled
FLAT_LONG = 4                     #Lorenzo Giannuzzo: quarters of standing still before "settled"
DAYTYPES = ["weekday", "saturday", "sunday"]
REGIMES = ["closed", "low", "medium", "high"]


#Lorenzo Giannuzzo: ── states ───────────────────────────────────────────────────────────────────
def bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior quantiles of the positive readings.

    Quantiles rather than a fixed grid, so that every bin is populated whatever
    the scale of the typology. Duplicate edges, which appear when a typology
    spends most of its time at one level, are collapsed: fewer bins that mean
    something beat twelve bins three of which are the same number.
    """
    pos = values[np.isfinite(values) & (values > ZERO_EPS)]
    if pos.size < 100:
        raise ValueError("fewer than 100 positive readings; cannot bin")
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(pos, qs))
    return edges.astype("float64")


def to_state(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """0 for a reading at zero, 1..B for a positive one. NaN becomes -1."""
    st = np.full(values.shape, -1, dtype="int16")
    finite = np.isfinite(values)
    st[finite] = 0
    pos = finite & (values > ZERO_EPS)
    st[pos] = np.searchsorted(edges, values[pos], side="right") + 1
    return st


def context_of(states: np.ndarray, flat_long: int = FLAT_LONG) -> np.ndarray:
    """0 down, 1 up, 2 flat and recent, 3 flat and settled."""
    d = np.zeros_like(states)
    d[:, 1:] = np.sign(states[:, 1:] - states[:, :-1])
    ctx = np.full(states.shape, 2, dtype="int8")
    ctx[d < 0] = 0
    ctx[d > 0] = 1
    dwell = np.zeros(states.shape, dtype="int32")
    for t in range(1, states.shape[1]):
        dwell[:, t] = np.where(d[:, t] == 0, dwell[:, t - 1] + 1, 0)
    ctx[(d == 0) & (dwell >= flat_long)] = 3
    return ctx


#Lorenzo Giannuzzo: ── conditioning cells ───────────────────────────────────────────────────────
def time_blocks(n_quarters: int, n_blocks: int) -> np.ndarray:
    """Which block of hours each quarter of the day belongs to."""
    return (np.arange(n_quarters) * n_blocks // n_quarters).astype("int8")


def backoff(counts: np.ndarray, axes: list[int], k: float) -> np.ndarray:
    """Interpolate every row toward its parent, then normalise.

    `counts` is indexed (regime, season, daytype, block, from, to). The parents
    are built by summing over the conditioning axes in the order given, so the
    first parent drops the first conditioning listed and the last is the pooled
    table of the typology. A row of n transitions keeps weight n / (n + k) of
    itself and takes the rest from the level above, which means a well observed
    row is left alone and a thin one is never degenerate.

    The order therefore ranks the conditionings from the most expendable to the
    least, and the block of hours has to come last. Dropping it first, which is
    the intuitive reading of "back off to something coarser", hands every thin
    cell a parent that knows nothing about the time of day, and what comes out is
    a curve with the right marginal distribution of levels and no daily rhythm at
    all.
    """
    levels = [counts]
    cur = counts
    for ax in axes:
        cur = cur.sum(axis=ax, keepdims=True)
        levels.append(cur)

    def normed(a: np.ndarray) -> np.ndarray:
        tot = a.sum(axis=-1, keepdims=True)
        return np.divide(a, tot, out=np.zeros_like(a), where=tot > 0)

    #Lorenzo Giannuzzo: from the top down: each level is smoothed against the one above it
    prob = normed(levels[-1])
    for a in reversed(levels[:-1]):
        n = a.sum(axis=-1, keepdims=True)
        w = n / (n + k)
        prob = w * normed(a) + (1.0 - w) * np.broadcast_to(prob, a.shape)
        prob = normed(prob)

    #Lorenzo Giannuzzo: A row can still be empty at every level when the state it starts from never
    # occurs at all, arriving at zero on a rising step being the clear case. Such
    # a row is unreachable by construction, but leaving it at zero puts a trap in
    # the artifact, so it takes the pooled distribution of destinations.
    marginal = counts.sum(axis=tuple(range(counts.ndim - 1)))
    marginal = (marginal / marginal.sum() if marginal.sum() > 0
                else np.full(counts.shape[-1], 1.0 / counts.shape[-1]))
    empty = prob.sum(axis=-1) <= 0
    if empty.any():
        prob[empty] = marginal
    return prob


#Lorenzo Giannuzzo: ── estimation ───────────────────────────────────────────────────────────────
def estimate_one(curves: np.ndarray, days: pd.DataFrame, pods: pd.DataFrame,
                 par: dict) -> dict:
    """Every array the generator needs for one typology and size stratum."""
    n_q = curves.shape[1]
    n_bins = int(par["n_bins"])
    n_blocks = int(par["n_blocks"])
    k = float(par["shrinkage"])

    edges = bin_edges(curves, n_bins)
    B = len(edges) + 1                       #Lorenzo Giannuzzo: positive bins
    S = B + 1                                #Lorenzo Giannuzzo: plus the zero state
    states = to_state(curves, edges)
    ctx = context_of(states)
    blocks = time_blocks(n_q, n_blocks)

    #Lorenzo Giannuzzo: ── daily regime ─────────────────────────────────────────────────────────
    # The regime of a day is read against the point's own ordinary day, not
    # against the daily energy of the whole stratum. Daily energy varies far more
    # between points than inside one, so terciles cut on the pooled figure sort
    # the points by size and not the days by how busy they were: a small point
    # answers low on every day of its year and a large one high on every day of
    # its, and the layer that is supposed to tell a slack day from a full one
    # ends up telling a small meter from a large one, which is the work of the
    # size strata. Dividing by the point's own median positive day first makes
    # the three regimes mean the same thing for everybody, which is what lets the
    # quarter-hourly tables below stop mixing two populations: without it the
    # table for the low regime pools the quiet days of inhabited homes with the
    # days of empty ones, and a profile drawn from it can be neither.
    energy = days["energy"].to_numpy(dtype="float64")
    pod_col = days["pod"].to_numpy()
    positive = energy > ZERO_EPS
    own_day = (pd.DataFrame({"pod": pod_col,
                             "e": np.where(positive, energy, np.nan)})
               .groupby("pod")["e"].median())
    own = pd.Series(pod_col).map(own_day).to_numpy(dtype="float64")
    usable = np.isfinite(own) & (own > 0)
    rel = np.divide(energy, own, out=np.zeros_like(energy), where=usable)
    pos_r = rel[positive & usable]
    cuts = (np.quantile(pos_r, [1 / 3, 2 / 3]) if pos_r.size >= 30
            else np.array([0.0, 0.0]))
    regime = np.where(~positive, 0, np.searchsorted(cuts, rel) + 1)
    regime = np.clip(regime, 0, 3).astype("int8")

    #Lorenzo Giannuzzo: ── closure class ────────────────────────────────────────────────────────
    # How readily a point closes is a property of the point, not a probability of
    # the day, and estimating one daily chain over the pooled population turns it
    # into the second. The domestic typology says it plainly: a fifth of all its
    # metered days are at zero, and yet the typical domestic point is at zero on
    # four days in a thousand. The zero days are not spread over the population,
    # they belong to a minority of it, the empty flats and the houses lived in for
    # a fortnight in August, while an inhabited home never stops drawing. A single
    # pooled chain hands that fifth to every generated profile, so every synthetic
    # household became half a holiday home: the year spent at zero came out at
    # 0.354 against a metered 0.004, and since the annual energy is pinned to the
    # anchor, the same total spread over a third fewer days dropped the median
    # daily energy to 0.299 kWh against 1.307.
    #
    # So the points are grouped by their own share of zero days, one daily chain
    # is estimated per group, and the generator takes the group of the point its
    # profile is anchored to. The propensity is drawn once, from a real point,
    # and then held, exactly as the annual energy is.
    zero_by_pod = (pd.DataFrame({"pod": pod_col, "z": energy <= ZERO_EPS})
                   .groupby("pod")["z"].mean())
    c_cuts = np.asarray(par.get("closure_cuts", [0.02, 0.20]), dtype="float64")
    n_cls = len(c_cuts) + 1
    cls_by_pod = pd.Series(np.searchsorted(c_cuts, zero_by_pod.to_numpy()),
                           index=zero_by_pod.index)
    c_idx = pd.Series(pod_col).map(cls_by_pod).fillna(0).to_numpy().astype("int8")

    season_lab = sorted(days["season"].unique().tolist())
    s_idx = days["season"].map({s: i for i, s in enumerate(season_lab)}).to_numpy()
    d_idx = days["daytype"].map({d: i for i, d in enumerate(DAYTYPES)}).fillna(0) \
                           .to_numpy().astype("int8")
    n_seasons = len(season_lab)

    #Lorenzo Giannuzzo: regime transitions, day to day, conditioned on season and day type of the
    # arriving day. Only consecutive days of the same POD are counted.
    r_counts = np.zeros((n_cls, n_seasons, len(DAYTYPES),
                         len(REGIMES), len(REGIMES)))
    order = np.lexsort((days["date"].to_numpy(), days["pod"].to_numpy()))
    pod_o = days["pod"].to_numpy()[order]
    date_o = days["date"].to_numpy()[order]
    reg_o, s_o, d_o = regime[order], s_idx[order], d_idx[order]
    c_o = c_idx[order]
    step = (pod_o[1:] == pod_o[:-1]) & (
        (date_o[1:] - date_o[:-1]) == np.timedelta64(1, "D"))
    np.add.at(r_counts, (c_o[1:][step], s_o[1:][step], d_o[1:][step],
                         reg_o[:-1][step], reg_o[1:][step]), 1.0)
    #Lorenzo Giannuzzo: The closure class is dropped last, after the season and the day type, so a
    # thin class leans on the day type of its own class before it leans on the
    # population. Dropping it first would undo the whole point of having it.
    r_prob = backoff(r_counts[:, :, :, None, :, :], [3, 1, 2, 0], k)[:, :, :, 0]

    #Lorenzo Giannuzzo: ── quarter-hourly transitions ───────────────────────────────────────────
    # indexed (regime, season, daytype, block, from x momentum, to)
    counts = np.zeros((len(REGIMES), n_seasons, len(DAYTYPES), n_blocks,
                       S * N_CTX, S))
    src, dst = states[:, :-1], states[:, 1:]
    ok = (src >= 0) & (dst >= 0)
    rows, cols = np.nonzero(ok)
    from_idx = src[rows, cols] * N_CTX + ctx[rows, cols]
    np.add.at(counts,
              (regime[rows], s_idx[rows], d_idx[rows], blocks[cols + 1],
               from_idx, dst[rows, cols]), 1.0)
    #Lorenzo Giannuzzo: regime first, season next, day type after that, the hour of the day last
    probs = backoff(counts, [0, 1, 2, 3], k)

    #Lorenzo Giannuzzo: ── returning from a bin to a value ──────────────────────────────────────
    # The empirical quantiles of what was actually read inside each bin, per
    # block of hours, so that the reconstruction carries the shape of the bin
    # rather than a uniform draw across it.
    n_qt = int(par["n_emission_quantiles"])
    grid = np.linspace(0, 1, n_qt)
    emission = np.zeros((S, n_blocks, n_qt))
    blk = np.broadcast_to(blocks, curves.shape)
    for b in range(1, S):
        sel_b = states == b
        for t in range(n_blocks):
            v = curves[sel_b & (blk == t)]
            if v.size >= 20:
                emission[b, t] = np.quantile(v, grid)
            else:
                v = curves[sel_b]
                emission[b, t] = np.quantile(v, grid) if v.size else 0.0

    #Lorenzo Giannuzzo: Persistence of the position inside the bin. Estimated on consecutive
    # quarters that stayed in the same bin, because across two different bins the
    # two positions are ranks in different distributions and their correlation
    # means nothing: pooling them in dilutes the number toward zero and the
    # generated curve then jumps around inside a bin that can be four times as
    # wide at the bottom as at the top.
    u = np.full(curves.shape, np.nan)
    for b in range(1, S):
        sel_b = states == b
        v = curves[sel_b]
        if v.size >= 20:
            u[sel_b] = np.searchsorted(np.sort(v), v) / max(v.size - 1, 1)
    same = (states[:, :-1] == states[:, 1:]) & (states[:, :-1] > 0)
    a, c = u[:, :-1][same], u[:, 1:][same]
    m = np.isfinite(a) & np.isfinite(c)
    rho = float(np.corrcoef(a[m], c[m])[0, 1]) if m.sum() > 100 else 0.0
    rho = float(np.clip(rho, 0.0, 0.995))

    #Lorenzo Giannuzzo: ── one matrix per real point, for a sample of them ─────────────────────
    # Everything above is the stratum, that is the average of shops open twelve
    # hours a day and of points that never switch off, and that average is
    # neither. A generated point that walks on it comes out in between: its load
    # factor lands in the middle of the distribution and the tail of nearly flat
    # users, which really exists, is never produced. So a sample of the real
    # points keeps its own table, collapsed over regime, season and day type to
    # stay small, carrying what is individual, namely the hours at which that
    # point moves and the levels it moves between. The generator mixes the two,
    # and the sample is spread over the annual energy so that the whole range of
    # behaviours inside the stratum can be drawn.
    n_keep = int(par["n_pod_matrices"])
    order_e = np.argsort(pods["annual_kWh"].fillna(0).to_numpy())
    take = order_e[np.linspace(0, len(order_e) - 1, min(n_keep, len(order_e))
                               ).round().astype(int)]
    take = np.unique(take)
    sample = pods.iloc[take]
    pod_pos = {p: i for i, p in enumerate(sample["pod"])}
    row_pod = days["pod"].map(pod_pos).to_numpy()
    pod_counts = np.zeros((len(sample), n_blocks, S, S), dtype="float32")
    keep_rows = np.isfinite(row_pod.astype("float64"))
    if keep_rows.any():
        sel = keep_rows[rows]
        np.add.at(pod_counts,
                  (row_pod[rows[sel]].astype(int), blocks[cols[sel] + 1],
                   src[rows[sel], cols[sel]], dst[rows[sel], cols[sel]]), 1.0)

    return {
        "edges": edges,
        "pod_counts": pod_counts,
        "sample_pod": sample["pod"].to_numpy().astype(str),
        "sample_power_kW": sample["power_kW"].to_numpy(dtype="float64"),
        "sample_annual_kWh": sample["annual_kWh"].to_numpy(dtype="float64"),
        "trans": probs.astype("float32"),
        "regime_trans": r_prob.astype("float32"),
        "emission": emission.astype("float32"),
        "rho": np.array([rho]),
        "seasons": np.array(season_lab, dtype=object),
        "daytypes": np.array(DAYTYPES, dtype=object),
        "regimes": np.array(REGIMES, dtype=object),
        "n_blocks": np.array([n_blocks]),
        "n_ctx": np.array([N_CTX]),
        "flat_long": np.array([FLAT_LONG]),
        "n_quarters": np.array([n_q]),
        #Lorenzo Giannuzzo: the size anchors a generated profile draws from, so that n profiles are
        # n different users and not n draws from one average user
        # the identity of the real points, so that a generated profile can be
        # anchored to one of them and walk on its own chain rather than on the
        # average of the stratum
        "pods": pods["pod"].to_numpy().astype(str),
        #Lorenzo Giannuzzo: the closure class of each of those points, in the same order, so that a
        # generated profile anchored to one of them walks the daily chain of its
        # own kind rather than that of the pooled population
        "closure_class": pods["pod"].map(cls_by_pod).fillna(0)
                              .to_numpy().astype("int8"),
        "closure_cuts": c_cuts,
        "power_kW": pods["power_kW"].to_numpy(dtype="float64"),
        "annual_kWh": pods["annual_kWh"].to_numpy(dtype="float64"),
        "meta": np.array([json.dumps({
            "n_pods": int(len(pods)),
            "n_pod_days": int(len(days)),
            "n_states": int(S),
            "n_bins_effective": int(B),
            "zero_day_share": float((energy <= ZERO_EPS).mean()),
            #Lorenzo Giannuzzo: the pooled share above says how many days of the stratum are at
            # zero, the median below says how many days the typical point of it
            # spends at zero. When the two disagree the zero days belong to a
            # minority of the points, which is what the closure class carries.
            "zero_day_share_median_pod": float(zero_by_pod.median()),
            "closure_class_shares": ",".join(
                f"{100 * float((cls_by_pod == c).mean()):.0f}%"
                for c in range(n_cls)),
            "rho_within_bin": rho,
        })], dtype=object),
    }


def size_strata(pods: pd.DataFrame, max_strata: int,
                min_pods: int) -> tuple[np.ndarray, str, np.ndarray]:
    """Split a typology by size, and say what the split was made on.

    A shop of three kilowatts and one of fifty do not share a distribution of
    levels, and a chain estimated on both mixes them. Contractual power is the
    natural variable, since it is also what a caller asks for, but it does not
    always discriminate: nearly every domestic point in this archive is declared
    at three kilowatts, so its quantiles collapse onto one value and the split
    never happens. Where that is the case the annual energy is used instead, and
    the manifest records which of the two the model rests on.

    The number of strata is adaptive, roughly one per thirty points, and it is
    reduced until no stratum falls below the minimum: a stratum of five points is
    not a model, it is noise with a file name.
    """
    def cut(values: pd.Series, n: int) -> np.ndarray:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        return np.unique(np.quantile(values.dropna(), qs))

    n_max = int(np.clip(len(pods) // 30, 1, max_strata))
    for n in range(n_max, 1, -1):
        for var in ("power_kW", "annual_kWh"):
            values = pods[var]
            if values.notna().sum() < min_pods * n:
                continue
            edges = cut(values, n)
            if len(edges) != n - 1:
                #Lorenzo Giannuzzo: the cuts fell on the same number, which is what happens when
                # nearly every point declares the same power: that variable
                # cannot split this typology, so the next one is tried before
                # giving up a stratum
                continue
            labels = np.digitize(values.fillna(values.median()).to_numpy(), edges)
            sizes = np.bincount(labels, minlength=n)
            if sizes.min() >= min_pods:
                return edges, var, labels
    return np.array([]), "none", np.zeros(len(pods), dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate the transition model.")
    ap.add_argument("--level", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--typology", type=str, default=None)
    ap.add_argument("--all", action="store_true",
                    help="every typology that meets the estimation thresholds")
    args = ap.parse_args()
    if not args.all and not args.typology:
        ap.error("give --typology or --all")

    cfg = load_config()
    est = cfg.get("estimation", {}) or {}
    par = {
        "n_bins": est.get("n_bins", 12),
        "n_blocks": est.get("n_blocks", 6),
        "shrinkage": est.get("shrinkage", 30),
        "n_emission_quantiles": est.get("n_emission_quantiles", 25),
        "closure_cuts": est.get("closure_cuts", [0.02, 0.20]),
        "max_strata": est.get("max_size_strata", 3),
        "n_pod_matrices": est.get("n_pod_matrices", 60),
    }
    min_days = int(est.get("min_valid_days", 180))
    min_pods = int(est.get("min_pods", 10))
    min_month = int(est.get("min_pod_days_per_month", 200))
    outdir = cfg.models_dir

    days_all = pd.read_parquet(cfg.cache_dir / "days.parquet")
    users = pd.read_parquet(cfg.cache_dir / "users.parquet")
    curves_all = np.load(cfg.cache_dir / "curves.npy", mmap_mode="r")
    col = level_column(args.level)

    print(f"\n{'=' * 78}\nSTAGE 3 — ESTIMATION, level {args.level}\n{'=' * 78}\n")

    valid = days_all[days_all["valid"]]
    per_pod = valid.groupby("pod").agg(n_days_valid=("date", "nunique"),
                                       annual_kWh=("energy", "sum"))
    users = users.drop(columns=["n_days_read", "n_days_valid", "annual_kWh"],
                       errors="ignore").merge(per_pod, on="pod", how="left")
    users["annual_kWh"] = users["annual_kWh"] / users["n_days_valid"] * 365.0
    pool_users = users[users["n_days_valid"].fillna(0) >= min_days]
    print(f"  estimation pool: {len(pool_users):,} points with >= {min_days} valid days")

    if args.all:
        wanted = [t for t, n in census(pool_users, args.level)["n_pods"].items()
                  if n >= min_pods]
    else:
        wanted = [normalise_typology(args.typology, args.level)]

    valid_idx = days_all["valid"].to_numpy()
    manifest = []
    for typ in wanted:
        sel_pods = pool_users[pool_users[col] == typ]
        if len(sel_pods) < min_pods:
            print(f"  {typ:<12s} SKIPPED, {len(sel_pods)} points, "
                  f"{min_pods} required")
            continue
        edges_p, strat_var, strata = size_strata(
            sel_pods, int(par["max_strata"]), min_pods)

        for s in np.unique(strata):
            pods_s = sel_pods[strata == s]
            if len(pods_s) < min_pods:
                print(f"  L{args.level}_{typ}_s{int(s)} SKIPPED, "
                      f"{len(pods_s)} points in the stratum")
                continue
            mask = valid_idx & days_all["pod"].isin(set(pods_s["pod"])).to_numpy()
            d = days_all[mask].reset_index(drop=True)
            c = np.asarray(curves_all[mask], dtype="float64")
            months = d.groupby("month").size()
            thin = [int(m) for m in range(1, 13)
                    if int(months.get(m, 0)) < min_month]
            key = f"L{args.level}_{typ}" + (f"_s{int(s)}" if len(edges_p) else "")
            try:
                model = estimate_one(c, d, pods_s, par)
            except ValueError as exc:
                print(f"  {key:<20s} SKIPPED, {exc}")
                continue
            np.savez_compressed(outdir / f"{key}.npz", **model)
            info = json.loads(model["meta"][0])
            manifest.append({
                "key": key, "level": args.level, "typology": typ,
                "size_stratum": int(s),
                "stratified_on": strat_var,
                "power_kW_min": float(pods_s["power_kW"].min()),
                "power_kW_max": float(pods_s["power_kW"].max()),
                "annual_kWh_min": float(pods_s["annual_kWh"].min()),
                "annual_kWh_max": float(pods_s["annual_kWh"].max()),
                "n_pods": info["n_pods"], "n_pod_days": info["n_pod_days"],
                "n_states": info["n_states"],
                "zero_day_share": round(info["zero_day_share"], 4),
                "zero_day_share_median_pod": round(
                    info["zero_day_share_median_pod"], 4),
                "closure_class_shares": info["closure_class_shares"],
                "rho_within_bin": round(info["rho_within_bin"], 3),
                "thin_months": ",".join(map(str, thin)),
            })
            print(f"  {key:<20s} {info['n_pods']:>4} points  "
                  f"{info['n_pod_days']:>7,} days  {info['n_states']:>3} states  "
                  f"rho {info['rho_within_bin']:.2f}  "
                  f"zero days {100 * info['zero_day_share']:.0f}%"
                  + (f"  THIN MONTHS {thin}" if thin else ""))

    if manifest:
        pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
        print(f"\n  {len(manifest)} models written to results/models/\n")
    else:
        print("\n  no model met the thresholds\n")


if __name__ == "__main__":
    main()