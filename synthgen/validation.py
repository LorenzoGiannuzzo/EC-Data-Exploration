"""Stage 5 — Validation.

Six comparisons between what was generated and the real points of the same
typology, written as figures in the output directory.

The point of each one, briefly. The distribution of daily energy says whether the
generated points consume the right amounts; the load factor says whether they
consume them in a plausible shape, since a flat curve and a peaky one can carry
the same total. The hour of the peak is the cheapest test of whether the daily
rhythm survived. The autocorrelation is the test the model is most likely to fail:
a chain that only knows its level forgets too quickly, and the curve of the
synthetic falls below the real one long before it should. The mean day by season
and day type checks the seasonality the conditioning was supposed to carry. The
last one is not a comparison with reality but among the generated points
themselves: n profiles that differ only by noise would be useless whatever their
average looks like.

Three numbers sit beside the figures without one of their own, because they are
what the figures cannot show. Day to day repeatability is the distance between
the shape of a day and the shape of the next, and it separates a point that keeps
its own schedule from one that redraws itself every midnight; a mean day can be
exact while this number is twice what it should be. The share of the year at zero
and the ramp percentiles say how much of the time the point stands still and how
violently it moves when it does not.

The autocorrelation is reported at one day and at one week. Half a day, which the
first version of this file used, sits at the antiphase of the daily cycle and
reads close to zero on any curve whatever its rhythm.

Total variation between the mean curves closes the set with a single number, on
the same convention used elsewhere, one half of the sum of absolute differences
between two distributions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .calendar import annotate

TZ = "Europe/Rome"

DPI = 150
#Lorenzo Giannuzzo: Points used for the per point metrics on a large typology. They are medians
# over the points, so a few hundred settle them, and the domestic typology
# holds ten thousand of them.
MAX_PODS_METRICS = 500
#Lorenzo Giannuzzo: fixed so that a panel with no metered data does not hand the metered colour to
# the synthetic curve and make two panels of the same figure disagree
C_REAL, C_SYN = "#1f77b4", "#d62728"


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    """One half of the sum of absolute differences between two distributions."""
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(0.5 * np.abs(a / sa - b / sb).sum())


def daily_matrix(values: np.ndarray, n_per_day: int) -> np.ndarray:
    usable = (len(values) // n_per_day) * n_per_day
    return values[:usable].reshape(-1, n_per_day)


def contiguous_runs(dates: np.ndarray) -> list[np.ndarray]:
    """Positions of the maximal runs of consecutive calendar days.

    Preprocessing drops the days it cannot complete, so the rows of one point
    are not a contiguous stretch of time. Raveling them treats a February day
    as if it followed a January one, and every lag beyond a few hours is then
    measured across a seam that no meter ever saw.
    """
    order = np.argsort(dates, kind="stable")
    d = dates[order]
    if len(d) == 0:
        return []
    brk = np.where((d[1:] - d[:-1]) != np.timedelta64(1, "D"))[0] + 1
    return [g for g in np.split(order, brk) if len(g)]


def acf_runs(mat: np.ndarray, runs: list[np.ndarray],
             lags: np.ndarray) -> np.ndarray:
    """Autocorrelation of one point, counting only pairs inside a run.

    The mean and the variance are those of the whole point, so the estimator
    stays the usual biased one and the lags remain comparable with each other.
    What changes is the numerator, which never straddles a gap.
    """
    if not runs:
        return np.full(len(lags), np.nan)
    whole = np.concatenate([mat[r].ravel() for r in runs])
    mu = float(whole.mean())
    var = float(np.dot(whole - mu, whole - mu))
    if var <= 0:
        return np.full(len(lags), np.nan)
    segments = [mat[r].ravel() - mu for r in runs]
    out = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        num, pairs = 0.0, 0
        for seg in segments:
            if len(seg) > lag:
                num += float(np.dot(seg[:-lag], seg[lag:]))
                pairs += len(seg) - lag
        if pairs > 0:
            out[i] = num / var
    return out


def daily_frames(one: pd.DataFrame, n_per_day: int) -> tuple[np.ndarray, np.ndarray]:
    """One profile as a (n_days, n_per_day) matrix, with its dates.

    Built by grouping on the civil date rather than by reshaping the year.
    A tz aware year holds 92 quarter-hours in March and 100 in October, so a
    reshape stays aligned only until the first daylight saving day and is off
    by four slots for the seven months that follow.
    """
    one = one.sort_values("timestamp")
    slot = one.groupby("date").cumcount()
    piv = one.assign(slot=slot).pivot(index="date", columns="slot", values="kWh")
    if piv.shape[1] > n_per_day:
        piv = piv.iloc[:, :n_per_day]
    piv = piv.dropna(axis=0, how="any")
    return piv.to_numpy(dtype="float64"), piv.index.to_numpy()


def repeatability(mat: np.ndarray, dates: np.ndarray) -> float:
    """Median total variation between the shape of a day and of the next one.

    Only consecutive days count. A point that opens at the same hour every
    morning scores low whatever its level, and a chain that redraws its day
    from the stratum every midnight scores high however well its mean day
    matches. It is the one number the mean day cannot see.
    """
    if len(dates) < 2:
        return np.nan
    order = np.argsort(dates, kind="stable")
    mat, dates = mat[order], dates[order]
    step = (dates[1:] - dates[:-1]) == np.timedelta64(1, "D")
    if not step.any():
        return np.nan
    a, b = mat[:-1][step], mat[1:][step]
    sa, sb = a.sum(axis=1), b.sum(axis=1)
    ok = (sa > 0) & (sb > 0)
    if not ok.any():
        return np.nan
    tv = 0.5 * np.abs(a[ok] / sa[ok, None] - b[ok] / sb[ok, None]).sum(axis=1)
    return float(np.median(tv))


def ramps(mat: np.ndarray) -> np.ndarray:
    """Absolute step to step differences of one point, as a share of its peak.

    Normalising on the peak lets points of very different size be pooled, and
    turns the result into a reading of how spiky a curve is rather than of how
    large it is. Steps are taken inside a day, so no ramp crosses a gap.
    """
    peak = float(mat.max()) if mat.size else 0.0
    if peak <= 0:
        return np.array([])
    return np.abs(np.diff(mat, axis=1)).ravel() / peak


def load_real(cfg, keys: pd.DataFrame,
              models_dir: Path | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """The metered days of the points the chosen models were estimated on.

    The points of the strata that were actually used, read from the models
    themselves, and not every point that carries the code. The two are not the
    same population: a typology holds points that never entered the estimation
    pool, and the anchors a generated profile is drawn from come from the pool
    alone. Comparing against the wider set puts a difference of composition into
    every figure and lets it be read as a defect of the model.

    Falls back on the whole typology when the models cannot be found, so that a
    validation run against a directory of generated files still works.
    """
    days = pd.read_parquet(cfg.cache_dir / "days.parquet")
    users = pd.read_parquet(cfg.cache_dir / "users.parquet")
    curves = np.load(cfg.cache_dir / "curves.npy", mmap_mode="r")
    level = int(keys["level"].iloc[0])
    typ = str(keys["typology"].iloc[0])

    models_dir = Path(models_dir) if models_dir else cfg.models_dir
    pods: set[str] = set()
    for key in keys["key"].astype(str) if "key" in keys else []:
        path = models_dir / f"{key}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as z:
            if "pods" in z.files:
                pods |= {str(p) for p in z["pods"]}
    source = f"{len(pods)} points of the strata used"
    if not pods:
        pods = set(users.loc[users[f"ateco_l{level}"].astype(str) == typ, "pod"])
        source = f"{len(pods)} points of typology {typ}, models not found"
    print(f"  metered reference: {source}")

    mask = days["valid"].to_numpy() & days["pod"].isin(pods).to_numpy()
    return np.asarray(curves[mask], dtype="float64"), days[mask].reset_index(drop=True)


def load_synth(man: pd.DataFrame, out: Path) -> tuple[np.ndarray, pd.DataFrame]:
    frames = []
    for name in man["file"]:
        f = pd.read_csv(out / name)
        #Lorenzo Giannuzzo: the offset changes at the daylight saving boundary, so the column holds
        # two offsets and only a pass through UTC parses it as a single dtype
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True).dt.tz_convert(TZ)
        f["profile"] = name
        frames.append(f)
    s = pd.concat(frames, ignore_index=True)
    s["date"] = pd.to_datetime(s["timestamp"].dt.tz_localize(None).dt.date)
    return s["kWh"].to_numpy(), s


def validate(man: pd.DataFrame, keys: pd.DataFrame, out: Path, cfg,
             year: int) -> pd.DataFrame:
    vdir = out / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    per_day_r = 96
    real, rdays = load_real(cfg, keys, cfg.models_dir)
    _, synth = load_synth(man, out)
    step_h = 0.25 if len(synth) / max(man.shape[0], 1) > 10000 else 1.0
    per_day_s = int(round(24 / step_h))

    sd = (synth.groupby(["profile", "date"])["kWh"]
                .agg(energy="sum", peak="max").reset_index())
    sd = annotate(sd, cfg["preprocessing"]["seasons"])
    rd = rdays.assign(energy=real.sum(axis=1), peak=real.max(axis=1))

    #Lorenzo Giannuzzo: The metered side is not rescaled. Every generated profile carries the annual
    # energy of the real point it was anchored to, and those points are the ones
    # loaded above, so the two populations already stand at the same size and any
    # remaining difference is a result rather than a nuisance.
    #
    # Rescaling was worse than doing nothing in both of the ways it was tried. On
    # the mean, the metered reference followed whichever profile the synthetic run
    # happened to draw, and the reported median daily energy of the metered side
    # moved from 11.9 to 27.5 kWh between two runs in which the metered data had
    # not changed at all. On the median, the check that compares the two medians
    # of daily energy became an identity, since that is the quantity the factor
    # was fitted on, and it reported a perfect agreement it could not fail to
    # report. The ratio is printed instead, as the reading it always was.
    ratio = float(np.median(sd["energy"])) / max(float(np.median(rd["energy"])), 1e-9)
    print(f"  synthetic median day is {ratio:.3f} times the metered one")
    scale = 1.0
    summary = []

    #Lorenzo Giannuzzo: 1 ── daily energy ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    hi = float(np.nanpercentile(sd["energy"], 99.5))
    bins = np.linspace(0, max(hi, 1e-6), 60)
    ax.hist(rd["energy"] * scale, bins=bins, density=True, alpha=0.55, color=C_REAL,
            label=f"metered ({len(rd):,} days)")
    ax.hist(sd["energy"], bins=bins, density=True, alpha=0.55, color=C_SYN,
            label=f"synthetic ({len(sd):,} days)")
    ax.set_xlabel("daily energy (kWh)"); ax.set_ylabel("density")
    ax.set_title("Daily energy"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(vdir / "01_daily_energy.png", dpi=DPI)
    plt.close(fig)
    summary.append({"check": "median daily energy",
                    "metered": float(np.median(rd["energy"] * scale)),
                    "synthetic": float(np.median(sd["energy"]))})

    #Lorenzo Giannuzzo: 2 ── load factor -------------------------------------------------------
    lf_r = (rd["energy"] / per_day_r) / rd["peak"].replace(0, np.nan)
    lf_s = (sd["energy"] / per_day_s) / sd["peak"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 50)
    ax.hist(lf_r.dropna(), bins=bins, density=True, alpha=0.55, color=C_REAL, label="metered")
    ax.hist(lf_s.dropna(), bins=bins, density=True, alpha=0.55, color=C_SYN, label="synthetic")
    ax.set_xlabel("load factor (mean over peak)"); ax.set_ylabel("density")
    ax.set_title("Load factor"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(vdir / "02_load_factor.png", dpi=DPI)
    plt.close(fig)
    #Lorenzo Giannuzzo: 1b ── annual energy per point ------------------------------------------
    # The daily figure above is a median over days and says nothing about how the
    # year adds up for one point, which is the number a network study is sized on
    # and the one an aggregate simulation gets wrong first. A metered point is
    # carried to a full year before the two are compared, since preprocessing
    # drops the days it cannot complete and a point with three hundred valid days
    # would otherwise look like a smaller consumer than it is.
    r_days = rd.groupby("pod")["date"].nunique()
    r_year = (rd.groupby("pod")["energy"].sum() * scale
              / r_days.replace(0, np.nan) * 365.0).dropna()
    s_year = sd.groupby("profile")["energy"].sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    lo = max(min(float(r_year.min()), float(s_year.min())), 1.0)
    hi = max(float(r_year.max()), float(s_year.max()))
    bins = np.geomspace(lo, hi * 1.05, 45)
    ax.hist(r_year, bins=bins, density=True, alpha=0.55, color=C_REAL,
            label=f"metered ({len(r_year):,} points)")
    ax.hist(s_year, bins=bins, density=True, alpha=0.55, color=C_SYN,
            label=f"synthetic ({len(s_year):,} profiles)")
    ax.set_xscale("log")
    ax.set_xlabel("annual energy per point (kWh)"); ax.set_ylabel("density")
    ax.set_title("Annual energy"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(vdir / "09_annual_energy.png", dpi=DPI)
    plt.close(fig)

    summary.append({"check": "median annual energy (kWh)",
                    "metered": float(np.median(r_year)),
                    "synthetic": float(np.median(s_year))})
    #Lorenzo Giannuzzo: The spread of the population, as the ratio of its ninth decile to its
    # first. A generated set can carry the right median and still be far too
    # wide, which is what a single profile holding most of the energy looks
    # like before anyone notices it.
    def decile_spread(v) -> float:
        lo_, hi_ = np.percentile(v, [10, 90])
        return float(hi_ / lo_) if lo_ > 0 else float("nan")
    summary.append({"check": "annual energy p90 / p10",
                    "metered": decile_spread(r_year),
                    "synthetic": decile_spread(s_year)})

    summary.append({"check": "median load factor",
                    "metered": float(lf_r.median()),
                    "synthetic": float(lf_s.median())})

    #Lorenzo Giannuzzo: 3 ── hour of the peak --------------------------------------------------
    hr_r = np.argmax(real, axis=1) * 24 / per_day_r
    hour_s = synth["timestamp"].dt.hour + synth["timestamp"].dt.minute / 60
    idx = synth.groupby(["profile", "date"])["kWh"].idxmax()
    hr_s = hour_s.loc[idx].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, 25, 1)
    ax.hist(hr_r, bins=bins, density=True, alpha=0.55, color=C_REAL, label="metered")
    ax.hist(hr_s, bins=bins, density=True, alpha=0.55, color=C_SYN, label="synthetic")
    ax.set_xlabel("hour of the daily peak"); ax.set_ylabel("density")
    ax.set_xticks(range(0, 25, 3)); ax.set_title("When the peak falls")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(vdir / "03_peak_hour.png", dpi=DPI)
    plt.close(fig)

    #Lorenzo Giannuzzo: 4 ── autocorrelation ---------------------------------------------------
    # Both sides are read one point at a time and averaged over points, never on
    # the stacked table: consecutive rows there often belong to different points
    # at very different levels, and the jump between them shows up as a loss of
    # correlation that has nothing to do with any meter. On the metered side the
    # rows of a single point are not contiguous either, since preprocessing drops
    # the days it cannot complete, so the pairs are counted inside runs of
    # consecutive days alone. The two sides then measure the same quantity.
    rpods_sample = pd.Series(rdays["pod"].unique()).head(12).tolist()
    #Lorenzo Giannuzzo: A geometric grid alone almost never lands on the daily lag, and on a curve
    # with square fronts the nearest grid point falls in a trough of the
    # correlogram: the exact multiples of a day are therefore forced into it, so
    # that the number reported at 24 and 168 hours is read at 24 and 168 hours.
    def lag_grid(n_per_day: int) -> np.ndarray:
        geo = np.round(np.geomspace(1, n_per_day * 8, 28)).astype(int)
        exact = n_per_day * np.array([1, 2, 3, 7])
        return np.unique(np.concatenate([geo, exact]))

    lags_r, lags_s = lag_grid(per_day_r), lag_grid(per_day_s)
    rdates = rdays["date"].to_numpy().astype("datetime64[D]")

    #Lorenzo Giannuzzo: The row positions of every point, resolved once. Asking for them inside the
    # loop instead, with a comparison against the whole column, costs one pass
    # over the table per point: on typology 47 that is 171 points against 55,873
    # rows and goes unnoticed, on the domestic typology it is 10,009 against
    # 3,520,454 and comes to thirty five billion comparisons.
    pod_rows = {str(k): np.asarray(v) for k, v in
                rdays.groupby("pod", sort=False).indices.items()}

    acf_r = []
    for pod in rpods_sample:
        rows = pod_rows.get(str(pod), np.empty(0, dtype=int))
        if len(rows) >= 30:
            runs = [rows[g] for g in contiguous_runs(rdates[rows])]
            acf_r.append(acf_runs(real, runs, lags_r))

    acf_s, synth_daily = [], {}
    for name in man["file"]:
        mat, dts = daily_frames(synth[synth["profile"] == name], per_day_s)
        if not len(mat):
            continue
        synth_daily[name] = (mat, dts.astype("datetime64[D]"))
        if len(acf_s) < 8:
            runs = [g for g in contiguous_runs(dts.astype("datetime64[D]"))]
            acf_s.append(acf_runs(mat, runs, lags_s))

    fig, ax = plt.subplots(figsize=(7, 4))
    if acf_r:
        ax.plot(lags_r * 24 / per_day_r, np.nanmean(acf_r, axis=0), marker="o",
                ms=3, color=C_REAL, label=f"metered ({len(acf_r)} points)")
    if acf_s:
        ax.plot(lags_s * 24 / per_day_s, np.nanmean(acf_s, axis=0), marker="s",
                ms=3, color=C_SYN, label=f"synthetic ({len(acf_s)} points)")
    ax.axhline(0, color="0.6", lw=0.8)
    for h in (24, 168):
        ax.axvline(h, color="0.85", lw=0.8, zorder=0)
    ax.set_xscale("log"); ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorrelation"); ax.set_title("Persistence")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(vdir / "04_autocorrelation.png", dpi=DPI)
    plt.close(fig)

    #Lorenzo Giannuzzo: Reported at one day and at one week, where the daily habit and the weekly
    # one live. Half a day sits at the antiphase of the daily cycle, so it reads
    # close to zero on any curve whatever its rhythm, and says nothing.
    if acf_r and acf_s:
        mr, ms_ = np.nanmean(acf_r, axis=0), np.nanmean(acf_s, axis=0)
        for hours in (24, 168):
            i = int(np.argmin(np.abs(lags_r * 24 / per_day_r - hours)))
            j = int(np.argmin(np.abs(lags_s * 24 / per_day_s - hours)))
            summary.append({"check": f"autocorrelation at {hours} hours",
                            "metered": float(mr[i]),
                            "synthetic": float(ms_[j])})

    #Lorenzo Giannuzzo: 4b ── repeatability, idle time and spikiness ---------------------------
    # Three numbers the distributions above cannot produce. The first says
    # whether a point repeats its own routine, which is what separates a curve
    # with square fronts from one that redraws itself every midnight. The second
    # says how much of the year it stands at zero. The third says how violently
    # it moves when it moves.
    # All three are medians over the points, so on a large typology they are read
    # on a sample of them rather than on every one. The sample is taken at an even
    # stride over the points as they are ordered in the table, which is by code,
    # so it is the same sample on every run and is not drawn towards the points
    # that happen to sit at the front. Below the cap nothing is sampled and the
    # numbers are the ones every earlier run reported.
    #
    # The ramps are pooled across the sampled points before their percentiles are
    # taken, which is why the cap matters twice: the domestic typology would
    # otherwise pool 334 million of them, or 2.7 GB held at once.
    all_pods = list(pod_rows.keys())
    if len(all_pods) > MAX_PODS_METRICS:
        stride = len(all_pods) / MAX_PODS_METRICS
        sampled = [all_pods[int(i * stride)] for i in range(MAX_PODS_METRICS)]
        print(f"  per point metrics read on {len(sampled):,} of "
              f"{len(all_pods):,} metered points")
    else:
        sampled = all_pods

    rep_r, zero_r, ramp_r = [], [], []
    for pod in sampled:
        rows = pod_rows[pod]
        if len(rows) < 30:
            continue
        mat = real[rows]
        rep_r.append(repeatability(mat, rdates[rows]))
        zero_r.append(float((mat <= 0).mean()))
        ramp_r.append(ramps(mat))

    rep_s, zero_s, ramp_s = [], [], []
    for name, (mat, dts) in synth_daily.items():
        rep_s.append(repeatability(mat, dts))
        zero_s.append(float((mat <= 0).mean()))
        ramp_s.append(ramps(mat))

    def med(v: list) -> float:
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if v else float("nan")

    def ramp_p(v: list, q: float) -> float:
        v = [a for a in v if len(a)]
        return float(np.percentile(np.concatenate(v), q)) if v else float("nan")

    summary.append({"check": "day to day repeatability (TV)",
                    "metered": med(rep_r), "synthetic": med(rep_s)})
    summary.append({"check": "share of the year at zero",
                    "metered": med(zero_r), "synthetic": med(zero_s)})
    for q in (90, 99):
        summary.append({"check": f"ramp p{q} (share of peak)",
                        "metered": ramp_p(ramp_r, q),
                        "synthetic": ramp_p(ramp_s, q)})

    #Lorenzo Giannuzzo: 5 ── the mean day, by season and day type ------------------------------
    seasons = [s for s in cfg["preprocessing"]["seasons"]]
    dts = ["weekday", "saturday", "sunday"]
    fig, axes = plt.subplots(len(dts), len(seasons),
                             figsize=(3.2 * len(seasons), 2.3 * len(dts)),
                             sharex=True, squeeze=False)
    tv_rows = []
    for r, dt in enumerate(dts):
        for c, se in enumerate(seasons):
            ax = axes[r][c]
            mr = real[(rdays["season"] == se) & (rdays["daytype"] == dt)]
            ms = synth[(synth["date"].isin(
                sd.loc[(sd["season"] == se) & (sd["daytype"] == dt), "date"]))]
            if len(mr):
                x = np.arange(per_day_r) * 24 / per_day_r
                mu = mr.mean(axis=0) * scale
                ax.plot(x, mu, lw=1.2, color=C_REAL, label="metered")
                ax.fill_between(x, np.percentile(mr, 25, axis=0) * scale,
                                np.percentile(mr, 75, axis=0) * scale, alpha=0.2,
                                color=C_REAL)
            if len(ms):
                g = ms.groupby(ms["timestamp"].dt.hour
                               + ms["timestamp"].dt.minute / 60)["kWh"].mean()
                ax.plot(g.index.to_numpy(), g.to_numpy(), lw=1.2, color=C_SYN,
                        label="synthetic")
                if len(mr):
                    ref = np.interp(g.index.to_numpy(),
                                    np.arange(per_day_r) * 24 / per_day_r, mu)
                    tv_rows.append({"season": se, "daytype": dt,
                                    "total_variation": total_variation(
                                        ref, g.to_numpy())})
            if r == 0:
                ax.set_title(se)
            if c == 0:
                ax.set_ylabel(dt)
    fig.legend(handles=[Line2D([], [], color=C_REAL, label="metered"),
                        Line2D([], [], color=C_SYN, label="synthetic")],
               loc="upper right", frameon=False, fontsize=8)
    fig.supxlabel("hour of the day"); fig.tight_layout()
    fig.savefig(vdir / "05_mean_day.png", dpi=DPI); plt.close(fig)

    #Lorenzo Giannuzzo: 6 ── diversity among the generated points ------------------------------
    prof_mean = (synth.groupby(["profile", synth["timestamp"].dt.hour])["kWh"]
                       .mean().unstack().to_numpy())
    prof_mean = prof_mean / np.maximum(prof_mean.sum(axis=1, keepdims=True), 1e-9)
    dsyn = [total_variation(prof_mean[i], prof_mean[j])
            for i in range(len(prof_mean)) for j in range(i + 1, len(prof_mean))]
    hourly_r = real.reshape(len(real), 24, -1).sum(axis=2)
    by_pod = pd.DataFrame(hourly_r).groupby(rdays["pod"].to_numpy()).mean().to_numpy()
    by_pod = by_pod / np.maximum(by_pod.sum(axis=1, keepdims=True), 1e-9)
    sub = by_pod[:80]
    dreal = [total_variation(sub[i], sub[j])
             for i in range(len(sub)) for j in range(i + 1, len(sub))]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(max(dreal, default=0.5), max(dsyn, default=0.5)), 50)
    ax.hist(dreal, bins=bins, density=True, alpha=0.55, color=C_REAL,
            label="between metered points")
    ax.hist(dsyn, bins=bins, density=True, alpha=0.55, color=C_SYN,
            label="between synthetic points")
    ax.set_xlabel("total variation between mean daily shapes")
    ax.set_ylabel("density"); ax.set_title("Diversity")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(vdir / "06_diversity.png", dpi=DPI)
    plt.close(fig)
    summary.append({"check": "median distance between points",
                    "metered": float(np.median(dreal)) if dreal else np.nan,
                    "synthetic": float(np.median(dsyn)) if dsyn else np.nan})

    #Lorenzo Giannuzzo: 7 ── the profiles themselves, side by side -----------------------------
    # The distributions above can all agree while the curves still look wrong, so
    # a week of each is drawn as it is, one metered point and one generated point
    # per row, in winter and in summer.
    # paired by annual energy: putting the first metered point next to the first
    # generated one compares a large shop with a small one and reads as a defect
    # of the model when it is only a defect of the pairing
    # The real point shown next to a generated one is the one closest to it in
    # annual energy. Pairing them by position instead puts a point drawing five
    # times the other beside it, and the panel then says nothing about shape.
    n_show = min(4, len(man))
    r_energy = rd.groupby("pod")["energy"].sum() * scale
    s_energy = sd.groupby("profile")["energy"].sum()
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 1.9 * n_show), squeeze=False,
                             sharex="col")
    for r in range(n_show):
        prof_name = man["file"].iloc[r]
        target = float(s_energy.get(prof_name, np.nan))
        pod = (r_energy.sub(target).abs().idxmin()
               if np.isfinite(target) and len(r_energy) else None)
        for c, (se, ttl) in enumerate([(seasons[0], "a winter week"),
                                       (seasons[-1], "a summer week")]):
            ax = axes[r][c]
            if pod is not None:
                sel = np.where((rdays["season"] == se).to_numpy()
                               & (rdays["pod"] == pod).to_numpy())[0]
                if len(sel) >= 7:
                    block = real[sel[:7]].ravel() * scale
                    ax.plot(np.arange(len(block)) / per_day_r, block, lw=0.8,
                            color=C_REAL)
            prof = synth[synth["profile"] == prof_name]
            ps = prof[prof["date"].isin(
                sd.loc[sd["season"] == se, "date"])]["kWh"].to_numpy()
            if len(ps) >= 7 * per_day_s:
                ax.plot(np.arange(7 * per_day_s) / per_day_s,
                        ps[:7 * per_day_s], lw=0.8, color=C_SYN)
            ax.set_ylabel("kWh", fontsize=8)
            ax.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(ttl, fontsize=10)
            if r == n_show - 1:
                ax.set_xlabel("days")
    fig.legend(handles=[Line2D([], [], color=C_REAL, label="metered"),
                        Line2D([], [], color=C_SYN, label="synthetic")],
               loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(vdir / "07_example_weeks.png", dpi=DPI)
    plt.close(fig)

    #Lorenzo Giannuzzo: 8 ── load duration curve -----------------------------------------------
    # Sorted from the highest quarter to the lowest. For a network this is the
    # most direct reading of the two: it shows at once whether the peaks are of
    # the right height and how long the load stays near them.
    # Read on a grid of quantiles rather than by sorting and drawing every
    # quarter-hour of the population. The two are the same curve, since the
    # duration curve is the quantile function reversed, but the sort is not: the
    # domestic typology holds 3,520,454 metered days, which is 337,963,584
    # values, and asking matplotlib to transform that many points for a symlog
    # axis asks it for 2.5 GiB in one array on top of the sort itself. A grid of
    # a few thousand points draws the identical curve at the width of a figure,
    # where 338 million points would in any case fall on top of one another.
    #
    # The grid is denser at the two ends, where the curve turns: the top per
    # cent carries the peaks a network is sized on, and the bottom carries the
    # share of the year at zero.
    def duration_curve(values: np.ndarray, factor: float,
                       n_points: int = 3000) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype="float64").ravel()
        edge = np.geomspace(1e-3, 50.0, n_points // 2)
        share = np.unique(np.concatenate([edge, 100.0 - edge[::-1], [0.0, 100.0]]))
        levels = np.percentile(values, 100.0 - share) * factor
        return share, levels

    fig, ax = plt.subplots(figsize=(7, 4))
    xr, yr = duration_curve(real, scale * 4)
    xs, ys = duration_curve(synth["kWh"].to_numpy(), 4 if per_day_s == 96 else 1)
    ax.plot(xr, yr, color=C_REAL, lw=1.2, label="metered")
    ax.plot(xs, ys, color=C_SYN, lw=1.2, label="synthetic")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("share of the year above this level (%)")
    ax.set_ylabel("power (kW)"); ax.set_title("Load duration")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(vdir / "08_load_duration.png", dpi=DPI)
    plt.close(fig)
    summary.append({"check": "99th percentile power (kW)",
                    "metered": float(np.percentile(real, 99) * scale * 4),
                    "synthetic": float(np.percentile(
                        synth["kWh"].to_numpy(), 99) * (4 if per_day_s == 96 else 1))})

    tv = pd.DataFrame(tv_rows)
    if len(tv):
        tv.to_csv(vdir / "total_variation_by_cell.csv", index=False)
        summary.append({"check": "total variation of the mean day",
                        "metered": 0.0,
                        "synthetic": float(tv["total_variation"].mean())})
    out_sum = pd.DataFrame(summary)
    out_sum.to_csv(vdir / "summary.csv", index=False)
    print(f"\n  validation written to {vdir}")
    print(out_sum.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    return out_sum