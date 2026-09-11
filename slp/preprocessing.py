"""Stage 1 — Pre-processing (Section 2.2).

Reads the monthly folders, applies the six filtering operations in the order the
paper declares, and factorizes every surviving daily curve into

    p(i,d,t) = E(i) * w(i,d) * s(i,d,t)                                  [Eq. 2]

Outputs
    cache/shapes.npy                (n_days, 96) float32, unit integral
    cache/days.parquet              pod, date, w, season, daytype, energy
    cache/users.parquet             pod, E, ateco levels, tariff, power
    paper_results/preprocessing_results/
        funnel.csv                  what each filter removed  -> Section 3.1
        composition_by_class.csv    PODs per activity class    -> Section 3.1
        composition_by_tariff.csv
        summary.txt                 [N] and [YYYY] for Section 2.2

Run
    python preprocessing.py

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
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.io import Q_COLS, load_year, split_ateco, to_number, _q_columns

SEASON_OF_MONTH: dict[int, str] = {}


# ── calendar ─────────────────────────────────────────────────────────────────
def dst_days(year: int) -> list[pd.Timestamp]:
    """The two civil days that do not have 96 quarter-hours: last Sunday of
    March (92) and last Sunday of October (100)."""
    out = []
    for month in (3, 10):
        d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        while d.dayofweek != 6:                       # 6 = Sunday
            d -= pd.Timedelta(days=1)
        out.append(d)
    return out


def italian_holidays(year: int) -> set[pd.Timestamp]:
    """Fixed national holidays plus Easter Monday."""
    fixed = [(1, 1), (1, 6), (4, 25), (5, 1), (6, 2),
             (8, 15), (11, 1), (12, 8), (12, 25), (12, 26)]
    days = {pd.Timestamp(year=year, month=m, day=d) for m, d in fixed}
    days.add(_easter_monday(year))
    return days


def _easter_monday(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm, then +1 day."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day) + pd.Timedelta(days=1)


def day_type(dates: pd.Series, holidays: set[pd.Timestamp]) -> pd.Series:
    """weekday / saturday / sunday, holidays counted as Sunday."""
    dow = dates.dt.dayofweek
    out = pd.Series("weekday", index=dates.index, dtype=object)
    out[dow == 5] = "saturday"
    out[dow == 6] = "sunday"
    out[dates.isin(holidays)] = "sunday"
    return out


def season(dates: pd.Series, mapping: dict[int, str]) -> pd.Series:
    return dates.dt.month.map(mapping)


# ── the six operations ───────────────────────────────────────────────────────
def unify_missing(arr: np.ndarray) -> np.ndarray:
    """Every encoding of 'missing' becomes NaN. Empty strings and NULLs are
    already NaN after the numeric cast; IEEE-754 NaN literals survive a null
    test and are caught here, and negatives cannot occur because the dataset
    records withdrawals from the network."""
    arr = arr.astype("float32", copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def classify_zero_runs(days: pd.DataFrame, min_run_days: int,
                       min_recurrences: int) -> pd.Series:
    """True where a zero day is genuine (a closure), False where it is a fault.

    A day is at zero when it draws nothing at all. Isolated zero days are left
    to the gap rule. A run of consecutive zero days is genuine when it is
    calendrically coherent, which here means either that it is made of
    non-working days alone, or that it lasts long enough to be a closure, or
    that the same POD exhibits several runs of the same length.
    """
    genuine = pd.Series(False, index=days.index)
    z = days[days["is_zero"]]
    for pod, grp in z.groupby("pod", sort=False):
        g = grp.sort_values("date")
        #Lorenzo Giannuzzo: consecutive runs
        breaks = (g["date"].diff() != pd.Timedelta(days=1)).cumsum()
        lengths = g.groupby(breaks)["date"].transform("size")
        run_id = breaks

        long_enough = lengths >= min_run_days
        non_working = g.groupby(run_id)["daytype"].transform(
            lambda s: (s != "weekday").all())
        counts = lengths.groupby(lengths).transform("size") / lengths
        recurrent = counts >= min_recurrences

        genuine.loc[g.index] = (long_enough | non_working | recurrent).values
    return genuine


def fill_gaps(shapes: np.ndarray, days: pd.DataFrame,
              short_max: int, medium_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill by gap length; return (filled, drop_mask).

    <= short_max quarter-hours : linear interpolation
    <= medium_max              : self-similarity, the median of the same
                                 quarter-hour on the same day type and season
                                 of the same POD
    >  medium_max              : the day is dropped
    """
    filled = shapes.copy()
    n = len(filled)
    drop = np.zeros(n, dtype=bool)

    nan_mask = np.isnan(filled)
    gap_len = nan_mask.sum(axis=1)

    #Lorenzo Giannuzzo: rows with no gap at all
    clean = gap_len == 0

    #Lorenzo Giannuzzo: rows to drop outright
    drop |= gap_len > medium_max

    # ── short gaps: linear interpolation along the row ───────────────────────
    idx_short = np.where((gap_len > 0) & (gap_len <= short_max) & ~drop)[0]
    x = np.arange(96)
    for i in idx_short:
        row = filled[i]
        m = np.isnan(row)
        if m.all():
            drop[i] = True
            continue
        row[m] = np.interp(x[m], x[~m], row[~m])
        filled[i] = row

    # ── medium gaps: self-similarity within (pod, daytype, season) ───────────
    idx_med = np.where((gap_len > short_max) & (gap_len <= medium_max) & ~drop)[0]
    if len(idx_med):
        key = days["pod"].astype(str) + "|" + days["daytype"] + "|" + days["season"]
        key = key.values
        #Lorenzo Giannuzzo: median profile per key, computed on the rows that are complete
        med: dict[str, np.ndarray] = {}
        for k in np.unique(key[idx_med]):
            pool = filled[(key == k) & clean]
            if len(pool):
                med[k] = np.nanmedian(pool, axis=0)
        for i in idx_med:
            ref = med.get(key[i])
            if ref is None:
                drop[i] = True
                continue
            row = filled[i]
            m = np.isnan(row)
            row[m] = ref[m]
            if np.isnan(row).any():               # the reference itself had holes
                drop[i] = True
            filled[i] = row

    return filled, drop


# ── pipeline ─────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config()
    year = cfg.get("data.year")            # None -> every month available
    pp = cfg["preprocessing"]
    out = cfg.results_dir("preprocessing")
    funnel: list[dict] = []

    def track(step: str, pods: int, days: int, note: str = "") -> None:
        funnel.append({"step": step, "pods": pods, "user_days": days, "note": note})
        print(f"  {step:38s} {pods:>7,} PODs {days:>10,} days   {note}")

    scope = f"year {year}" if year else "every month available"
    print(f"\n{'='*78}\nSTAGE 1 — PRE-PROCESSING (Section 2.2), {scope}\n{'='*78}\n")

    # ── load ─────────────────────────────────────────────────────────────────
    window = cfg.get("data.window")
    months = [(int(y), int(m)) for y, m in window] if window else None
    if months:
        span = f"{months[0][1]:02d}/{months[0][0]} to {months[-1][1]:02d}/{months[-1][0]}"
        print(f"Reading monthly folders (window, {len(months)} months: {span}):")
    else:
        print("Reading every monthly folder found:")
    meas, meta = load_year(cfg.data_dir, year,
                           cfg.get("data.meas_cols.date", "DataMisura"),
                           cfg.get("data.date_format"),
                           cfg.get("data.meas_cols.kind", "Tipologia"),
                           cfg.get("data.keep_kind", "AP"),
                           months=months)
    print()
    kinds = meas.attrs.get("kind_counts", {})
    if kinds:
        kept = cfg.get("data.keep_kind", "AP")
        others = ", ".join(f"{k}={v:,}" for k, v in sorted(kinds.items()) if k != kept)
        print(f"  rows by quantity: {kept}={kinds.get(kept, 0):,} kept   |   dropped: {others}")
    prosumers = meas.attrs.get("prosumer_pods", set())
    if prosumers:
        print(f"  PODs that also inject (AN rows present): {len(prosumers):,}\n")
    track("0. as read", meas["pod"].nunique(), len(meas), f"{cfg.get('data.keep_kind','AP')} only")

    #Lorenzo Giannuzzo: season map: month -> label
    global SEASON_OF_MONTH
    SEASON_OF_MONTH = {m: lab for lab, months in pp["seasons"].items() for m in months}

    qs = _q_columns(meas)
    shapes = unify_missing(meas[qs].to_numpy())

    # ── unit conversion, once, here ──────────────────────────────────────────
    # The export is in Wh; every threshold, energy and equation downstream is in
    # kWh. Converting at the door keeps the rest of the pipeline in one unit.
    runit = str(cfg.get("data.reading_unit", "kWh")).lower()
    if runit == "wh":
        shapes /= 1000.0
        print(f"\n  readings converted from Wh to kWh")
    elif runit != "kwh":
        raise ValueError(f"data.reading_unit must be 'Wh' or 'kWh', got {runit!r}")

    days = pd.DataFrame({
        "pod": meas["pod"].values,
        "date": meas["date"].values,
    })
    days = days.dropna(subset=["date"]).reset_index(drop=True)
    shapes = shapes[: len(days)]
    years_in_window = sorted(days["date"].dt.year.unique().tolist())
    hol = set().union(*(italian_holidays(y) for y in years_in_window))
    days["daytype"] = day_type(days["date"], hol)
    days["season"] = season(days["date"], SEASON_OF_MONTH)

    # ── 1. DST days ──────────────────────────────────────────────────────────
    dst = [d for y in years_in_window for d in dst_days(y)]
    keep = ~days["date"].isin(dst)
    days, shapes = days[keep].reset_index(drop=True), shapes[keep.values]
    track("1. DST days removed", days["pod"].nunique(), len(days),
          f"{[str(d.date()) for d in dst]}")

    # ── 2. outliers on contractual power ─────────────────────────────────────
    pcol = cfg.get("data.meas_cols.power")
    power_col = None
    if pcol:
        low = {c.lower(): c for c in meas.columns}
        power_col = low.get(str(pcol).lower())
    if power_col is not None:
        pmax = to_number(meas.loc[keep.values, power_col]).to_numpy(dtype="float32")
        if str(cfg.get("data.power_unit", "kW")).lower() == "w":
            pmax = pmax / 1000.0
        #Lorenzo Giannuzzo: A POD with no declared power carries no threshold; leaving it in would
        # censor every one of its readings against a limit of zero.
        floor = float(pp.get("min_contractual_power", 0.1))
        usable = np.isfinite(pmax) & (pmax > floor)
        #Lorenzo Giannuzzo: A quarter-hour of x kWh is a mean power of 4x kW over that quarter.
        limit = np.where(usable, pmax * float(pp["power_margin"]) / 4.0, np.inf)
        bad = shapes > limit[:, None]
        n_bad = int(np.nansum(bad))
        n_read = int(np.isfinite(shapes).sum())
        shapes[bad] = np.nan
        track("2. readings above contractual power", days["pod"].nunique(), len(days),
              f"{n_bad:,} censored ({100*n_bad/max(n_read,1):.3f}% of readings), "
              f"{int((~usable).sum()):,} rows without a usable threshold")
    else:
        track("2. outliers", days["pod"].nunique(), len(days),
              "SKIPPED: no contractual power column")

    # ── 3. zero runs ─────────────────────────────────────────────────────────
    daily_energy = np.nansum(shapes, axis=1)
    days["is_zero"] = (daily_energy == 0) & ~np.isnan(shapes).all(axis=1)
    genuine = classify_zero_runs(
        days, max(2, int(pp["zero_run_hours"] // 24)),
        int(pp["zero_run_min_recurrences"]))
    faulty_zero = days["is_zero"] & ~genuine
    shapes[faulty_zero.values] = np.nan
    track("3. zero runs classified", days["pod"].nunique(), len(days),
          f"{int(genuine.sum()):,} genuine, {int(faulty_zero.sum()):,} faults")

    # ── 4. gaps ──────────────────────────────────────────────────────────────
    shapes, drop = fill_gaps(shapes, days,
                             int(pp["gap_short_max"]), int(pp["gap_medium_max"]))
    days, shapes = days[~drop].reset_index(drop=True), shapes[~drop]
    track("4. long gaps: days dropped", days["pod"].nunique(), len(days),
          f"{int(drop.sum()):,} days")

    # ── 5. zero-energy days: valid observations, but they carry no shape ─────
    # A day at zero is not a missing day: the POD is known to have drawn nothing.
    # Eq. 2 normalises on the daily energy, so 0/0 leaves it without a shape and
    # it cannot enter the dictionary. It stays a valid day for the completeness
    # threshold of the next step, and its frequency becomes a feature: dropping
    # these days outright would cost every seasonally closed POD its eligibility.
    daily_energy = shapes.sum(axis=1)
    n_nz = (shapes > 0).sum(axis=1)
    min_nz = int(pp.get("min_nonzero_quarters", 1))
    days["energy"] = daily_energy
    days["has_shape"] = (daily_energy > 0) & (n_nz >= min_nz)
    n_zero = int((daily_energy <= 0).sum())
    n_thin = int(((daily_energy > 0) & (n_nz < min_nz)).sum())
    track("5. days without a shape, kept as days", days["pod"].nunique(), len(days),
          f"{n_zero:,} at zero + {n_thin:,} under {min_nz} non-zero quarters "
          f"= {100*(n_zero+n_thin)/max(len(days),1):.1f}%")

    # ── 6. completeness ──────────────────────────────────────────────────────
    per_pod = days.groupby("pod").agg(n_days=("date", "size"),
                                      n_seasons=("season", "nunique"))
    n_seasons_expected = len(pp["seasons"])
    calendar_days = days["date"].nunique()
    min_days = int(pp["min_valid_days"])
    if min_days > calendar_days:
        raise ValueError(
            f"min_valid_days={min_days} exceeds the {calendar_days} calendar days "
            f"present. No POD can pass. Lower it in config.yaml.")
    ok = per_pod["n_days"] >= min_days
    if pp["require_all_seasons"]:
        ok &= per_pod["n_seasons"] == n_seasons_expected
    good_pods = set(per_pod[ok].index)
    m = days["pod"].isin(good_pods)
    days, shapes = days[m].reset_index(drop=True), shapes[m.values]
    track("6. completeness thresholds", days["pod"].nunique(), len(days),
          f">= {min_days} of {calendar_days} days, all {n_seasons_expected} seasons")

    # ── Eq. 1: annualised energy ─────────────────────────────────────────────
    # E is the scale feature and the quantity normalised to 1000 kWh in Stage 3.
    # It is inflated to a full year so that PODs with different completeness are
    # comparable; it is therefore NOT the denominator of the day weights.
    # The period may span more than a year, and each POD covers a different slice
    # of it, so the annual energy is the observed mean daily energy scaled to 365
    # days. Over a single year this is exactly Eq. 1; over a longer period it is
    # the same quantity, an annual rate rather than a total.
    agg = days.groupby("pod")["energy"].agg(["sum", "size"])
    E = (agg["sum"] / agg["size"] * 365.0).rename("E")
    #Lorenzo Giannuzzo: Fraction of the observed year spent at zero. The frequencies of Eq. 4 are
    # weighted by energy and are therefore blind to it: a shop closed for three
    # months and one open all year would share the same frequency vector.
    zero_frac = (1.0 - days.groupby("pod")["has_shape"].mean()).rename("zero_day_fraction")

    # ── Eq. 2: factorization ─────────────────────────────────────────────────
    # The weights are the composition of the observed year and sum to one by
    # construction. Dividing by the annualised E instead would make them sum to
    # |D_i|/365, and the two conditions of Eq. 2 could not both hold.
    observed = days.groupby("pod")["energy"].sum()
    days["w"] = days["energy"] / days["pod"].map(observed).values
    has = days["has_shape"].to_numpy()

    norm = str(pp.get("shape_normalisation", "unit_integral")).lower()
    raw = shapes[has]
    if norm == "unit_integral":
        #Lorenzo Giannuzzo: Eq. 2: every shape is a distribution over the day and sums to one.
        shapes = (raw / days.loc[has, "energy"].to_numpy()[:, None]).astype("float32")
    elif norm == "min_max":
        #Lorenzo Giannuzzo: The outline alone: the peak is 1 whatever it is, and the energy under
        # the curve is discarded. A day already flat has its noise stretched to
        # full scale, which is the risk this normalisation carries here.
        lo = raw.min(axis=1, keepdims=True)
        hi = raw.max(axis=1, keepdims=True)
        rng_ = np.where(hi - lo > 0, hi - lo, 1.0)
        shapes = ((raw - lo) / rng_).astype("float32")
    else:
        raise ValueError(f"shape_normalisation must be unit_integral | min_max, "
                         f"got {norm!r}")
    print(f"  shapes normalised: {norm}")
    #Lorenzo Giannuzzo: shapes.npy holds only the days that have one; days.parquet holds them all,
    # and days[days.has_shape] indexes into shapes row by row.
    days["shape_idx"] = -1
    days.loc[has, "shape_idx"] = np.arange(int(has.sum()))

    # ── metadata ─────────────────────────────────────────────────────────────
    ateco_col = cfg.get("data.meta_cols.ateco", "CCATETE")
    low = {c.lower(): c for c in meta.columns}
    acol = low.get(str(ateco_col).lower())
    lv = meta[acol].apply(split_ateco) if acol else pd.Series([(None, None, None)] * len(meta))
    meta["ateco_l1"] = [x[0] for x in lv]
    meta["ateco_l2"] = [x[1] for x in lv]
    meta["ateco_l3"] = [x[2] for x in lv]

    users = pd.DataFrame({"pod": E.index, "E": E.values}).merge(
        zero_frac.reset_index(), on="pod", how="left").merge(
        meta.drop_duplicates("pod"), on="pod", how="left")
    #Lorenzo Giannuzzo: A prosumer is a POD that also records injected active energy. The readings
    # kept here are withdrawals, so it cannot be told from their sign.
    users["prosumer"] = users["pod"].isin(prosumers)

    # ── write ────────────────────────────────────────────────────────────────
    np.save(cfg.cache_dir / "shapes.npy", shapes)
    days.drop(columns=["is_zero"], errors="ignore").to_parquet(
        cfg.cache_dir / "days.parquet", index=False)
    users.to_parquet(cfg.cache_dir / "users.parquet", index=False)

    pd.DataFrame(funnel).to_csv(out / "funnel.csv", index=False)

    for lvl in ("ateco_l1", "ateco_l2", "ateco_l3"):
        if lvl in users:
            (users.groupby(lvl, dropna=False)
                  .agg(n_pods=("pod", "size"), energy_MWh=("E", lambda s: s.sum() / 1000))
                  .sort_values("n_pods", ascending=False)
                  .to_csv(out / f"composition_by_{lvl}.csv"))

    tcol = low.get(str(cfg.get("data.meta_cols.tariff_desc", "D_49DES")).lower())
    if tcol and tcol in users:
        (users.groupby(tcol, dropna=False)
              .agg(n_pods=("pod", "size"))
              .sort_values("n_pods", ascending=False)
              .to_csv(out / "composition_by_tariff.csv"))

    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        d0, d1 = days["date"].min(), days["date"].max()
        f.write(f"Period [YYYY]                {d0.date()} to {d1.date()}\n")
        f.write(f"Calendar days spanned        {days['date'].nunique():,}\n")
        f.write(f"PODs retained [N]            {len(users):,}\n")
        f.write(f"User-days retained           {len(days):,}\n")
        f.write(f"Daily shapes for [N x 365]   {len(shapes):,}\n")
        f.write(f"Days at zero, no shape       {n_zero:,} "
                f"({100*n_zero/max(len(days),1):.1f}% of valid days)\n")
        f.write(f"Days under {min_nz} non-zero QH      {n_thin:,} "
                f"({100*n_thin/max(len(days),1):.1f}%), no shape either\n")
        f.write(f"PODs that also inject        {int(users['prosumer'].sum()):,}\n")
        f.write(f"Shape normalisation          {norm}\n")
        f.write(f"Mean valid days per POD      {len(days)/max(len(users),1):.1f}\n")
        f.write(f"Annual energy, total GWh     {users['E'].sum()/1e6:.1f}\n")
        for lvl in ("ateco_l1", "ateco_l2", "ateco_l3"):
            if lvl in users:
                f.write(f"Distinct {lvl}              {users[lvl].nunique()}\n")

        months = sorted(pd.to_datetime(days["date"]).dt.month.unique())
        absent = sorted(set(range(1, 13)) - set(months))
        f.write(f"\nMonths present               {[int(m) for m in months]}\n")
        if absent:
            f.write(f"Months absent                {absent}\n")
            f.write(
                "\nNote on Eq. 1. The annualisation scales the observed energy to a\n"
                "full year assuming the absent days resemble the observed ones. Where\n"
                "whole months are missing this assumption is false, and the annual\n"
                "energy is biased by whatever the absent months would have carried.\n"
                "The bias is common to every POD, since the same months are absent\n"
                "for all of them, so it displaces the level of E without disturbing\n"
                "the ordering of the users, which is what the clustering reads. It\n"
                "does bear on any absolute figure in kWh and is declared as such.\n")

    print(f"\n{'='*78}")
    print(f"  [N]    = {len(users):,} PODs")
    print(f"  period = {days['date'].min().date()} to {days['date'].max().date()} "
          f"({days['date'].nunique():,} calendar days)")
    print(f"  daily shapes = {len(shapes):,}")
    print(f"\n  cache/   shapes.npy, days.parquet, users.parquet")
    print(f"  results/ {out.relative_to(out.parent.parent)}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
