"""Stage 1 — Pre-processing.

Same filtering the SLP pipeline performs, at the same thresholds, with one
deliberate difference: the daily curve is not factorised into energy, day weight
and shape. Here the level in kilowatt-hours per quarter-hour is the primary
quantity and it stays attached to the curve, because the chain estimated later
walks on levels and separating them would reintroduce exactly the split this
project rules out.

Order of operations, which matters:
  1. daylight saving days removed, they cannot be laid on a 96 slot grid
  2. readings above contractual power censored to missing
  3. zero runs classified, genuine closures kept, faults censored
  4. gaps filled by length, long ones drop the day
  5. days marked valid or not

Outputs
    cache/synthgen/curves.npy        (n_days, 96) float32, kWh per quarter-hour
    cache/synthgen/days.parquet      pod, date, daytype, season, month, energy, valid
    cache/synthgen/users.parquet     pod, ateco levels, power, tariff, prosumer
    results/preprocessing/funnel.csv what each step removed

Run
    python -m synthgen.preprocessing
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import annotate, dst_days
from .config import load_config
from .io import load_archive
from .taxonomy import split_ateco, is_domestic


def classify_zero_runs(days: pd.DataFrame, min_run_days: int,
                       min_recurrences: int) -> pd.Series:
    """True where a zero day is a genuine closure, False where it is a fault.

    A run of consecutive zero days is genuine when it is calendrically coherent,
    which means that it is made of non-working days alone, or that it lasts long
    enough to be a closure, or that the same POD shows several runs of the same
    length.
    """
    genuine = pd.Series(False, index=days.index)
    z = days[days["is_zero"]]
    for _, grp in z.groupby("pod", sort=False):
        g = grp.sort_values("date")
        breaks = (g["date"].diff() != pd.Timedelta(days=1)).cumsum()
        lengths = g.groupby(breaks)["date"].transform("size")
        long_enough = lengths >= min_run_days
        non_working = g.groupby(breaks)["daytype"].transform(
            lambda s: (s != "weekday").all())
        counts = lengths.groupby(lengths).transform("size") / lengths
        genuine.loc[g.index] = (long_enough | non_working
                                | (counts >= min_recurrences)).values
    return genuine


def fill_gaps(curves: np.ndarray, days: pd.DataFrame,
              short_max: int, medium_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill by gap length and return (filled, drop_mask).

    Up to short_max quarter-hours the row is interpolated linearly. Up to
    medium_max it is completed with the median of the same quarter-hour on the
    same day type and season of the same POD. Beyond that the day is dropped,
    since a quarter of the day reconstructed is no longer an observation.
    """
    filled = curves          #Lorenzo Giannuzzo: in place: the caller does not need the original
    n = len(filled)
    drop = np.zeros(n, dtype=bool)
    nan_mask = np.isnan(filled)
    gap_len = nan_mask.sum(axis=1)
    clean = gap_len == 0
    drop |= gap_len > medium_max

    x = np.arange(filled.shape[1])
    for i in np.where((gap_len > 0) & (gap_len <= short_max) & ~drop)[0]:
        row = filled[i]
        m = np.isnan(row)
        if m.all():
            drop[i] = True
            continue
        row[m] = np.interp(x[m], x[~m], row[~m])
        filled[i] = row

    idx_med = np.where((gap_len > short_max) & (gap_len <= medium_max) & ~drop)[0]
    if len(idx_med):
        key = (days["pod"].astype(str) + "|" + days["daytype"].astype(str)
               + "|" + days["season"].astype(str)).to_numpy()
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
            if np.isnan(row).any():
                drop[i] = True
            filled[i] = row
    return filled, drop


def main(folders: list[str] | None = None) -> None:
    cfg = load_config()
    pp = cfg["preprocessing"]
    out = cfg.results_dir("preprocessing")
    funnel: list[dict] = []

    def track(step: str, pods: int, n_days: int, note: str = "") -> None:
        funnel.append({"step": step, "pods": pods, "pod_days": n_days, "note": note})
        print(f"  {step:<40s} {pods:>7,} PODs {n_days:>10,} days   {note}")

    print(f"\n{'=' * 78}\nSTAGE 1 — PRE-PROCESSING\n{'=' * 78}\n")
    print("Reading the archive:")
    days, curves, meta = load_archive(cfg.data_dir, cfg["data"], folders)
    print()
    track("0. as read", days["pod"].nunique(), len(days),
          f"{days.attrs['n_folders']} folders, "
          f"{days.attrs['n_duplicates_resolved']} duplicate POD-days resolved")

    days = annotate(days, pp["seasons"])
    curves[~np.isfinite(curves)] = np.nan

    #Lorenzo Giannuzzo: ── 1. daylight saving days ──────────────────────────────────────────────
    years = sorted(days["date"].dt.year.unique().tolist())
    dst = [d for y in years for d in dst_days(int(y))]
    keep = ~days["date"].isin(dst)
    days, curves = days[keep].reset_index(drop=True), curves[keep.to_numpy()]
    track("1. daylight saving days removed", days["pod"].nunique(), len(days),
          ", ".join(str(d.date()) for d in dst))

    #Lorenzo Giannuzzo: ── 2. readings above contractual power ──────────────────────────────────
    # A quarter-hour of x kWh is a mean power of 4x kW over that quarter, so the
    # comparable limit is the contractual power divided by four, with a margin.
    floor = float(pp.get("min_contractual_power", 0.1))
    pmax = days["power_kW"].to_numpy(dtype="float32")
    usable = np.isfinite(pmax) & (pmax > floor)
    limit = np.where(usable, pmax * float(pp["power_margin"]) / 4.0, np.inf)
    bad = curves > limit[:, None]
    n_bad, n_read = int(np.nansum(bad)), int(np.isfinite(curves).sum())
    curves[bad] = np.nan
    track("2. readings above contractual power", days["pod"].nunique(), len(days),
          f"{n_bad:,} censored ({100 * n_bad / max(n_read, 1):.3f}%), "
          f"{int((~usable).sum()):,} rows without a threshold")

    #Lorenzo Giannuzzo: ── 3. zero runs ─────────────────────────────────────────────────────────
    daily = np.nansum(curves, axis=1)
    days["is_zero"] = (daily == 0) & ~np.isnan(curves).all(axis=1)
    genuine = classify_zero_runs(days, max(2, int(pp["zero_run_hours"] // 24)),
                                 int(pp["zero_run_min_recurrences"]))
    faulty = days["is_zero"] & ~genuine
    curves[faulty.to_numpy()] = np.nan
    track("3. zero runs classified", days["pod"].nunique(), len(days),
          f"{int(genuine.sum()):,} genuine closures, {int(faulty.sum()):,} faults")

    #Lorenzo Giannuzzo: ── 4. gaps ──────────────────────────────────────────────────────────────
    curves, drop = fill_gaps(curves, days, int(pp["gap_short_max"]),
                             int(pp["gap_medium_max"]))
    days, curves = days[~drop].reset_index(drop=True), curves[~drop]
    track("4. long gaps: days dropped", days["pod"].nunique(), len(days),
          f"{int(drop.sum()):,} days")

    #Lorenzo Giannuzzo: ── 5. valid days ────────────────────────────────────────────────────────
    # A day at zero is an observation, not a hole: the point is known to have
    # drawn nothing, and a shop closed in the low season needs those days to keep
    # its eligibility. It is valid, and it is generable.
    days["energy"] = np.nansum(curves, axis=1)
    days["valid"] = ~np.isnan(curves).any(axis=1)
    days["is_zero_day"] = days["energy"] <= 0
    track("5. days marked", days["pod"].nunique(), int(days["valid"].sum()),
          f"{int((~days['valid']).sum()):,} incomplete, "
          f"{int(days['is_zero_day'].sum()):,} at zero")

    #Lorenzo Giannuzzo: ── users ────────────────────────────────────────────────────────────────
    lv = meta["ateco_raw"].apply(split_ateco)
    meta["ateco_l1"] = [x[0] for x in lv]
    meta["ateco_l2"] = [x[1] for x in lv]
    meta["ateco_l3"] = [x[2] for x in lv]
    meta["domestic"] = meta["ateco_raw"].apply(is_domestic)

    per_pod = days.groupby("pod").agg(
        power_kW=("power_kW", "max"),
        prosumer=("prosumer", "any"),
        n_days_read=("date", "size"),
        n_days_valid=("valid", "sum"))
    users = per_pod.reset_index().merge(meta, on="pod", how="left")
    #Lorenzo Giannuzzo: The contractual power is declared in both files and they do not always
    # agree, the measurement file being the contemporaneous one. The larger of
    # the two is kept, since the censoring threshold must not be too tight.
    users["power_kW"] = users[["power_kW", "power_meta_kW"]].max(axis=1)

    days = days.drop(columns=["is_zero"], errors="ignore")
    np.save(cfg.cache_dir / "curves.npy", curves.astype("float32"))
    days.to_parquet(cfg.cache_dir / "days.parquet", index=False)
    users.to_parquet(cfg.cache_dir / "users.parquet", index=False)
    pd.DataFrame(funnel).to_csv(out / "funnel.csv", index=False)

    print(f"\n  period      {days['date'].min().date()} to {days['date'].max().date()}"
          f"   ({days['date'].nunique():,} calendar days)")
    print(f"  PODs        {len(users):,}   of which domestic "
          f"{int(users['domestic'].fillna(False).sum()):,}")
    print(f"  valid days  {int(days['valid'].sum()):,}")
    print(f"\n  cache/synthgen/  curves.npy, days.parquet, users.parquet")
    print(f"  results/         preprocessing/funnel.csv\n")


if __name__ == "__main__":
    main()