"""Stage 2 — Eligibility.

Which points hold a year of data, and what each typology is left with once the
requirement is applied.

The requirement is stricter than the one the SLP pipeline uses, and deliberately
so. There, 250 valid days are enough because the profiles average over a coarse
grid. Here a month absent from the estimate is a month the generator cannot
produce, so the point has to cover the calendar: `min_valid_days` distinct valid
days, and every calendar month present with at least `min_days_per_month` of them.

Outputs
    results/eligibility/eligible_pods.csv     one row per point that passes
    results/eligibility/rejected_pods.csv     one row per point that does not, with the reason
    results/eligibility/census_level{1,2,3}.csv   points per typology, before and after

Run
    python -m synthgen.eligibility
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .taxonomy import census, level_column


def coverage(days: pd.DataFrame, min_days_per_month: int) -> pd.DataFrame:
    """Valid days per point, and how many calendar months it covers."""
    v = days[days["valid"]]
    per_pod = v.groupby("pod").agg(
        n_days_valid=("date", "nunique"),
        n_days_zero=("is_zero_day", "sum"),
        energy_kWh=("energy", "sum"),
        first_day=("date", "min"),
        last_day=("date", "max"))
    per_month = (v.groupby(["pod", "month"])["date"].nunique()
                  .rename("n").reset_index())
    covered = (per_month[per_month["n"] >= min_days_per_month]
               .groupby("pod")["month"].nunique().rename("months_covered"))
    per_pod = per_pod.join(covered).fillna({"months_covered": 0})
    per_pod["months_covered"] = per_pod["months_covered"].astype(int)
    wide = (per_month.pivot(index="pod", columns="month", values="n")
                     .reindex(columns=range(1, 13)).fillna(0).astype(int))
    wide.columns = [f"days_m{m:02d}" for m in wide.columns]
    return per_pod.join(wide)


def main() -> None:
    cfg = load_config()
    el = cfg["eligibility"]
    out = cfg.results_dir("eligibility")
    min_days = int(el["min_valid_days"])
    min_per_month = int(el["min_days_per_month"])

    days = pd.read_parquet(cfg.cache_dir / "days.parquet")
    users = pd.read_parquet(cfg.cache_dir / "users.parquet")
    print(f"\n{'=' * 78}\nSTAGE 2 — ELIGIBILITY\n{'=' * 78}\n")
    print(f"  requirement: >= {min_days} valid days"
          + (f", every month with >= {min_per_month} of them"
             if bool(el.get("require_all_months", True)) else ""))

    cov = coverage(days, min_per_month)
    #Lorenzo Giannuzzo: users.parquet already carries a day count from stage 1; the coverage table
    # is the authoritative one here, so the older columns step aside rather than
    # collide under a merge suffix.
    df = users.drop(columns=["n_days_read", "n_days_valid"],
                    errors="ignore").merge(cov, on="pod", how="left")
    for c in ("n_days_valid", "months_covered", "n_days_zero", "energy_kWh"):
        df[c] = df[c].fillna(0)

    enough_days = df["n_days_valid"] >= min_days
    all_months = (df["months_covered"] >= 12) if bool(
        el.get("require_all_months", True)) else True
    df["eligible"] = enough_days & all_months
    df["reason"] = np.where(
        df["eligible"], "",
        np.where(~enough_days, "too few valid days", "a calendar month is missing"))

    keep = ["pod", "eligible", "reason", "n_days_valid", "months_covered",
            "n_days_zero", "energy_kWh", "power_kW", "tariff", "tariff_desc",
            "domestic", "prosumer", "ateco_raw", "ateco_l1", "ateco_l2",
            "ateco_l3", "first_day", "last_day"]
    keep += [c for c in df.columns if c.startswith("days_m")]
    df = df[[c for c in keep if c in df.columns]]

    df[df["eligible"]].drop(columns=["eligible", "reason"]).to_csv(
        out / "eligible_pods.csv", index=False)
    df[~df["eligible"]].to_csv(out / "rejected_pods.csv", index=False)

    n_ok = int(df["eligible"].sum())
    print(f"  {n_ok:,} of {len(df):,} points pass "
          f"({100 * n_ok / max(len(df), 1):.1f}%)")
    if n_ok < len(df):
        print("  rejected for: "
              + ", ".join(f"{r} {int(n):,}" for r, n
                          in df.loc[~df["eligible"], "reason"].value_counts().items()))

    #Lorenzo Giannuzzo: ── what each level is left with ─────────────────────────────────────────
    ok = df[df["eligible"]]
    for level in (1, 2, 3):
        col = level_column(level)
        before = census(df, level)
        after = census(ok, level)
        tab = (before[["n_pods"]].rename(columns={"n_pods": "n_pods_all"})
               .join(after[["n_pods"]].rename(columns={"n_pods": "n_pods_eligible"}),
                     how="left").fillna({"n_pods_eligible": 0}))
        tab["n_pods_eligible"] = tab["n_pods_eligible"].astype(int)
        if "power_kW_median" in after:
            tab = tab.join(after[["power_kW_median"]], how="left")
        tab = tab.sort_values("n_pods_eligible", ascending=False)
        tab.to_csv(out / f"census_level{level}.csv")
        usable = tab["n_pods_eligible"]
        print(f"\n  level {level} ({col}): {int(df[col].notna().sum()):,} points carry "
              f"a code this deep, {len(tab)} typologies")
        print(f"    eligible typologies with >= 30 points {int((usable >= 30).sum())}, "
              f">= 10 {int((usable >= 10).sum())}, "
              f">= 5 {int((usable >= 5).sum())}, "
              f"with none {int((usable == 0).sum())}")
        head = tab[usable > 0].head(8)
        if len(head):
            print("    largest: " + ", ".join(
                f"{i} ({int(r.n_pods_eligible)})" for i, r in head.iterrows()))

    print(f"\n  results/eligibility/  eligible_pods.csv, rejected_pods.csv, "
          f"census_level1..3.csv\n")


if __name__ == "__main__":
    main()
