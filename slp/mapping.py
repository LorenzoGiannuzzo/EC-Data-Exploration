"""Mapping stage, Section 2.6.

The activity classes enter here and nowhere earlier. Everything upstream is built without
them, so what follows is a test of the classification rather than a restatement of it.

A single contingency table n(k, c) counts points of delivery, and their energy, by
data-driven profile k and activity class c. It is read in both directions:

  M1  class multiplicity, on the row. How many profiles a single activity class spans.
      A class that maps to one profile is predicted by its code; a class spread over
      several is not.

  M2  profile aggregation, on the column. How many activity classes a single profile
      subsumes. A profile that gathers many classes is describing a behaviour the
      classification does not name.

  M3  real coverage, on the national profiles. How many activity classes each published
      profile is actually applied to, which is the quantity the regulation implicitly
      claims to be one.

Counting the raw number of non-empty cells would make every figure grow with sample size
and reward noise. The counts reported are effective numbers, the exponential of the
Shannon entropy of the row or column, which answers "how many, in effect" and is
insensitive to a long tail of single points. A hard count covering eighty per cent of the
mass is reported beside it because it reads more easily in a table. Both come with
bootstrap intervals, without which a class of thirty points would produce a number that
looks like a result and is not.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import assignment  # noqa: E402
from common.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
_CFG = load_config()
CACHE = _CFG.cache_dir

#Lorenzo Giannuzzo: the contingency tables stay at the root of the stage because all three
#metrics descend from them and none of the three owns them. Each metric then writes into
#its own folder, which is the folder a reader following its section of the paper opens.
OUT = _CFG.results_dir("mapping")
OUT_M1 = _CFG.results_dir("mapping", "multiplicity")
OUT_M2 = _CFG.results_dir("mapping", "aggregation")
OUT_M3 = _CFG.results_dir("mapping", "coverage")

ACTIVITY_LEVEL = "ateco_l1"     # ateco_l1 | ateco_l2 | ateco_l3
N_MIN = 25                      # a class below this is pooled into the residual
N_BOOTSTRAP = 500
# Public lighting and vehicle charging are a handful of points on codes that cannot be
# decoded with confidence: they are left out of every figure rather than shown as noise.
GSE_KEEP = ("PDMM", "PAUM")
RNG = np.random.default_rng(20260728)


# ------------------------------------------------------------------- effective numbers
def effective_number(counts: np.ndarray) -> float:
    """exp of the Shannon entropy: how many categories the mass is spread over.

    Uniform over four categories returns four. Ninety-five per cent on one and the rest
    scattered returns close to one, which is the honest answer and is what a raw count of
    non-empty cells fails to give.
    """
    c = np.asarray(counts, float)
    c = c[c > 0]
    if c.sum() <= 0:
        return np.nan
    p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def hard_count(counts: np.ndarray, cover: float = 0.80) -> int:
    """Smallest number of categories holding `cover` of the mass."""
    c = np.sort(np.asarray(counts, float))[::-1]
    if c.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(c) / c.sum(), cover) + 1)


def bootstrap_effective(labels: np.ndarray, weights: np.ndarray | None = None,
                        n: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Percentile interval for the effective number, resampling the points themselves."""
    if len(labels) == 0:
        return (np.nan, np.nan)
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, float)
    cats = np.unique(labels)
    out = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, len(labels), len(labels))
        counts = np.array([w[idx][labels[idx] == c].sum() for c in cats])
        out[i] = effective_number(counts)
    return (float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)))


# ---------------------------------------------------------------------- the table
def build_table(users: pd.DataFrame, groups: pd.DataFrame,
                level: str = ACTIVITY_LEVEL) -> pd.DataFrame:
    """One row per point, carrying its profile, its activity class and its energy."""
    cols = ["pod", level, "E", "D_TIPTA"]
    u = users[cols].rename(columns={level: "activity"})
    m = groups.merge(u, on="pod", how="inner")
    m = m[m["activity"].notna()].copy()
    m["profile"] = "DDSLP_" + m["group"].astype(int).astype(str)
    small = m["activity"].value_counts()
    rare = set(small[small < N_MIN].index)
    m["activity_grouped"] = np.where(m["activity"].isin(rare), "other (below n_min)",
                                     m["activity"])
    return m


def multiplicity(m: pd.DataFrame, weight: str = "pod") -> pd.DataFrame:
    """M1, read on the row: how many profiles each activity class spans."""
    rows = []
    for cls, x in m.groupby("activity_grouped"):
        w = None if weight == "pod" else x["E"].to_numpy()
        counts = (x.groupby("profile").size() if weight == "pod"
                  else x.groupby("profile")["E"].sum()).to_numpy()
        lo, hi = bootstrap_effective(x["profile"].to_numpy(), w)
        n_max = m["profile"].nunique()
        rows.append({"activity": cls, "weight": weight, "n_pod": len(x),
                     "energy_kWh": float(x["E"].sum()),
                     "M1_effective": effective_number(counts),
                     # An effective number cannot exceed the number of categories
                     # available, so it grows with K by construction and two runs at
                     # different K are not comparable until it is divided by that maximum.
                     # The ratio is the share of the maximum possible spread, and it is
                     # the quantity to report when the configuration changes.
                     "M1_evenness": effective_number(counts) / n_max,
                     "M1_ci_low": lo, "M1_ci_high": hi,
                     "M1_hard80": hard_count(counts),
                     "dominant_share": float(counts.max() / counts.sum())})
    return pd.DataFrame(rows).sort_values("M1_effective", ascending=False)


def aggregation(m: pd.DataFrame, weight: str = "pod") -> pd.DataFrame:
    """M2, read on the column: how many activity classes each profile subsumes."""
    rows = []
    for prof, x in m.groupby("profile"):
        w = None if weight == "pod" else x["E"].to_numpy()
        counts = (x.groupby("activity_grouped").size() if weight == "pod"
                  else x.groupby("activity_grouped")["E"].sum()).to_numpy()
        lo, hi = bootstrap_effective(x["activity_grouped"].to_numpy(), w)
        n_max = m["activity_grouped"].nunique()
        rows.append({"profile": prof, "weight": weight, "n_pod": len(x),
                     "energy_kWh": float(x["E"].sum()),
                     "M2_effective": effective_number(counts),
                     "M2_evenness": effective_number(counts) / n_max,
                     "M2_ci_low": lo, "M2_ci_high": hi,
                     "M2_hard80": hard_count(counts),
                     "n_classes_present": int((counts > 0).sum()),
                     "dominant_share": float(counts.max() / counts.sum())})
    return pd.DataFrame(rows).sort_values("M2_effective", ascending=False)


def real_coverage(m: pd.DataFrame, users: pd.DataFrame,
                  weight: str = "pod") -> pd.DataFrame:
    """M3: how many activity classes each *national* profile is actually applied to."""
    g = assignment.gse_profile(users)[["pod", "gse_column"]]
    g = g[g["gse_column"].isin(GSE_KEEP)].rename(columns={"gse_column": "national"})
    a = assignment.arera_key(users)
    a = a[a["arera_applicable"]].copy()
    a["national"] = ("ARERA " + a["arera_class"].astype(str) + " "
                     + a["arera_residency"].astype(str))
    both = pd.concat([g[["pod", "national"]], a[["pod", "national"]]], ignore_index=True)
    x = m.merge(both, on="pod", how="inner")
    n_cls_max = m["activity_grouped"].nunique()
    n_prof_max = m["profile"].nunique()
    rows = []
    for prof, y in x.groupby("national"):
        w = None if weight == "pod" else y["E"].to_numpy()
        counts = (y.groupby("activity_grouped").size() if weight == "pod"
                  else y.groupby("activity_grouped")["E"].sum()).to_numpy()
        lo, hi = bootstrap_effective(y["activity_grouped"].to_numpy(), w)
        ddslp = (y.groupby("profile").size() if weight == "pod"
                 else y.groupby("profile")["E"].sum()).to_numpy()
        lo_d, hi_d = bootstrap_effective(y["profile"].to_numpy(), w)
        rows.append({"national_profile": prof, "weight": weight, "n_pod": len(y),
                     # what the regulation declares: one profile stands for the whole
                     # tariff category, which is the claim being tested
                     "declared": 1.0,
                     "M3_classes_effective": effective_number(counts),
                     "M3_classes_evenness": effective_number(counts) / n_cls_max,
                     "M3_ci_low": lo, "M3_ci_high": hi,
                     "M3_classes_hard80": hard_count(counts),
                     "n_classes_present": int((counts > 0).sum()),
                     # and how many distinct consumption behaviours are actually hiding
                     # under that single published curve
                     "M3_ddslp_effective": effective_number(ddslp),
                     "M3_ddslp_evenness": effective_number(ddslp) / n_prof_max,
                     "M3_ddslp_ci_low": lo_d, "M3_ddslp_ci_high": hi_d,
                     "n_ddslp_present": int((ddslp > 0).sum())})
    return pd.DataFrame(rows).sort_values("M3_classes_effective", ascending=False)


# ------------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    print(f"\n{'='*78}\n  MAPPING, Section 2.6\n{'='*78}")

    users = pd.read_parquet(CACHE / "users.parquet")
    groups = pd.read_parquet(CACHE / "groups.parquet")
    m = build_table(users, groups)
    print(f"  {len(m)} points, {m['activity_grouped'].nunique()} classes at "
          f"{ACTIVITY_LEVEL} (n_min = {N_MIN})")

    ct = pd.crosstab(m["activity_grouped"], m["profile"])
    ct.to_csv(OUT / "contingency_pod.csv")
    pd.crosstab(m["activity_grouped"], m["profile"],
                values=m["E"], aggfunc="sum").to_csv(OUT / "contingency_energy.csv")

    for weight in ("pod", "energy"):
        multiplicity(m, weight).to_csv(OUT_M1 / f"m1_multiplicity_{weight}.csv",
                                       index=False)
        aggregation(m, weight).to_csv(OUT_M2 / f"m2_aggregation_{weight}.csv",
                                      index=False)
        real_coverage(m, users, weight).to_csv(OUT_M3 / f"m3_coverage_{weight}.csv",
                                               index=False)

    m1 = pd.read_csv(OUT_M1 / "m1_multiplicity_pod.csv")
    m2 = pd.read_csv(OUT_M2 / "m2_aggregation_pod.csv")
    K = m["profile"].nunique()
    C = m["activity_grouped"].nunique()
    print(f"\n  K = {K} profiles, C = {C} activity classes")
    print(f"  M1, activity class -> profiles   median {m1['M1_effective'].median():.2f}"
          f" of {K}   ({m1['M1_evenness'].median()*100:.0f}% of the maximum spread)")
    print(f"  M2, profile -> activity classes  median {m2['M2_effective'].median():.2f}"
          f" of {C}   ({m2['M2_evenness'].median()*100:.0f}% of the maximum spread)")

    #Lorenzo Giannuzzo: the figures are not drawn here. They were being drawn three times
    #a run, once by this stage, once by the figures stage and once more on the way past,
    #and a failure in one of them was reported three times over.
    print(f"\n  results under {OUT}   ({time.time()-t0:.0f}s)\n")


if __name__ == "__main__":
    main()