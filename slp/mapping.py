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

  M3  real coverage, on the national profiles, read on the users each published profile
      is applied to: how many data-driven profiles they distribute over (the behaviours
      the curve stands in for, Eq. 9 with the national profile in place of the class),
      how many activity classes they carry, and whom the profile would describe through
      those behaviours (Eq. 12), with the share falling outside its declared category.

Only the published catalogue enters. A group below n_min carries no profile in
generation.py and none in comparison.py, so it carries none here either: counting it
would compute the metrics on a catalogue of K profiles while the paper publishes fewer.

Counting the raw number of non-empty cells would make every figure grow with sample size
and reward noise. The counts reported are effective numbers, the exponential of the
Shannon entropy of the row or column, which answers "how many, in effect" and is
insensitive to a long tail of single points. A hard count covering eighty per cent of the
mass is reported beside it because it reads more easily in a table. Both come with
bootstrap intervals, without which a class of thirty points would produce a number that
looks like a result and is not.

Every effective number is also set against what an uninformative label would give. The
analytical value is exp(H(A)), the effective number of profiles of the whole population;
the reference actually used for a class of n points is the distribution of the effective
number over n points drawn at random from the population, which carries the downward bias
of the entropy on small samples that the analytical value does not.

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
SENSITIVITY_LEVEL = "ateco_l2"
RESIDUAL = "other (below n_min)"
DOMESTIC_PREFIX = ("DO",)
N_BOOTSTRAP = int(_CFG.get("mapping.n_bootstrap", 500))
N_PERMUTATIONS = int(_CFG.get("mapping.null_permutations", 1000))
K_OFFSETS = list(_CFG.get("mapping.sensitivity_k_offsets", [-2, -1, 1, 2]) or [])
#Lorenzo Giannuzzo: Public lighting and vehicle charging are a handful of points on codes that cannot be
# decoded with confidence: they are left out of every figure rather than shown as noise.
GSE_KEEP = ("PDMM", "PAUM")
#Lorenzo Giannuzzo: the category each published profile declares, on the activity dimension
# the mapping can read. Power class and residency are not activity classes, so the share
# outside the category computed on this dimension alone is a lower bound.
DECLARED_DOMESTIC = {"PDMM": True, "PAUM": False}
RNG = np.random.default_rng(20260728)


def n_min_classes(k_published: int) -> int:
    #Lorenzo Giannuzzo: the rule declared in config.yaml, max(min_class_size, multiple x K),
    # with K the number of published profiles the metrics are computed over
    base = int(_CFG.get("labels.min_class_size", 25))
    mult = int(_CFG.get("labels.min_class_multiple_of_k", 3))
    return max(base, mult * int(k_published))


def n_min_groups(K: int) -> int:
    #Lorenzo Giannuzzo: the same rule clustering.py applies to the groups
    cfg = _CFG.get("clustering.min_group_size")
    return int(cfg) if cfg not in (None, "null") else max(30, 3 * int(K))


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


def _codes(labels: np.ndarray) -> tuple[np.ndarray, int]:
    cats, inv = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return inv, len(cats)


def bootstrap_effective(labels: np.ndarray, weights: np.ndarray | None = None,
                        n: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Percentile interval for the effective number, resampling the points themselves."""
    if len(labels) == 0:
        return (np.nan, np.nan)
    inv, n_cat = _codes(labels)
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, float)
    out = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, len(labels), len(labels))
        out[i] = effective_number(np.bincount(inv[idx], weights=w[idx], minlength=n_cat))
    return (float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)))


def null_effective(pop_labels: np.ndarray, pop_weights: np.ndarray | None, size: int,
                   n: int = N_PERMUTATIONS) -> np.ndarray:
    """Effective number over `size` points drawn without replacement from the population.

    This is the distribution a class of that size would show if its activity code carried
    no information on the profile of its members, which is the null of Section 2.6.
    """
    inv, n_cat = _codes(pop_labels)
    w = np.ones(len(inv)) if pop_weights is None else np.asarray(pop_weights, float)
    out = np.empty(n)
    for i in range(n):
        idx = RNG.choice(len(inv), size=size, replace=False)
        out[i] = effective_number(np.bincount(inv[idx], weights=w[idx], minlength=n_cat))
    return out


# ---------------------------------------------------------------------- the table
def build_table(users: pd.DataFrame, groups: pd.DataFrame,
                level: str = ACTIVITY_LEVEL, n_min: int | None = None) -> pd.DataFrame:
    """One row per point of a published profile, with its profile, class and energy."""
    cols = ["pod", level, "E"] + [c for c in ("D_TIPTA",) if c in users.columns]
    u = users[cols].rename(columns={level: "activity"})
    g = groups[~groups["below_n_min"]] if "below_n_min" in groups else groups
    m = g.merge(u, on="pod", how="inner")
    m = m[m["activity"].notna()].copy()
    m["activity"] = m["activity"].astype(str)
    m["profile"] = "DDSLP_" + m["group"].astype(int).astype(str)
    if n_min is None:
        n_min = n_min_classes(m["profile"].nunique())
    small = m["activity"].value_counts()
    rare = set(small[small < n_min].index)
    m["activity_grouped"] = np.where(m["activity"].isin(rare), RESIDUAL, m["activity"])
    m.attrs["n_min_classes"] = n_min
    return m


def multiplicity(m: pd.DataFrame, weight: str = "pod", with_null: bool = True) -> pd.DataFrame:
    """M1, read on the row: how many profiles each activity class spans."""
    rows = []
    pop_w = None if weight == "pod" else m["E"].to_numpy()
    n_max = m["profile"].nunique()
    pop_counts = (m.groupby("profile").size() if weight == "pod"
                  else m.groupby("profile")["E"].sum()).to_numpy()
    reference = effective_number(pop_counts)
    null_cache: dict[int, np.ndarray] = {}
    for cls, x in m.groupby("activity_grouped"):
        w = None if weight == "pod" else x["E"].to_numpy()
        counts = (x.groupby("profile").size() if weight == "pod"
                  else x.groupby("profile")["E"].sum()).to_numpy()
        lo, hi = bootstrap_effective(x["profile"].to_numpy(), w)
        eff = effective_number(counts)
        row = {"activity": cls, "weight": weight, "n_pod": len(x),
               "energy_kWh": float(x["E"].sum()),
               "M1_effective": eff,
               #Lorenzo Giannuzzo: An effective number cannot exceed the number of categories
               # available, so it grows with K by construction and two runs at
               # different K are not comparable until it is divided by that maximum.
               "M1_evenness": eff / n_max,
               "M1_ci_low": lo, "M1_ci_high": hi,
               "M1_hard80": hard_count(counts),
               "dominant_share": float(counts.max() / counts.sum()),
               "reference_uninformative": reference}
        if with_null:
            n = len(x)
            if n not in null_cache:
                null_cache[n] = null_effective(m["profile"].to_numpy(), pop_w, n)
            nul = null_cache[n]
            row.update({"null_mean": float(np.mean(nul)),
                        "null_p05": float(np.percentile(nul, 5)),
                        "null_p95": float(np.percentile(nul, 95)),
                        #Lorenzo Giannuzzo: the share of random classes of the same size
                        # spanning no more profiles than this one; below 0.05 the class is
                        # more coherent than a random label of its size
                        "null_quantile": float(np.mean(nul <= eff)),
                        "coherent_at_5pct": bool(np.mean(nul <= eff) < 0.05)})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["activity", "weight", "n_pod", "M1_effective"])
    return pd.DataFrame(rows).sort_values("M1_effective", ascending=False)


def aggregation(m: pd.DataFrame, weight: str = "pod") -> pd.DataFrame:
    """M2, read on the column: how many activity classes each profile subsumes."""
    rows = []
    n_max = m["activity_grouped"].nunique()
    for prof, x in m.groupby("profile"):
        w = None if weight == "pod" else x["E"].to_numpy()
        counts = (x.groupby("activity_grouped").size() if weight == "pod"
                  else x.groupby("activity_grouped")["E"].sum()).to_numpy()
        lo, hi = bootstrap_effective(x["activity_grouped"].to_numpy(), w)
        dom = x["activity"].str.startswith(DOMESTIC_PREFIX)
        rows.append({"profile": prof, "weight": weight, "n_pod": len(x),
                     "energy_kWh": float(x["E"].sum()),
                     "M2_effective": effective_number(counts),
                     "M2_evenness": effective_number(counts) / n_max,
                     "M2_ci_low": lo, "M2_ci_high": hi,
                     "M2_hard80": hard_count(counts),
                     "n_classes_present": int((counts > 0).sum()),
                     "dominant_share": float(counts.max() / counts.sum()),
                     "share_domestic_pod": float(dom.mean()),
                     "share_domestic_energy": float(x.loc[dom, "E"].sum() / x["E"].sum())
                     if x["E"].sum() > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("M2_effective", ascending=False)


def lift_table(m: pd.DataFrame) -> pd.DataFrame:
    """Eq. 11, lift(a, c) = p(c|a) / p(c) = p(a, c) / (p(a) p(c)), by points and by energy."""
    out = []
    for weight in ("pod", "energy"):
        ct = (pd.crosstab(m["profile"], m["activity_grouped"]) if weight == "pod"
              else pd.crosstab(m["profile"], m["activity_grouped"], values=m["E"],
                               aggfunc="sum").fillna(0.0))
        tot = ct.to_numpy().sum()
        p_joint = ct / tot
        p_a = p_joint.sum(axis=1)
        p_c = p_joint.sum(axis=0)
        lift = p_joint.div(p_a, axis=0).div(p_c, axis=1)
        long = lift.stack().rename("lift").reset_index()
        long["weight"] = weight
        long["p_class_given_profile"] = (ct.div(ct.sum(axis=1), axis=0)).stack().to_numpy()
        long["p_profile_given_class"] = (ct.div(ct.sum(axis=0), axis=1)).stack().to_numpy()
        long["count"] = ct.stack().to_numpy()
        out.append(long)
    return pd.concat(out, ignore_index=True)


def national_frame(m: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """One row per point of a published profile and national profile applied to it."""
    g = assignment.gse_profile(users)[["pod", "gse_column"]]
    g = g[g["gse_column"].isin(GSE_KEEP)].rename(columns={"gse_column": "national"})
    a = assignment.arera_key(users)
    a = a[a["arera_applicable"]].copy()
    a["national"] = ("ARERA " + a["arera_class"].astype(str) + " "
                     + a["arera_residency"].astype(str))
    both = pd.concat([g[["pod", "national"]], a[["pod", "national"]]], ignore_index=True)
    return m.merge(both, on="pod", how="inner")


def declared_domestic(national: str) -> bool:
    if str(national).startswith("ARERA"):
        return True
    return DECLARED_DOMESTIC.get(str(national), True)


def real_coverage(m: pd.DataFrame, users: pd.DataFrame,
                  weight: str = "pod") -> tuple[pd.DataFrame, pd.DataFrame]:
    """M3 on the users each national profile is applied to, and its reach (Eq. 12)."""
    x = national_frame(m, users)
    n_cls_max = m["activity_grouped"].nunique()
    n_prof_max = m["profile"].nunique()
    #Lorenzo Giannuzzo: the composition of every behaviour over all its members, not only
    # over the points a given national profile reaches, since Eq. 12 asks whom the
    # behaviours it is built on describe
    comp = (pd.crosstab(m["profile"], m["activity_grouped"]) if weight == "pod"
            else pd.crosstab(m["profile"], m["activity_grouped"], values=m["E"],
                             aggfunc="sum").fillna(0.0))
    comp = comp.div(comp.sum(axis=1), axis=0)
    rows, reach_rows = [], []
    for prof, y in x.groupby("national"):
        w = None if weight == "pod" else y["E"].to_numpy()
        counts = (y.groupby("activity_grouped").size() if weight == "pod"
                  else y.groupby("activity_grouped")["E"].sum()).to_numpy()
        lo, hi = bootstrap_effective(y["activity_grouped"].to_numpy(), w)
        dd = (y.groupby("profile").size() if weight == "pod"
              else y.groupby("profile")["E"].sum())
        lo_d, hi_d = bootstrap_effective(y["profile"].to_numpy(), w)
        q = dd / dd.sum()
        reach = (comp.loc[q.index].mul(q, axis=0)).sum(axis=0)
        dom_cols = [c for c in reach.index if str(c).startswith(DOMESTIC_PREFIX)]
        inside = reach[dom_cols].sum() if declared_domestic(prof) else 1.0 - reach[dom_cols].sum()
        rows.append({"national_profile": prof, "weight": weight, "n_pod": len(y),
                     "declared": 1.0,
                     "declared_domestic": declared_domestic(prof),
                     "M3_classes_effective": effective_number(counts),
                     "M3_classes_evenness": effective_number(counts) / n_cls_max,
                     "M3_ci_low": lo, "M3_ci_high": hi,
                     "M3_classes_hard80": hard_count(counts),
                     "n_classes_present": int((counts > 0).sum()),
                     "M3_ddslp_effective": effective_number(dd.to_numpy()),
                     "M3_ddslp_evenness": effective_number(dd.to_numpy()) / n_prof_max,
                     "M3_ddslp_ci_low": lo_d, "M3_ddslp_ci_high": hi_d,
                     "n_ddslp_present": int((dd > 0).sum()),
                     "reach_outside_category": float(1.0 - inside),
                     "reach_domestic_share": float(reach[dom_cols].sum())})
        for c, v in reach.items():
            reach_rows.append({"national_profile": prof, "weight": weight,
                               "activity": c, "reach_share": float(v)})
        for a_, v in q.items():
            reach_rows.append({"national_profile": prof, "weight": weight,
                               "activity": f"__profile__{a_}", "reach_share": float(v)})
    return (pd.DataFrame(rows).sort_values("M3_ddslp_effective", ascending=False),
            pd.DataFrame(reach_rows))


def sensitivity_k(users: pd.DataFrame, groups: pd.DataFrame, base_m1: pd.DataFrame) -> pd.DataFrame:
    """Section 3.5: the mapping on the same tree cut at K + offset."""
    path = CACHE / "user_linkage.npy"
    if not path.exists() or not K_OFFSETS:
        return pd.DataFrame()
    from scipy.cluster.hierarchy import fcluster
    from scipy.stats import spearmanr
    Z = np.load(path)
    K0 = int(groups["group"].nunique())
    base = base_m1.set_index("activity")["M1_effective"]
    rows = []
    for off in [0] + [int(o) for o in K_OFFSETS]:
        K = K0 + off
        if K < 2:
            continue
        lab = fcluster(Z, K, criterion="maxclust")
        g = pd.DataFrame({"pod": groups["pod"].to_numpy(), "group": lab})
        sizes = g["group"].value_counts()
        g["below_n_min"] = g["group"].map(sizes < n_min_groups(K))
        mk = build_table(users, g, n_min=base_m1.attrs.get("n_min_classes"))
        if mk.empty or mk["profile"].nunique() < 2:
            #Lorenzo Giannuzzo: a cut that leaves fewer than two published profiles has no
            # multiplicity to measure; it is recorded as such rather than stopping the stage
            rows.append({"K": K, "offset": off, "published_profiles": int(mk["profile"].nunique()),
                         "pods_in_published": int(len(mk))})
            continue
        m1k = multiplicity(mk, "pod", with_null=False).set_index("activity")["M1_effective"]
        common = base.index.intersection(m1k.index)
        rho = spearmanr(base[common], m1k[common]).statistic if len(common) > 2 else np.nan
        pdmm = national_frame(mk, users)
        pdmm = pdmm[pdmm["national"] == "PDMM"]
        rows.append({"K": K, "offset": off,
                     "published_profiles": int(mk["profile"].nunique()),
                     "pods_in_published": int(len(mk)),
                     "M1_median": float(m1k.median()),
                     "reference_uninformative": effective_number(
                         mk.groupby("profile").size().to_numpy()),
                     "spearman_M1_vs_base": float(rho),
                     "PDMM_behaviours": effective_number(
                         pdmm.groupby("profile").size().to_numpy()) if len(pdmm) else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    print(f"\n{'='*78}\n  MAPPING, Section 2.6\n{'='*78}")

    users = pd.read_parquet(CACHE / "users.parquet")
    groups = pd.read_parquet(CACHE / "groups.parquet")
    m = build_table(users, groups)
    n_min = m.attrs["n_min_classes"]
    n_out = int(groups["below_n_min"].sum()) if "below_n_min" in groups else 0
    print(f"  {len(m)} points in {m['profile'].nunique()} published profiles "
          f"({n_out} points in groups below n_min left out), "
          f"{m['activity_grouped'].nunique()} classes at {ACTIVITY_LEVEL} (n_min = {n_min})")

    ct = pd.crosstab(m["activity_grouped"], m["profile"])
    ct.to_csv(OUT / "contingency_pod.csv")
    pd.crosstab(m["activity_grouped"], m["profile"],
                values=m["E"], aggfunc="sum").fillna(0.0).to_csv(OUT / "contingency_energy.csv")

    m1_base = None
    for weight in ("pod", "energy"):
        m1 = multiplicity(m, weight)
        m1.to_csv(OUT_M1 / f"m1_multiplicity_{weight}.csv", index=False)
        if weight == "pod":
            m1_base = m1
        aggregation(m, weight).to_csv(OUT_M2 / f"m2_aggregation_{weight}.csv", index=False)
        cov, reach = real_coverage(m, users, weight)
        cov.to_csv(OUT_M3 / f"m3_coverage_{weight}.csv", index=False)
        reach.to_csv(OUT_M3 / f"m3_reach_{weight}.csv", index=False)
    lift_table(m).to_csv(OUT_M2 / "lift.csv", index=False)

    #Lorenzo Giannuzzo: Section 3.5, the class level of the taxonomy and the number of profiles
    if SENSITIVITY_LEVEL in users.columns:
        m_l2 = build_table(users, groups, SENSITIVITY_LEVEL)
        multiplicity(m_l2, "pod", with_null=False).to_csv(
            OUT_M1 / f"m1_multiplicity_pod_{SENSITIVITY_LEVEL}.csv", index=False)
    m1_base.attrs["n_min_classes"] = n_min
    sk = sensitivity_k(users, groups, m1_base)
    if len(sk):
        sk.to_csv(OUT / "sensitivity_k.csv", index=False)
        print("\n  Sensitivity to K (same tree):")
        print(sk.round(3).to_string(index=False))

    m1 = pd.read_csv(OUT_M1 / "m1_multiplicity_pod.csv")
    m2 = pd.read_csv(OUT_M2 / "m2_aggregation_pod.csv")
    K = m["profile"].nunique()
    C = m["activity_grouped"].nunique()
    print(f"\n  K = {K} published profiles, C = {C} activity classes")
    print(f"  uninformative reference exp(H(A)) = {m1['reference_uninformative'].iloc[0]:.2f}")
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
