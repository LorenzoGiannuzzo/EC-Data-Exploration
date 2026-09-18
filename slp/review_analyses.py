"""Review analyses — the tables answering the comments of the external review.

Every analysis writes CSV tables under paper_results/7_review/tables and is independent of
the others, so that one that cannot run does not cost the others their results.

  R1  aggregation size         the portfolio misallocation of every family as a function of
                               the number of users aggregated, from single users to the
                               whole population (comment M2)
  R2  partition-free tests     how far the activity class organizes the space of behaviors
                               without any clustering: share of variance explained,
                               permutation test, neighbor purity, silhouette (comment M3)
  R3  bootstrap of Table 2     confidence intervals of the distances between national and
                               data-driven profiles, stability of the nearest profile and of
                               the uncovered profile (comment M5)
  R4  sensitivity              the metrics of Section 2.6 and the allocation error recomputed
                               on the partitions obtained with other values of the weight of
                               the allocation features, of the zero-replacement threshold and
                               of K (comment M8)
  R5  attrition                composition of the points read, retained and clustered
                               (comment on Section 2.2)
  R6  grid                     the data-driven and activity-based catalogs built on the
                               seasonal and on the monthly grid (comment on Section 2.4)

Run
    python main.py --stage review
    python review_analyses.py --only R2 R5

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

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extensions as X  # noqa: E402
from common import assignment  # noqa: E402
from common.config import load_config  # noqa: E402

_CFG = load_config()
OUT = _CFG.results_dir("review")
REV = _CFG.get("review", {}) or {}
SEED = int(_CFG.get("clustering.random_state", 42))
SIZES = list(REV.get("aggregation_sizes", [1, 10, 25, 50, 100, 250, 500, 1000]))
SHUFFLES = int(REV.get("aggregation_shuffles", 5))
N_PERM = int(REV.get("permutations", 199))
N_BOOT = int(REV.get("bootstrap_table2", 200))
KNN = int(REV.get("neighbors", 10))
SIL_SAMPLE = int(REV.get("silhouette_sample", 4000))
SENS_PORTFOLIO = bool(REV.get("sensitivity_portfolio", True))

FAM_SHORT = {"GSE": "GSE", "ARERA": "ARERA", "DD_in": "Data-driven", "DD_cv": "Data-driven, out-of-sample",
             "ACT_in": "Activity-based", "ACT_cv": "Activity-based, out-of-sample"}


def save(table: pd.DataFrame, name: str) -> None:
    folder = OUT / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / f"{name}.csv", index=False)
    print(f"  {name} (table)")


def is_domestic_label(code: object) -> bool:
    return str(code).startswith(("DO",))


# ============================================================ shared: references per user
def user_month_masks(inp: X.Inputs, refs_of: dict[str, dict]) -> dict:
    """The user-months every family of a scope can be evaluated on, and the scaling of each.

    refs_of maps a pod to {family: (curve, banded)}. Returns, for every pod, the list of
    valid months, following the rule of the comparison stage: at least the minimum number
    of observed hours and a positive energy for the metered curve and for every reference.
    """
    import comparison as cp
    hcal = inp.hcal
    month = hcal["month"].to_numpy()
    pods, obs, seen = inp.observed
    pos = {p: i for i, p in enumerate(pods)}
    valid = {}
    for pod, refs in refs_of.items():
        i = pos.get(pod)
        if i is None:
            continue
        ms = []
        for m in range(1, 13):
            hours = (month == m) & np.repeat(seen[i], 24)
            if hours.sum() < cp.MIN_HOURS_PER_MONTH or obs[i][hours].sum() <= 0:
                continue
            if all(ref[hours].sum() > 0 for ref, _ in refs.values()):
                ms.append(m)
        if ms:
            valid[pod] = ms
    return valid


def national_refs(inp: X.Inputs, pod: object) -> dict:
    import comparison as cp
    arera, gse = inp.national
    a = inp.assign.set_index("pod")
    out = {}
    if pod not in a.index:
        return out
    rec = a.loc[pod]
    if rec["gse_column"] in gse:
        out["GSE"] = (gse[rec["gse_column"]], cp.is_banded(f"GSE {rec['gse_column']}"))
    if rec["arera_applicable"] and (rec["arera_class"], rec["arera_residency"]) in arera:
        out["ARERA"] = (arera[(rec["arera_class"], rec["arera_residency"])], False)
    return out


def scaled_matrix(inp: X.Inputs, pods_scope: list, valid: dict, ref_fn) -> tuple[np.ndarray, np.ndarray]:
    """Metered and reference hourly curves of the users of a scope, over their valid months.

    ref_fn(pod) returns (curve, banded) or None. The reference of each user-month receives
    the metered energy of that user-month, per band for a banded profile, as in Section 2.5.
    """
    import comparison as cp
    hcal = inp.hcal
    month = hcal["month"].to_numpy()
    bands = hcal["band"].to_numpy()
    pods, obs, seen = inp.observed
    pos = {p: i for i, p in enumerate(pods)}
    O = np.zeros((len(pods_scope), len(hcal)), dtype="float32")
    R = np.zeros_like(O)
    for j, pod in enumerate(pods_scope):
        i = pos[pod]
        ref = ref_fn(pod)
        if ref is None:
            continue
        curve, banded = ref
        for m in valid[pod]:
            hours = (month == m) & np.repeat(seen[i], 24)
            o = obs[i][hours]
            O[j, hours] = o
            R[j, hours] = cp.scale_reference(curve[hours], o, bands[hours], banded)
    return O, R


def portfolio_tv(O: np.ndarray, R: np.ndarray, month: np.ndarray, hod: np.ndarray) -> dict:
    """Energy-weighted monthly TV of each row (a portfolio), hourly and on the mean day."""
    tv_h = np.zeros(len(O))
    tv_d = np.zeros(len(O))
    wsum = np.zeros(len(O))
    for m in np.unique(month):
        sel = month == m
        o, r = O[:, sel].astype("float64"), R[:, sel].astype("float64")
        so, sr = o.sum(axis=1), r.sum(axis=1)
        ok = (so > 0) & (sr > 0)
        if not ok.any():
            continue
        oo, rr = o[ok], r[ok]
        th = 0.5 * np.abs(oo / so[ok, None] - rr / sr[ok, None]).sum(axis=1)
        hh = hod[sel]
        do = np.stack([oo[:, hh == h].sum(axis=1) for h in range(24)], axis=1)
        dr = np.stack([rr[:, hh == h].sum(axis=1) for h in range(24)], axis=1)
        td = 0.5 * np.abs(do / do.sum(1, keepdims=True) - dr / dr.sum(1, keepdims=True)).sum(axis=1)
        tv_h[ok] += th * so[ok]
        tv_d[ok] += td * so[ok]
        wsum[ok] += so[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"hourly": tv_h / wsum, "mean_day": tv_d / wsum, "energy": wsum}


# ============================================================ R1 aggregation size (M2)
def run_R1(inp: X.Inputs) -> None:
    from scipy import sparse
    fam = X.build_families(inp)
    hcal = inp.hcal
    month = hcal["month"].to_numpy()
    hod = hcal["hour"].to_numpy()
    groups = inp.groups
    scopes = {"domestic": ("GSE", "ARERA", "DD_in", "DD_cv", "ACT_in", "ACT_cv"),
              "non_domestic": ("GSE", "DD_in", "DD_cv", "ACT_in", "ACT_cv")}
    refs_of = {s: {} for s in scopes}
    for pod in groups["pod"]:
        r = national_refs(inp, pod)
        for name in ("DD_in", "DD_cv", "ACT_in", "ACT_cv"):
            c = X.reference_of(fam, name, pod)
            if c is not None:
                r[name] = (c, False)
        if "ARERA" in r and len(r) == 6:
            refs_of["domestic"][pod] = r
        elif "ARERA" not in r and len(r) == 5:
            refs_of["non_domestic"][pod] = r
    rng = np.random.default_rng(SEED + 11)
    rows = []
    for scope, fams in scopes.items():
        valid = user_month_masks(inp, refs_of[scope])
        pods_scope = sorted(valid)
        N = len(pods_scope)
        if N == 0:
            continue
        sizes = [n for n in SIZES if n <= N] + ([N] if N not in SIZES else [])
        plans = {}
        for n in sizes:
            reps = 1 if n == N else (1 if n == 1 else SHUFFLES)
            plans[n] = []
            for _ in range(reps):
                perm = rng.permutation(N)
                P = N // n
                idx = perm[:P * n]
                A = sparse.csr_matrix((np.ones(P * n, dtype="float32"),
                                       (np.repeat(np.arange(P), n), idx)), shape=(P, N))
                plans[n].append(A)
        stats = {n: {} for n in sizes}
        for f in fams:
            O, R = scaled_matrix(inp, pods_scope, valid, lambda p, f=f: refs_of[scope][p][f])
            for n in sizes:
                vals = []
                for A in plans[n]:
                    so = np.asarray(A @ O) if n > 1 else O
                    sr = np.asarray(A @ R) if n > 1 else R
                    vals.append(portfolio_tv(so, sr, month, hod)["hourly"])
                stats[n][f] = np.concatenate(vals)
            del O, R
            print(f"    {scope}: {X.FAMILY_LABEL[f]} done")
        for n in sizes:
            r = {"Population": scope, "Users per aggregate [-]": n,
                 "Aggregates evaluated [-]": len(stats[n][fams[0]])}
            for f in fams:
                v = stats[n][f]
                v = v[np.isfinite(v)]
                r[f"{FAM_SHORT[f]}, median [%]"] = X.pct(np.median(v))
                r[f"{FAM_SHORT[f]}, 10th to 90th percentile [%]"] = (
                    f"{X.pct(np.percentile(v, 10))} to {X.pct(np.percentile(v, 90))}")
            rows.append(r)
    t = pd.DataFrame(rows)
    save(t, "table_r1_misallocation_by_aggregate_size")


# ============================================================ R2 partition-free tests (M3)
def _onehot(labels: np.ndarray) -> tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(pd.Series(labels).astype(str))
    return codes, len(uniq)


def ss_between(Z: np.ndarray, codes: np.ndarray, g: int) -> float:
    from scipy import sparse
    n = np.bincount(codes, minlength=g).astype(float)
    onehot = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(g, len(codes)))
    sums = np.asarray(onehot @ Z)
    mu = Z.mean(axis=0)
    nz = n > 0
    return float(((sums[nz] / n[nz, None] - mu) ** 2 * n[nz, None]).sum())


def run_R2(inp: X.Inputs) -> None:
    import mapping
    from sklearn.metrics import pairwise_distances, silhouette_score
    from sklearn.neighbors import NearestNeighbors
    groups = inp.groups
    m = mapping.build_table(inp.users, groups)
    pods_X, Xm = X.user_X(inp, m["pod"].to_numpy())
    info = m.set_index("pod").reindex(pods_X)
    a = inp.assign.set_index("pod")
    power = assignment.arera_class(inp.users.set_index("pod")["D_POTC"]).astype(object)

    #Lorenzo Giannuzzo: the second space is the one standard profiles are built on, the mean
    # curves of each user on the regulatory grid, each normalized to one, so that the test does
    # not depend on the representation this work adopts
    pc = inp.pod_cell
    ix = np.array([pc["index"][p] for p in pods_X])
    s = pc["sums"][ix].reshape(len(ix), pc["sums"].shape[1], 24, 4).sum(axis=3)
    tot = s.sum(axis=2, keepdims=True)
    Y = np.divide(s, tot, out=np.zeros_like(s), where=tot > 0).reshape(len(ix), -1)
    Y = (Y - Y.mean(0)) / np.where(Y.std(0) > 0, Y.std(0), 1)
    spaces = {"Two-stage representation (Section 2.3)": Xm,
              "Mean curves on the regulatory grid": Y}
    act = info["activity_grouped"].to_numpy()
    nondom = ~np.array([is_domestic_label(c) for c in info["activity"]])
    labelings = [
        ("Activity class", np.ones(len(pods_X), bool), act),
        ("Activity class, non-domestic points only", nondom, act),
        ("Tariff category", np.ones(len(pods_X), bool), a.reindex(pods_X)["gse_category"].to_numpy()),
        ("Contractual power class", np.ones(len(pods_X), bool), power.reindex(pods_X).astype(str).to_numpy()),
        ("Data-driven partition (reference)", np.ones(len(pods_X), bool), info["group"].astype(str).to_numpy()),
    ]
    rng = np.random.default_rng(SEED + 12)
    rows = []
    for sname, Z in spaces.items():
        nn = NearestNeighbors(n_neighbors=KNN + 1).fit(Z)
        _, nb = nn.kneighbors(Z)
        nb = nb[:, 1:]
        for lname, sel, lab in labelings:
            ok = sel & pd.notna(lab) & (pd.Series(lab).astype(str) != "nan").to_numpy()
            Zs = Z[ok]
            codes, g = _onehot(lab[ok])
            n = len(codes)
            if g < 2 or n < 10:
                continue
            sst = float(((Zs - Zs.mean(0)) ** 2).sum())
            ssb = ss_between(Zs, codes, g)
            F = (ssb / (g - 1)) / ((sst - ssb) / (n - g))
            null_ssb = np.array([ss_between(Zs, rng.permutation(codes), g) for _ in range(N_PERM)])
            p_F = (np.sum(null_ssb >= ssb) + 1) / (N_PERM + 1)
            #Lorenzo Giannuzzo: neighbor purity on the neighbors found among the points of the
            # labeling, so that a subset is not scored against points outside it
            if ok.all():
                nbs = nb
                full_codes = codes
            else:
                nn_s = NearestNeighbors(n_neighbors=KNN + 1).fit(Zs)
                nbs = nn_s.kneighbors(Zs)[1][:, 1:]
                full_codes = codes
            purity = float((full_codes[nbs] == full_codes[:, None]).mean())
            null_pur = np.array([(lambda c: (c[nbs] == c[:, None]).mean())(rng.permutation(full_codes))
                                 for _ in range(min(N_PERM, 199))])
            p_pur = (np.sum(null_pur >= purity) + 1) / (len(null_pur) + 1)
            sub = rng.choice(n, size=min(SIL_SAMPLE, n), replace=False)
            D = pairwise_distances(Zs[sub])
            try:
                sil = silhouette_score(D, codes[sub], metric="precomputed")
                null_sil = [silhouette_score(D, rng.permutation(codes[sub]), metric="precomputed")
                            for _ in range(30)]
                sil_txt, sil95 = X.fmt(sil), X.fmt(np.percentile(null_sil, 95))
            except ValueError:
                sil_txt, sil95 = "", ""
            rows.append({"Space": sname, "Labeling": lname, "Points of delivery [-]": n,
                         "Groups [-]": g,
                         "Share of variance explained [%]": X.pct(ssb / sst, 2),
                         "Pseudo-F [-]": X.fmt(F, 1),
                         "Permutation p-value, variance [-]": f"{p_F:.4f}",
                         f"Same-label share of the {KNN} nearest neighbors [%]": X.pct(purity),
                         "Same as random labels, mean [%]": X.pct(null_pur.mean()),
                         "Neighbor purity over random [-]": X.fmt(purity / null_pur.mean(), 2),
                         "Permutation p-value, neighbors [-]": f"{p_pur:.4f}",
                         "Silhouette [-]": sil_txt,
                         "Silhouette of random labels, 95th percentile [-]": sil95})
            print(f"    {sname[:24]} | {lname}")
    save(pd.DataFrame(rows), "table_r2_partition_free_tests")


# ============================================================ R3 bootstrap of Table 2 (M5)
def tv_monthly(v: np.ndarray, r: np.ndarray, month: np.ndarray, bands: np.ndarray, banded: bool) -> float:
    import comparison as cp
    vals, w = [], []
    for m in range(1, 13):
        sel = month == m
        ref = cp.scale_reference(r[sel], v[sel], bands[sel], banded)
        t = X.tv(v[sel], ref)
        if np.isfinite(t) and v[sel].sum() > 0:
            vals.append(t)
            w.append(v[sel].sum())
    return float(np.average(vals, weights=w)) if vals else np.nan


def run_R3(inp: X.Inputs) -> None:
    import comparison as cp
    arera, gse = inp.national
    hcal = inp.hcal
    month = hcal["month"].to_numpy()
    bands = hcal["band"].to_numpy()
    groups = inp.groups
    pc = inp.pod_cell
    ix = pc["index"]
    members = {g: np.array([ix[p] for p in x["pod"] if p in ix]) for g, x in groups.groupby("group")}
    keys = sorted(members)
    nat = {f"ARERA {k[0]} {k[1]}": (v, False) for k, v in arera.items()}
    nat.update({f"GSE {k}": (v, cp.is_banded(f"GSE {k}")) for k, v in gse.items()})
    names = list(nat)

    def distances(mem: dict) -> np.ndarray:
        cat = X.catalog(inp, {g: mem[g] for g in keys})
        dd = X.expand(inp, cat)
        D = np.zeros((len(names), len(keys)))
        for i, nm in enumerate(names):
            r, banded = nat[nm]
            for j in range(len(keys)):
                D[i, j] = tv_monthly(dd[j], r, month, bands, banded)
        return D

    base = distances(members)
    b1 = _CFG.results_dir("comparison") / "b1_ddslp_vs_national.csv"
    if b1.exists():
        ref = pd.read_csv(b1)
        ref = ref[ref["setting"] == "S1_monthly"].pivot_table(index="national", columns="ddslp",
                                                                values="total_variation")
        diffs = [abs(base[i, j] - ref.loc[nm, f"DDSLP_{g}"]) for i, nm in enumerate(names)
                 for j, g in enumerate(keys) if nm in ref.index and f"DDSLP_{g}" in ref.columns]
        if diffs:
            print(f"    base distances reproduce Table 2 within {max(diffs):.4f}")
    rng = np.random.default_rng(SEED + 13)
    boots = np.zeros((N_BOOT, len(names), len(keys)))
    for b in range(N_BOOT):
        mem = {g: rng.choice(members[g], size=len(members[g]), replace=True) for g in keys}
        boots[b] = distances(mem)
        if (b + 1) % 20 == 0:
            print(f"    bootstrap {b + 1}/{N_BOOT}")
    np.save(OUT / "r3_bootstrap_distances.npy", boots)

    rows = []
    order = np.argsort(base.min(axis=1))
    for i in order:
        j1, j2 = np.argsort(base[i])[:2]
        same = float(np.mean(boots[:, i, :].argmin(axis=1) == j1))
        margin = boots[:, i, j2] - boots[:, i, j1]
        rows.append({"National profile": X.arera_or_gse_name(names[i].replace("GSE ", "", 1)),
                     "Nearest data-driven profile": f"DD-SLP {keys[j1]}",
                     "TV [-]": X.fmt(base[i, j1]),
                     "95% interval [-]": f"{X.fmt(np.percentile(boots[:, i, j1], 2.5))} to "
                                         f"{X.fmt(np.percentile(boots[:, i, j1], 97.5))}",
                     "Second nearest": f"DD-SLP {keys[j2]}",
                     "TV of the second [-]": X.fmt(base[i, j2]),
                     "Margin, 95% interval [-]": f"{X.fmt(np.percentile(margin, 2.5))} to "
                                                 f"{X.fmt(np.percentile(margin, 97.5))}",
                     "Bootstraps with the same nearest profile [%]": X.pct(same, 0)})
    save(pd.DataFrame(rows), "table_r3_table2_bootstrap")

    #Lorenzo Giannuzzo: the uncovered criterion of Section 2.5 in every bootstrap: a profile is
    # uncovered when its nearest national profile lies farther than that of every residential one
    comp = _CFG.results_dir("comparison") / "ddslp_composition.csv"
    residential = set()
    if comp.exists():
        c = pd.read_csv(comp)
        residential = set(c.loc[c["residential"], "group"].astype(int))
    nearest_counts_base = np.bincount(base.argmin(axis=1), minlength=len(keys))
    counts_b = np.stack([np.bincount(boots[b].argmin(axis=1), minlength=len(keys)) for b in range(N_BOOT)])
    rows = []
    for j, g in enumerate(keys):
        mins = boots[:, :, j].min(axis=1)
        if residential:
            res_idx = [k for k, gg in enumerate(keys) if gg in residential]
            thr = boots[:, :, res_idx].min(axis=1).max(axis=1)
            unc = float(np.mean(mins > thr)) if g not in residential else np.nan
        else:
            unc = np.nan
        rows.append({"Data-driven profile": f"DD-SLP {g}",
                     "Residential": "yes" if g in residential else "no",
                     "Distance to the nearest national profile [-]": X.fmt(base[:, j].min()),
                     "95% interval [-]": f"{X.fmt(np.percentile(mins, 2.5))} to {X.fmt(np.percentile(mins, 97.5))}",
                     "National profiles for which it is the nearest [-]": int(nearest_counts_base[j]),
                     "Same count, 5th to 95th percentile [-]":
                         f"{int(np.percentile(counts_b[:, j], 5))} to {int(np.percentile(counts_b[:, j], 95))}",
                     "Bootstraps in which uncovered [%]": X.pct(unc, 0) if np.isfinite(unc) else ""})
    save(pd.DataFrame(rows), "table_r3_coverage_bootstrap")


# ============================================================ R4 sensitivity (M8)
def partition_metrics(inp: X.Inputs, gframe: pd.DataFrame, base_groups: pd.DataFrame,
                      n_min: int, portfolio: bool) -> dict:
    import mapping
    from sklearn.metrics import adjusted_rand_score
    both = gframe.merge(base_groups[["pod", "group"]], on="pod", suffixes=("", "_base"))
    m = mapping.build_table(inp.users, gframe, n_min=n_min)
    m1 = mapping.multiplicity(m, "pod", with_null=True)
    N = mapping.N_PERMUTATIONS
    p = ((m1["null_quantile"] * N + 1) / (N + 1)).to_numpy()
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    xc, nx = pd.factorize(m["activity_grouped"])[0], m["activity_grouped"].nunique()
    yc, ny = pd.factorize(m["profile"])[0], m["profile"].nunique()
    mi, hx, hy, _ = X._mi(xc, yc, nx, ny)
    nat = mapping.national_frame(m, inp.users)
    beh = nat.groupby("national").apply(
        lambda x: mapping.effective_number(x.groupby("profile").size().to_numpy()), include_groups=False)
    beh = beh[nat.groupby("national").size() >= 50]
    out = {"ARI against the adopted partition [-]": X.fmt(adjusted_rand_score(both["group_base"], both["group"])),
           "Profiles published [-]": int(m["profile"].nunique()),
           "Median class multiplicity [-]": X.fmt(m1["M1_effective"].median(), 2),
           "Uninformative reference [-]": X.fmt(m1["reference_uninformative"].iloc[0], 2),
           "Coherent classes after correction [-]": int((q < 0.05).sum()),
           "Profile entropy explained by the activity class [%]": X.pct(mi / hy if hy > 0 else np.nan),
           "Behaviors delivered by GSE PDMM [-]": X.fmt(beh.get("PDMM", np.nan), 1),
           "Behaviors delivered, range over national profiles [-]":
               f"{X.fmt(beh.min(), 1)} to {X.fmt(beh.max(), 1)}" if len(beh) else ""}
    if portfolio:
        pc = inp.pod_cell
        ix = pc["index"]
        keep = gframe[~gframe["below_n_min"]]
        mem = {g: np.array([ix[p_] for p_ in x["pod"] if p_ in ix]) for g, x in keep.groupby("group")}
        cat = X.catalog(inp, mem)
        dd = X.expand(inp, cat)
        pos = {g: i for i, g in enumerate(cat["keys"])}
        of = dict(zip(keep["pod"], keep["group"]))
        refs_of = {}
        for pod in keep["pod"]:
            r = national_refs(inp, pod)
            if "ARERA" in r:
                refs_of[pod] = {"DD": (dd[pos[of[pod]]], False)}
        valid = user_month_masks(inp, refs_of)
        pods_scope = sorted(valid)
        O, R = scaled_matrix(inp, pods_scope, valid, lambda p_: refs_of[p_]["DD"])
        hcal = inp.hcal
        res = portfolio_tv(O.sum(axis=0, keepdims=True), R.sum(axis=0, keepdims=True),
                           hcal["month"].to_numpy(), hcal["hour"].to_numpy())
        out["Portfolio misallocation of the data-driven catalog, domestic points [%]"] = X.pct(res["hourly"][0])
    return out


def run_R4(inp: X.Inputs) -> None:
    import mapping
    from scipy.cluster.hierarchy import fcluster, linkage
    cl = _CFG["clustering"]
    groups_all = pd.read_parquet(X.CACHE / "groups.parquet")
    K = int(groups_all["group"].nunique())
    base_m = mapping.build_table(inp.users, inp.groups)
    n_min = base_m.attrs["n_min_classes"]
    lam0, delta0 = float(cl["scale_weight"]), float(cl["zero_replacement"])
    variants = [("Adopted", "", lam0, delta0, K)]
    variants += [("Weight of the allocation features", v, float(v), delta0, K)
                 for v in cl.get("sensitivity_scale_weight", []) or []]
    variants += [("Zero-replacement threshold", v, lam0, float(v), K)
                 for v in cl.get("sensitivity_zero_replacement", []) or []]
    variants += [("Number of profiles K", K + o, lam0, delta0, K + o)
                 for o in mapping.K_OFFSETS if K + o >= 2]
    pods_all = groups_all["pod"].to_numpy()
    linkages: dict[tuple, np.ndarray] = {}
    rows = []
    for label, value, lam, delta, k in variants:
        key = (lam, delta)
        if key not in linkages:
            pods_X, Xv = user_X_params(inp, pods_all, lam, delta)
            linkages[key] = (pods_X, linkage(Xv, method="ward"))
        pods_X, Z = linkages[key]
        lab = fcluster(Z, k, criterion="maxclust")
        g = pd.DataFrame({"pod": pods_X, "group": lab})
        sizes = g["group"].value_counts()
        g["below_n_min"] = g["group"].isin(set(sizes[sizes < mapping.n_min_groups(k)].index))
        met = partition_metrics(inp, g, groups_all, n_min, SENS_PORTFOLIO)
        rows.append({"Choice varied": label, "Value": value if value != "" else
                     f"lambda = {lam0}, delta = {delta0}, K = {K}", **met})
        print(f"    {label} {value}: done")
    save(pd.DataFrame(rows), "table_r4_sensitivity")


def user_X_params(inp: X.Inputs, pods: np.ndarray, lam: float, delta: float):
    from clustering import user_matrix
    uv = inp.user_vectors
    uv = uv[uv["pod"].isin(pods)]
    fcols = [c for c in uv.columns if c.startswith("f_")]
    feat_cols = [c for c in uv.columns if c not in fcols and c != "pod"]
    Xv, _, _ = user_matrix(uv[fcols].to_numpy(), uv[feat_cols].to_numpy(), float(delta), float(lam),
                           str(_CFG["clustering"].get("block_normalisation", "variance")).lower())
    return uv["pod"].to_numpy(), Xv


# ============================================================ R5 attrition (Section 2.2)
def run_R5(inp: X.Inputs) -> None:
    from common.io import read_metadata, scan_data_dir, split_ateco
    idx = scan_data_dir(_CFG.data_dir)
    parts = [read_metadata(p) for p in idx["meta_file"] if p is not None]
    meta = pd.concat(parts, ignore_index=True).drop_duplicates("pod", keep="last")
    acol = {c.lower(): c for c in meta.columns}.get(str(_CFG.get("data.meta_cols.ateco", "CCATETE")).lower())
    if acol:
        meta["ateco_l1"] = [split_ateco(v)[0] for v in meta[acol]]
    retained = set(inp.users["pod"])
    clustered = set(inp.groups["pod"])
    stages = {"Points in the metadata": meta,
              "Excluded by pre-processing": meta[~meta["pod"].isin(retained)],
              "Retained": meta[meta["pod"].isin(retained)],
              "Entering the clustering": meta[meta["pod"].isin(clustered)]}
    rows = []
    for name, x in stages.items():
        if not len(x):
            continue
        cat = assignment.gse_category(x)
        dom = x[cat.values == "domestic"]
        desc = dom["FDESC"].astype(str).str.upper() if "FDESC" in dom else pd.Series(dtype=str)
        cls = assignment.arera_class(pd.to_numeric(dom["D_POTC"], errors="coerce")) if "D_POTC" in dom else None
        r = {"Population": name, "Points of delivery [-]": len(x),
             "Domestic, by tariff category [%]": X.pct((cat == "domestic").mean()),
             "Resident among domestic [%]": X.pct(desc.str.contains("RESIDENT").mean()
                                                  - desc.str.contains("NON RESID").mean()
                                                  if len(desc) else np.nan)}
        if cls is not None:
            for c in assignment.ARERA_CLASSES:
                r[f"Domestic in the {c} kW class [%]"] = X.pct((cls.astype(object) == c).mean())
        if "ateco_l1" in x:
            nd = x[cat.values != "domestic"]
            r["Non-domestic points with an activity code [%]"] = X.pct(nd["ateco_l1"].notna().mean()) if len(nd) else ""
        rows.append(r)
    save(pd.DataFrame(rows), "table_r5_attrition")


# ============================================================ R6 seasonal against monthly grid
class MonthGridInputs(X.Inputs):
    """The same inputs with the grid of Section 2.4 set to calendar months."""

    @property
    def grid(self) -> dict:
        def build():
            from generation import DAYTYPE_ORDER, MONTH_LABELS, calendar_days_per_cell, period_of
            seasons = _CFG.get("preprocessing.seasons")
            smap = {m: lab for lab, months in seasons.items() for m in months}
            d = self.days
            period = period_of(d["date"], smap, "month")
            present = set(zip(period, d["daytype"]))
            cells = [(p, t) for p in MONTH_LABELS for t in DAYTYPE_ORDER if (p, t) in present]
            cal = calendar_days_per_cell(seasons, grid="month")
            return {"cells": cells, "period": period.to_numpy(),
                    "cal_days": np.array([int(cal.get(c, 0)) for c in cells]),
                    "labels": [f"{p}|{t}" for p, t in cells]}
        return self._get("grid_month", build)

    @property
    def pod_cell(self) -> dict:
        return self._get("pod_cell_month", lambda: X.Inputs.pod_cell.fget(self))


def run_R6(inp: X.Inputs) -> None:
    minp = MonthGridInputs()
    for k in ("days", "users", "groups", "shapes", "uv", "hcal", "national", "assign", "observed"):
        if k in inp._cache:
            minp._cache[k] = inp._cache[k]
    groups = inp.groups
    act = X.activity_classes(inp)
    rows = []
    for gname, gi in (("Seasons (three) by day type", inp), ("Calendar months by day type", minp)):
        pc = gi.pod_cell
        ix = pc["index"]
        for cname, frame, col in (("Data-driven", groups, "group"), ("Activity-based", act, "activity_grouped")):
            frame = frame[frame["pod"].isin(list(ix))]
            mem = {k: np.array([ix[p] for p in x["pod"]]) for k, x in frame.groupby(col)}
            cat = X.catalog(gi, mem)
            curves = X.expand(gi, cat)
            pos = {k: i for i, k in enumerate(cat["keys"])}
            of = dict(zip(frame["pod"], frame[col]))
            refs_of = {}
            for pod in frame["pod"]:
                if "ARERA" in national_refs(inp, pod):
                    refs_of[pod] = {"ref": (curves[pos[of[pod]]], False)}
            valid = user_month_masks(inp, refs_of)
            pods_scope = sorted(valid)
            O, R = scaled_matrix(inp, pods_scope, valid, lambda p: refs_of[p]["ref"])
            hcal = inp.hcal
            um = portfolio_tv(O, R, hcal["month"].to_numpy(), hcal["hour"].to_numpy())
            #Lorenzo Giannuzzo: the user-month median needs the months separately
            month = hcal["month"].to_numpy()
            tvs = []
            for m in np.unique(month):
                sel = month == m
                so, sr = O[:, sel].sum(1), R[:, sel].sum(1)
                ok = (so > 0) & (sr > 0)
                tvs.append(0.5 * np.abs(O[ok][:, sel] / so[ok, None] - R[ok][:, sel] / sr[ok, None]).sum(1))
            tvs = np.concatenate(tvs)
            pf = portfolio_tv(O.sum(0, keepdims=True), R.sum(0, keepdims=True), month, hcal["hour"].to_numpy())
            rows.append({"Grid": gname, "Catalog": cname, "Cells per profile [-]": len(gi.grid["cells"]),
                         "User-months [-]": len(tvs), "Median TV, user-month [-]": X.fmt(np.median(tvs)),
                         "Portfolio misallocation, hourly [%]": X.pct(pf["hourly"][0]),
                         "Portfolio misallocation, mean day [%]": X.pct(pf["mean_day"][0])})
            print(f"    {gname} | {cname}")
    save(pd.DataFrame(rows), "table_r6_seasonal_against_monthly_grid")


# ============================================================ shared: in-sample families
def insample_families(inp: X.Inputs) -> dict:
    """The data-driven, activity-based and single local catalogs, built on all their users."""
    pc = inp.pod_cell
    ix = pc["index"]
    groups = inp.groups[inp.groups["pod"].isin(list(ix))]
    act = X.activity_classes(inp)
    act = act[act["pod"].isin(list(ix))]
    out = {}
    for name, frame, col in (("DD_in", groups, "group"), ("ACT_in", act, "activity_grouped")):
        mem = {k: np.array([ix[p] for p in x["pod"]]) for k, x in frame.groupby(col)}
        cat = X.catalog(inp, mem)
        out[name] = {"curves": X.expand(inp, cat), "keys": cat["keys"],
                     "of_pod": dict(zip(frame["pod"], frame[col]))}
    #Lorenzo Giannuzzo: the single local profile, the energy-weighted aggregate of every user of the
    # population, domestic and non-domestic alike, which segments nothing
    cat = X.catalog(inp, {"all": np.array([ix[p] for p in groups["pod"]])})
    out["LOCAL"] = {"curves": X.expand(inp, cat), "keys": ["all"],
                    "of_pod": {p: "all" for p in groups["pod"]}}
    return out


def refs_by_scope(inp: X.Inputs, fam: dict, names: tuple) -> dict:
    scopes = {"domestic": {}, "non_domestic": {}}
    for pod in inp.groups["pod"]:
        r = national_refs(inp, pod)
        for name in names:
            c = X.reference_of(fam, name, pod)
            if c is not None:
                r[name] = (c, False)
        if "ARERA" in r and all(n in r for n in names) and "GSE" in r:
            scopes["domestic"][pod] = r
        elif "ARERA" not in r and all(n in r for n in names) and "GSE" in r:
            scopes["non_domestic"][pod] = r
    return scopes


def user_month_stats(O: np.ndarray, R: np.ndarray, month: np.ndarray, hod: np.ndarray) -> dict:
    tvs, ws = [], []
    for m in np.unique(month):
        sel = month == m
        so, sr = O[:, sel].sum(1), R[:, sel].sum(1)
        ok = (so > 0) & (sr > 0)
        if ok.any():
            tvs.append(0.5 * np.abs(O[ok][:, sel] / so[ok, None] - R[ok][:, sel] / sr[ok, None]).sum(1))
            ws.append(so[ok])
    tv = np.concatenate(tvs)
    w = np.concatenate(ws)
    pf = portfolio_tv(O.sum(0, keepdims=True), R.sum(0, keepdims=True), month, hod)
    return {"n": len(tv), "median": float(np.median(tv)), "share": float((tv * w).sum() / w.sum()),
            "pf_hourly": float(pf["hourly"][0]), "pf_day": float(pf["mean_day"][0])}


SCOPE_TXT = {"domestic": "Domestic points carrying an ARERA profile",
             "non_domestic": "Points outside the ARERA tables"}
FAM7 = {"GSE": "GSE", "ARERA": "ARERA", "DD_in": "Data-driven", "ACT_in": "Activity-based",
        "LOCAL": "Single local profile"}


# ============================================================ R7 single local profile and aggregates
def run_R7(inp: X.Inputs) -> None:
    """The single local profile against the catalogs, on user-months, random and homogeneous aggregates."""
    from scipy import sparse
    fam = insample_families(inp)
    hcal = inp.hcal
    month = hcal["month"].to_numpy()
    hod = hcal["hour"].to_numpy()
    scopes = refs_by_scope(inp, fam, ("DD_in", "ACT_in", "LOCAL"))
    a = inp.assign.set_index("pod")
    grp = dict(zip(inp.groups["pod"], inp.groups["group"]))
    rng = np.random.default_rng(SEED + 21)
    rows_um, rows_rand, rows_hom = [], [], []
    for scope, refs_of in scopes.items():
        valid = user_month_masks(inp, refs_of)
        pods_scope = sorted(valid)
        N = len(pods_scope)
        if N == 0:
            continue
        fams = [f for f in ("GSE", "ARERA", "DD_in", "ACT_in", "LOCAL")
                if all(f in refs_of[p] for p in pods_scope)]

        def plan(labels: np.ndarray, n: int, reps: int) -> list:
            """Aggregates of n users drawn within each label, reps times."""
            out = []
            for _ in range(reps):
                rows_, cols_, k = [], [], 0
                for lab in np.unique(labels):
                    idx = rng.permutation(np.flatnonzero(labels == lab))
                    P = len(idx) // n
                    for q in range(P):
                        rows_ += [k] * n
                        cols_ += list(idx[q * n:(q + 1) * n])
                        k += 1
                if k:
                    out.append(sparse.csr_matrix((np.ones(len(cols_), dtype="float32"), (rows_, cols_)),
                                                 shape=(k, N)))
            return out

        sizes = [n for n in SIZES if 1 < n <= N]
        random_plans = {n: plan(np.zeros(N, int), n, SHUFFLES) for n in sizes}
        homog = {"Within the same data-driven profile": np.array([grp[p] for p in pods_scope])}
        if scope == "domestic":
            homog["Within the same ARERA profile"] = np.array(
                [f"{a.loc[p, 'arera_class']}|{a.loc[p, 'arera_residency']}" for p in pods_scope])
        hom_plans = {(h, n): plan(lab, n, SHUFFLES) for h, lab in homog.items()
                     for n in sizes if n <= 250}
        res_rand = {n: {} for n in sizes}
        res_hom = {k: {} for k in hom_plans}
        for f in fams:
            O, R = scaled_matrix(inp, pods_scope, valid, lambda p, f=f: refs_of[p][f])
            st = user_month_stats(O, R, month, hod)
            rows_um.append({"Population": SCOPE_TXT[scope], "Family of profiles": FAM7[f],
                            "User-months [-]": st["n"], "Median TV [-]": X.fmt(st["median"]),
                            "Misallocated share, user-months [%]": X.pct(st["share"]),
                            "Misallocated share, portfolio, hourly [%]": X.pct(st["pf_hourly"]),
                            "Misallocated share, portfolio, mean day [%]": X.pct(st["pf_day"])})
            for store, plans in ((res_rand, random_plans), (res_hom, hom_plans)):
                for key, As in plans.items():
                    vals = [portfolio_tv(np.asarray(A @ O), np.asarray(A @ R), month, hod)["hourly"] for A in As]
                    store[key][f] = np.concatenate(vals) if vals else np.array([])
            del O, R
            print(f"    {scope}: {FAM7[f]} done")
        for n in sizes:
            r = {"Population": SCOPE_TXT[scope], "Users per aggregate [-]": n}
            for f in fams:
                v = res_rand[n][f]
                r[f"{FAM7[f]}, median [%]"] = X.pct(np.nanmedian(v)) if len(v) else ""
            rows_rand.append(r)
        for (h, n) in hom_plans:
            r = {"Population": SCOPE_TXT[scope], "Aggregates drawn": h, "Users per aggregate [-]": n,
                 "Aggregates evaluated [-]": len(res_hom[(h, n)][fams[0]])}
            for f in fams:
                v = res_hom[(h, n)][f]
                r[f"{FAM7[f]}, median [%]"] = X.pct(np.nanmedian(v)) if len(v) else ""
                r[f"{FAM7[f]}, 10th to 90th percentile [%]"] = (
                    f"{X.pct(np.nanpercentile(v, 10))} to {X.pct(np.nanpercentile(v, 90))}" if len(v) else "")
            rows_hom.append(r)
    save(pd.DataFrame(rows_um), "table_r7_single_local_profile")
    save(pd.DataFrame(rows_rand), "table_r7_random_aggregates")
    save(pd.DataFrame(rows_hom), "table_r7_homogeneous_aggregates")


# ============================================================ R8 oracle profile of each user
def run_R8(inp: X.Inputs) -> None:
    """The error of the best fixed profile a user can have: its own, built on its other months."""
    import comparison as cp
    fam = insample_families(inp)
    hcal = inp.hcal
    month = hcal["month"].to_numpy()
    hod = hcal["hour"].to_numpy()
    bands = hcal["band"].to_numpy()
    labels = inp.grid["labels"]
    cell_of_label = {c: i for i, c in enumerate(labels)}
    hour_cell = cp.hcal_cell_key(hcal, labels).map(cell_of_label).to_numpy()
    scopes = refs_by_scope(inp, fam, ("DD_in", "ACT_in", "LOCAL"))

    #Lorenzo Giannuzzo: hourly sums of every user by calendar month and cell, from which the profile of
    # a user without any given month is obtained by subtraction
    d = inp.days
    ok = d["has_shape"].to_numpy()
    cell_idx = {c: i for i, c in enumerate(inp.grid["cells"])}
    ci = np.array([cell_idx.get((p_, t), -1) for p_, t in zip(inp.grid["period"], d["daytype"])])
    ym = (d["date"].dt.year * 12 + d["date"].dt.month - 1).to_numpy()
    yms = np.unique(ym[ok])
    ym_ix = {v: i for i, v in enumerate(yms)}
    pods_all = sorted(set(p for s in scopes.values() for p in s))
    pix = {p: i for i, p in enumerate(pods_all)}
    sel = ok & (ci >= 0) & d["pod"].isin(pix).to_numpy()
    pi = d.loc[sel, "pod"].map(pix).to_numpy()
    yi = np.array([ym_ix[v] for v in ym[sel]])
    cj = ci[sel]
    idx = d.loc[sel, "shape_idx"].to_numpy()
    e = d.loc[sel, "energy"].to_numpy(dtype="float64")
    n_c = len(labels)
    S = np.zeros((len(pods_all), len(yms), n_c, 24), dtype="float32")
    E = np.zeros((len(pods_all), len(yms), n_c))
    Nd = np.zeros_like(E)
    order = np.argsort(idx)
    for lo in range(0, len(order), 200_000):
        o = order[lo:lo + 200_000]
        block = np.asarray(inp.shapes[idx[o]], dtype="float32").reshape(-1, 24, 4).sum(axis=2)
        np.add.at(S, (pi[o], yi[o], cj[o]), block * e[o, None].astype("float32"))
    np.add.at(E, (pi, yi, cj), e)
    np.add.at(Nd, (pi, yi, cj), 1.0)
    from comparison import REFERENCE_YEAR
    pods_obs, obs, seen = inp.observed
    pos = {p: i for i, p in enumerate(pods_obs)}
    rows = []
    for scope, refs_of in scopes.items():
        valid = user_month_masks(inp, refs_of)
        pods_scope = sorted(valid)
        if not pods_scope:
            continue
        H = len(hcal)
        O = np.zeros((len(pods_scope), H), dtype="float32")
        R = np.zeros_like(O)
        for j, pod in enumerate(pods_scope):
            i = pos[pod]
            k = pix[pod]
            St, Et, Nt = S[k].sum(0), E[k].sum(0), Nd[k].sum(0)
            for m in valid[pod]:
                y = REFERENCE_YEAR * 12 + m - 1
                yk = ym_ix.get(y)
                s_ = St - (S[k, yk] if yk is not None else 0)
                e_ = Et - (E[k, yk] if yk is not None else 0)
                n_ = Nt - (Nd[k, yk] if yk is not None else 0)
                tot = s_.sum(1, keepdims=True)
                curve = np.divide(s_, tot, out=np.zeros_like(s_), where=tot > 0)
                mean_e = np.divide(e_, n_, out=np.zeros_like(e_), where=n_ > 0)
                hours = (month == m) & np.repeat(seen[i], 24)
                ref = curve[hour_cell[hours], hod[hours]] * mean_e[hour_cell[hours]]
                o = obs[i][hours]
                if ref.sum() <= 0:
                    continue
                O[j, hours] = o
                R[j, hours] = cp.scale_reference(ref, o, bands[hours], False)
        st = user_month_stats(O, R, month, hod)
        rows.append({"Population": SCOPE_TXT[scope], "Family of profiles": "Own profile of each user, other months",
                     "User-months [-]": st["n"], "Median TV [-]": X.fmt(st["median"]),
                     "Misallocated share, user-months [%]": X.pct(st["share"]),
                     "Misallocated share, portfolio, hourly [%]": X.pct(st["pf_hourly"]),
                     "Misallocated share, portfolio, mean day [%]": X.pct(st["pf_day"])})
        for f in ("GSE", "ARERA", "DD_in", "ACT_in", "LOCAL"):
            if not all(f in refs_of[p] for p in pods_scope):
                continue
            O2, R2 = scaled_matrix(inp, pods_scope, valid, lambda p, f=f: refs_of[p][f])
            #Lorenzo Giannuzzo: the same user-months as the own profile, so that the two are comparable
            R2[R.sum(1) == 0] = 0
            O2[R.sum(1) == 0] = 0
            st = user_month_stats(O2, R2, month, hod)
            rows.append({"Population": SCOPE_TXT[scope], "Family of profiles": FAM7[f],
                         "User-months [-]": st["n"], "Median TV [-]": X.fmt(st["median"]),
                         "Misallocated share, user-months [%]": X.pct(st["share"]),
                         "Misallocated share, portfolio, hourly [%]": X.pct(st["pf_hourly"]),
                         "Misallocated share, portfolio, mean day [%]": X.pct(st["pf_day"])})
            del O2, R2
        print(f"    {scope}: done")
    save(pd.DataFrame(rows), "table_r8_own_profile_of_each_user")


# ============================================================ R10 scales for Table 5 and the information
def run_R10(inp: X.Inputs) -> None:
    import comparison as cp
    import mapping
    #Lorenzo Giannuzzo: the information part, read against the largest share attainable given the
    # entropy of each labeling, which the imbalance of the classes caps well below one
    m = mapping.build_table(inp.users, inp.groups)
    a = inp.assign.set_index("pod")
    power = assignment.arera_class(inp.users.set_index("pod")["D_POTC"]).astype(object)
    m["gse_category"] = m["pod"].map(a["gse_category"])
    m["power_class"] = m["pod"].map(power).astype(str)
    nondom = ~m["activity"].astype(str).str.startswith("DO")
    rows = []
    for label, frame, col in (("Activity class", m, "activity_grouped"),
                              ("Activity class, non-domestic points only", m[nondom], "activity_grouped"),
                              ("Tariff category", m, "gse_category"),
                              ("Contractual power class", m, "power_class")):
        x = frame[frame[col].notna()]
        xc, nx = pd.factorize(x[col].astype(str))[0], x[col].astype(str).nunique()
        yc, ny = pd.factorize(x["profile"])[0], x["profile"].nunique()
        mi, hx, hy, _ = X._mi(xc, yc, nx, ny)
        cap = min(hx, hy) / hy if hy > 0 else np.nan
        rows.append({"Variable": label, "Points of delivery [-]": len(x),
                     "Entropy of the variable [nats]": X.fmt(hx), "Entropy of the profiles [nats]": X.fmt(hy),
                     "Mutual information [nats]": X.fmt(mi),
                     "Profile entropy explained [%]": X.pct(mi / hy),
                     "Largest share attainable [%]": X.pct(cap),
                     "Share of the attainable largest [%]": X.pct((mi / hy) / cap if cap > 0 else np.nan),
                     "Entropy of the variable shared with the profile [%]": X.pct(mi / hx if hx > 0 else np.nan)})
    save(pd.DataFrame(rows), "table_r10_information_ceiling")

    #Lorenzo Giannuzzo: the distance of each data-driven profile from its nearest neighbor within the
    # data-driven catalog, the scale against which the distance of a national profile is read
    arera, gse = inp.national
    hcal = inp.hcal
    month = hcal["month"].to_numpy()
    bands = hcal["band"].to_numpy()
    hod = hcal["hour"].to_numpy()
    labels = inp.grid["labels"]
    key = cp.hcal_cell_key(hcal, labels).to_numpy()
    pc = inp.pod_cell
    ix = pc["index"]
    mem = {g: np.array([ix[p] for p in x["pod"] if p in ix]) for g, x in inp.groups.groupby("group")}
    cat = X.catalog(inp, mem)
    dd = X.expand(inp, cat)
    dd_h = cat["curves"].reshape(*cat["curves"].shape[:2], 24, 4).sum(axis=3)
    keys = cat["keys"]
    single = {**{k: v for k, v in arera.items()}, **{k: v for k, v in gse.items() if not cp.is_banded(f"GSE {k}")}}
    nat_all = {**{k: (v, False) for k, v in arera.items()}, **{k: (v, cp.is_banded(f"GSE {k}")) for k, v in gse.items()}}
    nat_cell = {}
    for name, v in single.items():
        shp = np.zeros((len(labels), 24))
        for j, c in enumerate(labels):
            sel = key == c
            if sel.any():
                prof = np.bincount(hod[sel], weights=v[sel], minlength=24)
                shp[j] = prof / prof.sum() if prof.sum() > 0 else 0
        nat_cell[name] = shp

    def nrmsd_med(a_: np.ndarray, b_: np.ndarray) -> float:
        return float(np.median(np.sqrt(((a_ - b_) ** 2).mean(axis=1)) / (1.0 / 24)))

    rows = []
    for gi, g in enumerate(keys):
        dd_n = {h: nrmsd_med(dd_h[gi], dd_h[hi]) for hi, h in enumerate(keys) if h != g}
        dd_t = {h: tv_monthly(dd[gi], dd[hi], month, bands, False) for hi, h in enumerate(keys) if h != g}
        nat_n = {n: nrmsd_med(nat_cell[n], dd_h[gi]) for n in nat_cell}
        nat_t = {n: tv_monthly(dd[gi], v, month, bands, b) for n, (v, b) in nat_all.items()}
        bn, bt = min(dd_n, key=dd_n.get), min(dd_t, key=dd_t.get)
        nn, nt = min(nat_n, key=nat_n.get), min(nat_t, key=nat_t.get)
        rows.append({"Data-driven profile": f"DD-SLP {g}",
                     "Nearest other data-driven profile, nRMSD": f"DD-SLP {bn}",
                     "nRMSD to it [-]": X.fmt(dd_n[bn], 2),
                     "Nearest single-rate national profile, nRMSD": X.national_name(nn),
                     "nRMSD to it [-] ": X.fmt(nat_n[nn], 2),
                     "Nearest other data-driven profile, TV": f"DD-SLP {bt}",
                     "TV to it [-]": X.fmt(dd_t[bt]),
                     "Nearest national profile, TV": X.national_name(nt),
                     "TV to it [-] ": X.fmt(nat_t[nt])})
    save(pd.DataFrame(rows), "table_r10_distance_scales")


# ============================================================ shared: own profile of every user
def own_profile_matrix(inp: X.Inputs, pods_scope: list, valid: dict) -> tuple[np.ndarray, np.ndarray]:
    """Metered curves and own profiles (built on the other months of each user) over the valid months."""
    import comparison as cp
    from comparison import REFERENCE_YEAR
    hcal = inp.hcal
    month = hcal["month"].to_numpy()
    hod = hcal["hour"].to_numpy()
    bands = hcal["band"].to_numpy()
    labels = inp.grid["labels"]
    cell_of_label = {c: i for i, c in enumerate(labels)}
    hour_cell = cp.hcal_cell_key(hcal, labels).map(cell_of_label).to_numpy()
    d = inp.days
    ok = d["has_shape"].to_numpy()
    cell_idx = {c: i for i, c in enumerate(inp.grid["cells"])}
    ci = np.array([cell_idx.get((p_, t), -1) for p_, t in zip(inp.grid["period"], d["daytype"])])
    ym = (d["date"].dt.year * 12 + d["date"].dt.month - 1).to_numpy()
    yms = np.unique(ym[ok])
    ym_ix = {v: i for i, v in enumerate(yms)}
    pix = {p: i for i, p in enumerate(pods_scope)}
    sel = ok & (ci >= 0) & d["pod"].isin(pix).to_numpy()
    pi = d.loc[sel, "pod"].map(pix).to_numpy()
    yi = np.array([ym_ix[v] for v in ym[sel]])
    cj = ci[sel]
    idx = d.loc[sel, "shape_idx"].to_numpy()
    e = d.loc[sel, "energy"].to_numpy(dtype="float64")
    n_c = len(labels)
    S = np.zeros((len(pods_scope), len(yms), n_c, 24), dtype="float32")
    E = np.zeros((len(pods_scope), len(yms), n_c))
    Nd = np.zeros_like(E)
    order = np.argsort(idx)
    for lo in range(0, len(order), 200_000):
        o = order[lo:lo + 200_000]
        block = np.asarray(inp.shapes[idx[o]], dtype="float32").reshape(-1, 24, 4).sum(axis=2)
        np.add.at(S, (pi[o], yi[o], cj[o]), block * e[o, None].astype("float32"))
    np.add.at(E, (pi, yi, cj), e)
    np.add.at(Nd, (pi, yi, cj), 1.0)
    pods_obs, obs, seen = inp.observed
    pos = {p: i for i, p in enumerate(pods_obs)}
    O = np.zeros((len(pods_scope), len(hcal)), dtype="float32")
    R = np.zeros_like(O)
    for j, pod in enumerate(pods_scope):
        i = pos[pod]
        St, Et, Nt = S[j].sum(0), E[j].sum(0), Nd[j].sum(0)
        for m in valid[pod]:
            yk = ym_ix.get(REFERENCE_YEAR * 12 + m - 1)
            s_ = St - (S[j, yk] if yk is not None else 0)
            e_ = Et - (E[j, yk] if yk is not None else 0)
            n_ = Nt - (Nd[j, yk] if yk is not None else 0)
            tot = s_.sum(1, keepdims=True)
            curve = np.divide(s_, tot, out=np.zeros_like(s_), where=tot > 0)
            mean_e = np.divide(e_, n_, out=np.zeros_like(e_), where=n_ > 0)
            hours = (month == m) & np.repeat(seen[i], 24)
            ref = curve[hour_cell[hours], hod[hours]] * mean_e[hour_cell[hours]]
            o = obs[i][hours]
            O[j, hours] = o
            if ref.sum() > 0:
                R[j, hours] = cp.scale_reference(ref, o, bands[hours], False)
    return O, R


def peak_metrics(Oa: np.ndarray, Ra: np.ndarray, month: np.ndarray) -> dict:
    """Errors of the aggregate on its monthly peak, averaged over the months, for every row."""
    peak_err, peak_bias, top_err = [], [], []
    for m in np.unique(month):
        sel = np.flatnonzero(month == m)
        o, r = Oa[:, sel].astype("float64"), Ra[:, sel].astype("float64")
        ok = o.sum(1) > 0
        if not ok.any():
            continue
        po = o.max(1)
        at = o.argmax(1)
        pr = r.max(1)
        r_at = r[np.arange(len(r)), at]
        k = max(1, int(round(0.05 * len(sel))))
        top = np.argsort(-o, axis=1)[:, :k]
        eo = np.take_along_axis(o, top, 1).sum(1)
        er = np.take_along_axis(r, top, 1).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            peak_err.append(np.where(ok, np.abs(pr - po) / po, np.nan))
            peak_bias.append(np.where(ok, (r_at - po) / po, np.nan))
            top_err.append(np.where(ok, (er - eo) / eo, np.nan))
    return {"peak": np.nanmean(np.stack(peak_err), 0), "bias": np.nanmean(np.stack(peak_bias), 0),
            "top": np.nanmean(np.stack(top_err), 0)}


FAM11 = {**FAM7, "OWN": "Own profile of each user"}


# ============================================================ R11 uncertainty of the portfolio and peak errors
def run_R11(inp: X.Inputs) -> None:
    from scipy import sparse
    fam = insample_families(inp)
    hcal = inp.hcal
    month = hcal["month"].to_numpy()
    hod = hcal["hour"].to_numpy()
    scopes = refs_by_scope(inp, fam, ("DD_in", "ACT_in", "LOCAL"))
    rng = np.random.default_rng(SEED + 31)
    rows_u, rows_p = [], []
    for scope, refs_of in scopes.items():
        valid = user_month_masks(inp, refs_of)
        pods_scope = sorted(valid)
        N = len(pods_scope)
        if N == 0:
            continue
        O, R_own = own_profile_matrix(inp, pods_scope, valid)
        #Lorenzo Giannuzzo: every family is evaluated on the same user-months, those on which the own
        # profile of the user is also defined, so that the comparison is paired
        um = np.zeros((N, 13))
        np.add.at(um, (np.repeat(np.arange(N), (month > 0).sum()), np.tile(month, N)), R_own.ravel())
        keep = (um[:, month] > 0).astype("float32")
        O *= keep
        R_own *= keep
        fams = [f for f in ("GSE", "ARERA", "DD_in", "ACT_in", "LOCAL") if all(f in refs_of[p] for p in pods_scope)]
        #Lorenzo Giannuzzo: delete-a-group jackknife over random groups of users. Resampling users with
        # replacement would duplicate them and bias the misallocation of the portfolio upward, whereas
        # leaving a group out keeps every aggregate a genuine portfolio of distinct users
        G = int(REV.get("portfolio_jackknife_groups", 20))
        grp_of = rng.permutation(N) % G
        C = np.ones((G + 1, N), dtype="float32")
        for g_ in range(G):
            C[g_ + 1, grp_of == g_] = 0.0
        aggO = C @ O
        size_plans = {}
        for n in (100, 1000):
            if n < N:
                As = []
                for _ in range(SHUFFLES):
                    perm = rng.permutation(N)
                    P = N // n
                    As.append(sparse.csr_matrix((np.ones(P * n, dtype="float32"),
                                                 (np.repeat(np.arange(P), n), perm[:P * n])), shape=(P, N)))
                size_plans[n] = As
        res = {}
        for f in fams + ["OWN"]:
            if f == "OWN":
                R = R_own
            else:
                _, R = scaled_matrix(inp, pods_scope, valid, lambda p, f=f: refs_of[p][f])
                R *= keep
            aggR = C @ R
            pt = portfolio_tv(aggO, aggR, month, hod)
            pk = peak_metrics(aggO, aggR, month)
            res[f] = {"h": pt["hourly"], "d": pt["mean_day"], **pk}
            for n, As in size_plans.items():
                vals = {"peak": [], "bias": [], "top": [], "h": []}
                for A in As:
                    so, sr = np.asarray(A @ O), np.asarray(A @ R)
                    pkm = peak_metrics(so, sr, month)
                    for k_ in ("peak", "bias", "top"):
                        vals[k_].append(pkm[k_])
                    vals["h"].append(portfolio_tv(so, sr, month, hod)["hourly"])
                rows_p.append({"Population": SCOPE_TXT[scope], "Users per aggregate [-]": n, "Family of profiles": FAM11[f],
                               "Misallocated share, hourly, median [%]": X.pct(np.nanmedian(np.concatenate(vals["h"]))),
                               "Error on the monthly peak, median [%]": X.pct(np.nanmedian(np.concatenate(vals["peak"]))),
                               "Signed error at the metered peak hour, median [%]": X.pct(np.nanmedian(np.concatenate(vals["bias"]))),
                               "Error on the energy of the top 5% hours, median [%]": X.pct(np.nanmedian(np.concatenate(vals["top"])))})
            del R
            print(f"    {scope}: {FAM11[f]} done")
        base = res["ACT_in"]

        def basic(v: np.ndarray, pct_: bool = True) -> str:
            th, jk = v[0], v[1:]
            G_ = len(jk)
            se = np.sqrt((G_ - 1) / G_ * np.nansum((jk - np.nanmean(jk)) ** 2))
            a_, b_ = th - 1.96 * se, th + 1.96 * se
            return f"{X.pct(a_)} to {X.pct(b_)}" if pct_ else f"{X.fmt(100 * a_, 2)} to {X.fmt(100 * b_, 2)}"

        for f in fams + ["OWN"]:
            r_ = res[f]
            diff_all = r_["h"] - base["h"]
            diff = diff_all[1:]
            rows_u.append({"Population": SCOPE_TXT[scope], "Family of profiles": FAM11[f],
                           "Portfolio misallocation, hourly [%]": X.pct(r_["h"][0]),
                           "95% interval, hourly [%]": basic(r_["h"]),
                           "Portfolio misallocation, mean day [%]": X.pct(r_["d"][0]),
                           "95% interval, mean day [%]": basic(r_["d"]),
                           "Difference from the activity-based catalog, 95% interval [percentage points]":
                               "" if f == "ACT_in" else basic(diff_all, pct_=False),
                           "Groups left out in which the activity-based catalog still misallocates less [%]":
                               "" if f == "ACT_in" else X.pct(np.mean(diff > 0), 0),
                           "Error on the monthly peak, whole population [%]": X.pct(r_["peak"][0]),
                           "95% interval, peak [%]": basic(r_["peak"]),
                           "Signed error at the metered peak hour, whole population [%]": X.pct(r_["bias"][0]),
                           "Error on the energy of the top 5% hours, whole population [%]": X.pct(r_["top"][0])})
        del O, R_own, aggO
    save(pd.DataFrame(rows_u), "table_r11_portfolio_uncertainty_and_peak")
    save(pd.DataFrame(rows_p), "table_r11_peak_errors_by_aggregate_size")


# ============================================================ R13 activity labels of doubtful quality
def run_R13(inp: X.Inputs) -> None:
    """The mapping with the non-domestic points whose record looks like a dwelling left out."""
    import mapping
    from sklearn.neighbors import NearestNeighbors
    users = inp.users
    groups = inp.groups
    base = mapping.build_table(users, groups)
    n_min = base.attrs["n_min_classes"]
    e_thr = float(REV.get("label_energy_kwh", 1500))
    z_thr = float(REV.get("label_zero_share", 0.5))
    u = users.set_index("pod")
    nondom = ~base["activity"].astype(str).str.startswith("DO")
    E = base["pod"].map(u["E"])
    Z = base["pod"].map(u["zero_day_fraction"]) if "zero_day_fraction" in u else pd.Series(0, index=base.index)
    doubtful = set(base.loc[nondom & ((E < e_thr) | (Z > z_thr)), "pod"])
    kept = groups[~groups["pod"].isin(doubtful)]

    def summary(g: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        m = mapping.build_table(users, g, n_min=n_min)
        m1 = mapping.multiplicity(m, "pod", with_null=True)
        N = mapping.N_PERMUTATIONS
        p = ((m1["null_quantile"] * N + 1) / (N + 1)).to_numpy()
        o = np.argsort(p)
        q = np.empty(len(p))
        q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
        m1 = m1.assign(q=q)
        out = {}
        for lab, x in (("all points", m), ("non-domestic points", m[~m["activity"].astype(str).str.startswith("DO")])):
            xc, nx = pd.factorize(x["activity_grouped"])[0], x["activity_grouped"].nunique()
            yc, ny = pd.factorize(x["profile"])[0], x["profile"].nunique()
            mi, hx, hy, _ = X._mi(xc, yc, nx, ny)
            out[f"Profile entropy explained by the activity class, {lab} [%]"] = X.pct(mi / hy if hy > 0 else np.nan)
        nd = m[~m["activity"].astype(str).str.startswith("DO")]
        pods_X, Xm = X.user_X(inp, nd["pod"].to_numpy())
        lab = nd.set_index("pod").reindex(pods_X)["activity_grouped"].astype(str).to_numpy()
        codes, g_ = _onehot(lab)
        sst = float(((Xm - Xm.mean(0)) ** 2).sum())
        out["Variance of the representation explained, non-domestic points [%]"] = X.pct(ss_between(Xm, codes, g_) / sst)
        nb = NearestNeighbors(n_neighbors=KNN + 1).fit(Xm).kneighbors(Xm)[1][:, 1:]
        pur = float((codes[nb] == codes[:, None]).mean())
        rnd = float(np.mean([(lambda c: (c[nb] == c[:, None]).mean())(np.random.default_rng(SEED + b).permutation(codes)) for b in range(50)]))
        out["Neighbor purity over random labels, non-domestic points [-]"] = X.fmt(pur / rnd, 2)
        out["Points of delivery [-]"] = str(len(m))
        out["Non-domestic points of delivery [-]"] = str(len(nd))
        out["Median class multiplicity [-]"] = X.fmt(m1["M1_effective"].median(), 2)
        out["Classes coherent after correction [-]"] = str(int((q < 0.05).sum()))
        return out, m1.set_index("activity")

    s_all, c_all = summary(groups)
    s_kept, c_kept = summary(kept)
    rows = [{"Quantity": k, "All points": s_all[k], "Doubtful non-domestic labels excluded": s_kept.get(k, "")} for k in s_all]
    rows.insert(0, {"Quantity": "Criterion for a doubtful label",
                    "All points": "",
                    "Doubtful non-domestic labels excluded": f"annual energy below {e_thr:.0f} kWh or more than {100 * z_thr:.0f}% of days at zero ({len(doubtful)} points)"})
    save(pd.DataFrame(rows), "table_r13_label_quality_summary")
    rows = []
    for cls in c_all.index:
        rows.append({"Activity class": X.activity_name(cls),
                     "Points of delivery, all [-]": int(c_all.loc[cls, "n_pod"]),
                     "Class multiplicity, all [-]": X.fmt(c_all.loc[cls, "M1_effective"], 1),
                     "Coherent after correction, all": "yes" if c_all.loc[cls, "q"] < 0.05 else "no",
                     "Points of delivery, doubtful excluded [-]": int(c_kept.loc[cls, "n_pod"]) if cls in c_kept.index else "",
                     "Class multiplicity, doubtful excluded [-]": X.fmt(c_kept.loc[cls, "M1_effective"], 1) if cls in c_kept.index else "",
                     "Coherent after correction, doubtful excluded": ("yes" if c_kept.loc[cls, "q"] < 0.05 else "no") if cls in c_kept.index else ""})
    save(pd.DataFrame(rows), "table_r13_label_quality_classes")


# ============================================================================== main
ANALYSES = {
    "R1": ("misallocation by aggregate size", run_R1),
    "R2": ("partition-free tests", run_R2),
    "R3": ("bootstrap of Table 2", run_R3),
    "R4": ("sensitivity to the declared choices", run_R4),
    "R5": ("attrition", run_R5),
    "R6": ("seasonal against monthly grid", run_R6),
    "R7": ("single local profile, random and homogeneous aggregates", run_R7),
    "R8": ("own profile of each user", run_R8),
    "R10": ("scales of the information and of the distances", run_R10),
    "R11": ("uncertainty of the portfolio misallocation and peak errors", run_R11),
    "R13": ("activity labels of doubtful quality", run_R13),
}


def main(only: list[str] | None = None) -> None:
    t0 = time.time()
    print(f"\n{'='*78}\n  REVIEW ANALYSES\n{'='*78}")
    inp = X.Inputs()
    failed = []
    for key in (only or list(ANALYSES)):
        label, fn = ANALYSES[key]
        print(f"\n  [{key}] {label}")
        t1 = time.time()
        try:
            fn(inp)
            print(f"    done ({time.time() - t1:.0f}s)")
        except Exception as exc:
            failed.append(key)
            print(f"    ! {key} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n  tables under {OUT / 'tables'}   ({time.time()-t0:.0f}s)")
    if failed:
        print(f"  ! failed: {', '.join(failed)}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="analyses answering the external review")
    ap.add_argument("--only", nargs="*", choices=list(ANALYSES))
    main(ap.parse_args().only)