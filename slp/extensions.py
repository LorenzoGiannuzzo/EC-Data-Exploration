"""Extensions stage — the analyses that complete the results of Section 3.

Every analysis writes its result as a table of the paper, as CSV and as a formatted Excel
sheet, under paper_results/6_extensions/tables. The analyses are independent: one that
cannot run, for want of an input or because it fails, is reported and the others go on.

  A1  activity-based catalog      the catalog the regulation would build if it profiled
                                  by activity: one profile per activity class, built with
                                  the very procedure of Section 2.4, compared with the GSE,
                                  ARERA and data-driven families on the same user-months
  A2  out-of-sample validation    the data-driven and activity-based catalogs rebuilt on
                                  four folds of users and applied to the fifth
  A3  valuation in euro           the misallocated energy of every family at day-ahead and
                                  imbalance prices, when comparison.price_file and
                                  comparison.imbalance_price_file are set
  A4  dispersion and Eq. 7        how far the members lie from their own profile, against
                                  how far the nearest national profile lies from it
  A5  partition uncertainty       the metrics of Section 2.6 recomputed on the partitions
                                  of the subsamples used to select K
  A6  information                 mutual information between the profile of a user and its
                                  activity class, tariff category or power class
  A7  multiple testing            Benjamini-Hochberg correction of the coherence test
  A8  prosumers excluded          the partition rebuilt without the prosumers
  A9  quarter-hourly resolution   the user-month error at 15 minutes against 60 minutes
  A10 time bands                  the F1, F2 and F3 shares of every profile
  A11 representativeness          the composition of the dataset, against national
                                  figures declared in config.yaml

Run
    python main.py --stage extensions
    python extensions.py --only A1 A7

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

from common import assignment, calendar as C, national  # noqa: E402
from common.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
_CFG = load_config()
CACHE = _CFG.cache_dir
OUT = _CFG.results_dir("extensions")
DATA = ROOT.parent / "data"
SEED = int(_CFG.get("clustering.random_state", 42))
EXT = _CFG.get("extensions", {}) or {}
N_FOLDS = int(EXT.get("n_folds", 5))
N_PERM_INFO = int(EXT.get("information_permutations", 1000))
QH_SAMPLE = int(EXT.get("quarter_hourly_sample_pods", 1000))
DISP_MAX_DAYS = int(EXT.get("dispersion_max_days_per_cell", 2000))

FAMILY_LABEL = {
    "GSE": "GSE",
    "ARERA": "ARERA",
    "DD_in": "Data-driven (in-sample)",
    "DD_cv": "Data-driven (out-of-sample)",
    "ACT_in": "Activity-based (in-sample)",
    "ACT_cv": "Activity-based (out-of-sample)",
}
FAMILY_ORDER = list(FAMILY_LABEL)


# ============================================================================ helpers
def save_table(table: pd.DataFrame, name: str, bold: np.ndarray | None = None) -> None:
    #Lorenzo Giannuzzo: the writer of the figures stage, so that every table of the paper has
    # the same Excel format and is collected into paper_results/tables by the same step
    from figures import save_table as _save
    _save(table, name, OUT, bold)


def activity_name(code: object) -> str:
    try:
        from mapping_figures import activity_label
        return activity_label(code)
    except Exception:
        return str(code)


def national_name(key: object) -> str:
    """Readable name of a national profile, as the tables of the paper write it."""
    gse = {"PDMM": "GSE domestic, single rate", "PDMF": "GSE domestic, time bands",
           "PAUM": "GSE other uses, single rate", "PAUF": "GSE other uses, time bands"}
    if isinstance(key, tuple):
        cls, res = key
        r = str(res).lower()
        res_txt = ("non-resident" if "non" in r else "resident" if "resid" in r else "all")
        return f"ARERA {cls} kW, {res_txt}"
    return gse.get(str(key), str(key))


def arera_or_gse_name(label: object) -> str:
    """Readable name of a national profile as mapping.national_frame labels it."""
    s = str(label)
    if s.startswith("ARERA "):
        rest = s[len("ARERA "):]
        cls, _, res = rest.partition(" ")
        return national_name((cls, res))
    return national_name(s)


def fmt(x: float, nd: int = 3) -> str:
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pct(x: float, nd: int = 1) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"


def tv(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return np.nan
    return float(0.5 * np.abs(a / sa - b / sb).sum())


# ===================================================================== shared inputs
class Inputs:
    """Everything the analyses read, loaded once and only when first asked for."""

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def _get(self, key: str, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    # -- cache of the pipeline
    @property
    def days(self) -> pd.DataFrame:
        return self._get("days", lambda: pd.read_parquet(CACHE / "days.parquet"))

    @property
    def users(self) -> pd.DataFrame:
        return self._get("users", lambda: pd.read_parquet(CACHE / "users.parquet"))

    @property
    def groups(self) -> pd.DataFrame:
        def load():
            g = pd.read_parquet(CACHE / "groups.parquet")
            return g[~g["below_n_min"]] if "below_n_min" in g else g
        return self._get("groups", load)

    @property
    def shapes(self) -> np.ndarray:
        return self._get("shapes", lambda: np.load(CACHE / "shapes.npy", mmap_mode="r"))

    @property
    def user_vectors(self) -> pd.DataFrame:
        return self._get("uv", lambda: pd.read_parquet(CACHE / "user_vectors.parquet"))

    # -- the regulatory grid of Section 2.4
    @property
    def grid(self) -> dict:
        def build():
            from generation import DAYTYPE_ORDER, MONTH_LABELS, calendar_days_per_cell, period_of
            seasons = _CFG.get("preprocessing.seasons")
            grid = str(_CFG.get("generation.grid", "season")).lower()
            smap = {m: lab for lab, months in seasons.items() for m in months}
            d = self.days
            period = period_of(d["date"], smap, grid)
            periods = list(seasons) if grid == "season" else MONTH_LABELS
            present = pd.DataFrame({"p": period, "t": d["daytype"]}).drop_duplicates()
            present = set(zip(present["p"], present["t"]))
            cells = [(p, t) for p in periods for t in DAYTYPE_ORDER if (p, t) in present]
            cal = calendar_days_per_cell(seasons, grid=grid)
            return {"cells": cells, "period": period.to_numpy(),
                    "cal_days": np.array([int(cal.get(c, 0)) for c in cells]),
                    "labels": [f"{p}|{t}" for p, t in cells]}
        return self._get("grid", build)

    # -- reference calendar and national profiles of Section 2.5
    @property
    def hcal(self) -> pd.DataFrame:
        def build():
            from comparison import REFERENCE_YEAR
            smap = C.season_map_from_days(self.days)
            return C.hourly_index(C.build_calendar(REFERENCE_YEAR, smap))
        return self._get("hcal", build)

    @property
    def national(self) -> tuple[dict, dict]:
        def load():
            import comparison as cp
            tab, _ = national.load_arera(DATA, cp.PROVINCE, cache_dir=CACHE)
            gse_raw = national.load_gse(DATA, cache_dir=CACHE)
            return cp.arera_hourly_year(tab, self.hcal), cp.gse_hourly_year(gse_raw, self.hcal)
        return self._get("national", load)

    @property
    def assign(self) -> pd.DataFrame:
        import comparison as cp
        return self._get("assign", lambda: assignment.build(
            self.users, treatment=cp.GSE_TREATMENT, use_m_family=cp.USE_M_FAMILY))

    @property
    def observed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def build():
            import comparison as cp
            dictionary = np.load(CACHE / "dictionary.npy")
            pods, obs, seen, mode = cp.observed_hourly(self.days, dictionary, CACHE, self.hcal,
                                                       cp.REFERENCE_YEAR, self.user_vectors)
            print(f"    observed curves: {len(pods)} points [{mode}]")
            return pods, obs, seen
        return self._get("observed", build)

    # -- per-point sums over the grid, from which any catalog is rebuilt in one pass
    @property
    def pod_cell(self) -> dict:
        def build():
            d = self.days
            ok = d["has_shape"].to_numpy()
            grid = self.grid
            cell_of = {c: i for i, c in enumerate(grid["cells"])}
            ci = np.array([cell_of.get((p, t), -1) for p, t in
                           zip(grid["period"], d["daytype"])], dtype="int64")
            pods = np.sort(d.loc[ok, "pod"].unique())
            pod_ix = {p: i for i, p in enumerate(pods)}
            sel = ok & (ci >= 0)
            pi = d.loc[sel, "pod"].map(pod_ix).to_numpy()
            cj = ci[sel]
            idx = d.loc[sel, "shape_idx"].to_numpy()
            e = d.loc[sel, "energy"].to_numpy(dtype="float64")
            n_c = len(grid["cells"])
            sums = np.zeros((len(pods), n_c, 96))
            en = np.zeros((len(pods), n_c))
            nd = np.zeros((len(pods), n_c))
            order = np.argsort(idx)
            shp = self.shapes
            for lo in range(0, len(order), 200_000):
                s = order[lo:lo + 200_000]
                block = np.asarray(shp[idx[s]], dtype="float64")
                np.add.at(sums, (pi[s], cj[s]), block * e[s, None])
            np.add.at(en, (pi, cj), e)
            np.add.at(nd, (pi, cj), 1.0)
            return {"pods": pods, "index": pod_ix, "sums": sums, "energy": en, "ndays": nd}
        return self._get("pod_cell", build)


def catalog(inp: Inputs, members: dict[object, np.ndarray]) -> dict:
    """Eq. 6 for any grouping of the points, from the per-point sums.

    members maps a profile key to the indices of its points in pod_cell["pods"]. Returns the
    quarter-hourly curves and the calendar weights of every profile, built exactly as
    generation.py builds them: the energy-weighted aggregate curve of the members in each
    cell, and the mean daily energy of the cell times its calendar days, closed to one.
    """
    pc = inp.pod_cell
    cal = inp.grid["cal_days"].astype(float)
    keys = list(members)
    curves = np.zeros((len(keys), len(cal), 96))
    weights = np.zeros((len(keys), len(cal)))
    for k, key in enumerate(keys):
        ix = members[key]
        s = pc["sums"][ix].sum(axis=0)
        e = pc["energy"][ix].sum(axis=0)
        n = pc["ndays"][ix].sum(axis=0)
        tot = s.sum(axis=1, keepdims=True)
        curves[k] = np.divide(s, tot, out=np.zeros_like(s), where=tot > 0)
        mean_e = np.divide(e, n, out=np.zeros_like(e), where=n > 0)
        u = mean_e * cal
        weights[k] = u / u.sum() if u.sum() > 0 else 0.0
    return {"keys": keys, "curves": curves, "weights": weights}


def expand(inp: Inputs, cat: dict, per_hour: int = 1) -> np.ndarray:
    """Every profile of a catalog over the reference calendar, summing to one over the year.

    per_hour = 1 gives hourly values, per_hour = 4 quarter-hourly ones.
    """
    import comparison as cp
    hcal = inp.hcal
    labels = inp.grid["labels"]
    key = cp.hcal_cell_key(hcal, labels).to_numpy()
    days_of = hcal.drop_duplicates("date")
    day_key = cp.hcal_cell_key(days_of, labels).to_numpy()
    n_days = {c: int((day_key == c).sum()) for c in labels}
    out = np.zeros((len(cat["keys"]), len(hcal) * per_hour))
    hourly = cat["curves"].reshape(*cat["curves"].shape[:2], 24, 4)
    for j, c in enumerate(labels):
        m = np.repeat(key == c, per_hour)
        n = n_days[c]
        if n == 0:
            continue
        shape = (hourly[:, j].sum(axis=2) if per_hour == 1
                 else cat["curves"][:, j])
        out[:, m] = np.tile(shape, (1, n)) * (cat["weights"][:, j] / n)[:, None]
    return out


# ================================================================ A1 + A2 + A3 + A9 core
def ward_partition(X: np.ndarray, K: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    return fcluster(linkage(X, method="ward"), K, criterion="maxclust")


def user_X(inp: Inputs, pods: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The representation of Section 2.3, rebuilt from the cached frequencies and features."""
    from clustering import user_matrix
    uv = inp.user_vectors
    if pods is not None:
        uv = uv[uv["pod"].isin(pods)]
    fcols = [c for c in uv.columns if c.startswith("f_")]
    feat_cols = [c for c in uv.columns if c not in fcols and c != "pod"]
    cl = _CFG["clustering"]
    X, _, _ = user_matrix(uv[fcols].to_numpy(), uv[feat_cols].to_numpy(),
                          float(cl["zero_replacement"]), float(cl["scale_weight"]),
                          str(cl.get("block_normalisation", "variance")).lower())
    return uv["pod"].to_numpy(), X


def activity_classes(inp: Inputs) -> pd.DataFrame:
    """The activity class of every published point, pooled as in Section 2.6."""
    import mapping
    m = mapping.build_table(inp.users, inp.groups)
    return m[["pod", "activity_grouped", "group"]]


def build_families(inp: Inputs) -> dict:
    """The hourly references of the four catalogs, in-sample and out-of-sample."""
    pc = inp.pod_cell
    ix = pc["index"]
    groups = inp.groups[inp.groups["pod"].isin(list(ix))]
    act = activity_classes(inp)
    act = act[act["pod"].isin(list(ix))]
    K = int(groups["group"].nunique())

    def members_of(frame: pd.DataFrame, col: str) -> dict:
        return {k: np.array([ix[p] for p in x["pod"]]) for k, x in frame.groupby(col)}

    fam: dict[str, dict] = {}
    #Lorenzo Giannuzzo: in-sample, the data-driven catalog is rebuilt from the per-point sums
    # and checked against the one generation.py wrote, so that any later difference between
    # the families comes from the grouping and not from the construction
    dd_cat = catalog(inp, members_of(groups, "group"))
    try:
        cached = np.load(CACHE / "profiles.npy")
        if cached.shape == dd_cat["curves"].shape:
            print(f"    rebuilt data-driven curves differ from profiles.npy by at most "
                  f"{np.abs(cached - dd_cat['curves']).max():.2e}")
    except FileNotFoundError:
        pass
    fam["DD_in"] = {"curves": expand(inp, dd_cat), "keys": dd_cat["keys"],
                    "of_pod": dict(zip(groups["pod"], groups["group"]))}
    act_cat = catalog(inp, members_of(act, "activity_grouped"))
    fam["ACT_in"] = {"curves": expand(inp, act_cat), "keys": act_cat["keys"],
                     "of_pod": dict(zip(act["pod"], act["activity_grouped"]))}

    #Lorenzo Giannuzzo: out-of-sample, the points are split into folds; for each fold the users of
    # the other folds are clustered at the same K, their profiles are built from their own
    # days only, and the held-out users are assigned to the nearest centroid of the
    # representation, which is how a new user would be allocated to a published catalog
    pods_X, X = user_X(inp, groups["pod"].to_numpy())
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, N_FOLDS, len(pods_X))
    dd_cv = {"curves": {}, "of_pod": {}}
    act_cv = {"curves": {}, "of_pod": {}}
    act_of = dict(zip(act["pod"], act["activity_grouped"]))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        lab = ward_partition(X[tr], K)
        cent = np.stack([X[tr][lab == g].mean(axis=0) for g in np.unique(lab)])
        d2 = ((X[te][:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        te_lab = np.unique(lab)[d2.argmin(axis=1)]
        tr_frame = pd.DataFrame({"pod": pods_X[tr], "group": lab})
        cat_k = catalog(inp, members_of(tr_frame, "group"))
        curves_k = expand(inp, cat_k)
        pos = {g: i for i, g in enumerate(cat_k["keys"])}
        for p, g in zip(pods_X[te], te_lab):
            dd_cv["of_pod"][p] = (k, pos[g])
        dd_cv["curves"][k] = curves_k

        tr_act = act[act["pod"].isin(set(pods_X[tr]))]
        cat_a = catalog(inp, members_of(tr_act, "activity_grouped"))
        curves_a = expand(inp, cat_a)
        pos_a = {c: i for i, c in enumerate(cat_a["keys"])}
        for p in pods_X[te]:
            c = act_of.get(p)
            if c in pos_a:
                act_cv["of_pod"][p] = (k, pos_a[c])
        act_cv["curves"][k] = curves_a
        print(f"    fold {k + 1}/{N_FOLDS}: {int(tr.sum())} users to build, "
              f"{int(te.sum())} held out")
    fam["DD_cv"] = dd_cv
    fam["ACT_cv"] = act_cv
    return fam


def reference_of(fam: dict, name: str, pod: object) -> np.ndarray | None:
    f = fam[name]
    if name.endswith("_cv"):
        hit = f["of_pod"].get(pod)
        return None if hit is None else f["curves"][hit[0]][hit[1]]
    key = f["of_pod"].get(pod)
    if key is None:
        return None
    return f["curves"][f["keys"].index(key)]


def evaluate_families(inp: Inputs, fam: dict, prices: dict) -> dict:
    """User-month and portfolio error of every family, on two scopes.

    domestic       the user-months on which all six references are defined, which are the
                   domestic points carrying an ARERA profile
    non_domestic   the user-months of the points outside the ARERA tables on which the GSE,
                   data-driven and activity-based references are all defined
    """
    import comparison as cp
    hcal = inp.hcal
    hod = hcal["hour"].to_numpy()
    month = hcal["month"].to_numpy()
    bands = hcal["band"].to_numpy()
    arera, gse = inp.national
    pods, obs, seen = inp.observed
    a = inp.assign.set_index("pod")
    power = assignment.arera_class(inp.users.set_index("pod")["D_POTC"]).astype(object)
    rows = []
    n_year = len(hod)
    port: dict[tuple[str, str, int], list] = {}
    for i, pod in enumerate(pods):
        if pod not in a.index:
            continue
        rec = a.loc[pod]
        refs: dict[str, tuple[np.ndarray, bool]] = {}
        if rec["gse_column"] in gse:
            refs["GSE"] = (gse[rec["gse_column"]], cp.is_banded(f"GSE {rec['gse_column']}"))
        if rec["arera_applicable"] and (rec["arera_class"], rec["arera_residency"]) in arera:
            refs["ARERA"] = (arera[(rec["arera_class"], rec["arera_residency"])], False)
        for name in ("DD_in", "DD_cv", "ACT_in", "ACT_cv"):
            r = reference_of(fam, name, pod)
            if r is not None:
                refs[name] = (r, False)
        if "ARERA" in refs and len(refs) == 6:
            scope = "domestic"
        elif "ARERA" not in refs and all(k in refs for k in ("GSE", "DD_in", "DD_cv",
                                                             "ACT_in", "ACT_cv")):
            scope = "non_domestic"
        else:
            continue
        for m in range(1, 13):
            hours = (month == m) & np.repeat(seen[i], 24)
            if hours.sum() < cp.MIN_HOURS_PER_MONTH:
                continue
            o = obs[i][hours]
            if o.sum() <= 0:
                continue
            idxh = np.flatnonzero(hours)
            for name, (ref, banded) in refs.items():
                r = cp.scale_reference(ref[hours], o, bands[hours], banded)
                if r.sum() <= 0:
                    continue
                t = tv(o, r)
                row = {"pod": pod, "month": m, "scope": scope, "family": name,
                       "power_class": power.get(pod), "energy_kWh": float(o.sum()), "tv": t}
                for pk, pv in prices.items():
                    sgn, ab = cp.valuation(o, r, pv[hours])
                    row[f"value_abs_EUR_{pk}"] = ab
                rows.append(row)
                acc = port.setdefault((scope, name, m), [np.zeros(n_year), np.zeros(n_year)])
                np.add.at(acc[0], idxh, o)
                np.add.at(acc[1], idxh, r)
    um = pd.DataFrame(rows)
    #Lorenzo Giannuzzo: a user-month enters a scope only when every family of the scope could be
    # evaluated on it, so no family is credited with the easy months alone
    need = {"domestic": 6, "non_domestic": 5}
    cnt = um.groupby(["pod", "month", "scope"])["family"].transform("nunique")
    um = um[cnt.to_numpy() == um["scope"].map(need).to_numpy()]
    prow = []
    for (scope, name, m), (so, sr) in port.items():
        if so.sum() <= 0 or sr.sum() <= 0:
            continue
        do = np.bincount(hod, weights=so, minlength=24)
        dr = np.bincount(hod, weights=sr, minlength=24)
        row = {"scope": scope, "family": name, "month": m, "energy_kWh": float(so.sum()),
               "tv_hourly": tv(so, sr), "tv_mean_day": tv(do, dr)}
        for pk, pv in prices.items():
            sgn, ab = cp.valuation(so, sr, pv)
            row[f"portfolio_value_signed_EUR_{pk}"] = sgn
            row[f"portfolio_value_abs_EUR_{pk}"] = ab
        prow.append(row)
    return {"user_month": um, "portfolio": pd.DataFrame(prow)}


def tables_families(res: dict, prices: dict) -> None:
    um, pf = res["user_month"], res["portfolio"]
    um.to_csv(OUT / "a1_user_month.csv", index=False)
    pf.to_csv(OUT / "a1_portfolio_month.csv", index=False)
    scopes = {"domestic": ("GSE", "ARERA", "DD_in", "DD_cv", "ACT_in", "ACT_cv"),
              "non_domestic": ("GSE", "DD_in", "DD_cv", "ACT_in", "ACT_cv")}
    scope_txt = {"domestic": "Domestic points carrying an ARERA profile",
                 "non_domestic": "Points outside the ARERA tables"}
    rows = []
    for scope, fams in scopes.items():
        for f in fams:
            x = um[(um["scope"] == scope) & (um["family"] == f)]
            p = pf[(pf["scope"] == scope) & (pf["family"] == f)]
            if not len(x):
                continue
            w = p["energy_kWh"]
            rows.append({
                "Population": scope_txt[scope],
                "Family of profiles": FAMILY_LABEL[f],
                "User-months [-]": len(x),
                "Median TV, user-month [-]": fmt(x["tv"].median()),
                "Interquartile range, user-month [-]":
                    f"{fmt(x['tv'].quantile(.25))} to {fmt(x['tv'].quantile(.75))}",
                "Misallocated share, user-months [%]":
                    pct((x["tv"] * x["energy_kWh"]).sum() / x["energy_kWh"].sum()),
                "Misallocated share, portfolio, hourly [%]":
                    pct(np.average(p["tv_hourly"], weights=w)) if len(p) else "",
                "Misallocated share, portfolio, mean day [%]":
                    pct(np.average(p["tv_mean_day"], weights=w)) if len(p) else "",
            })
    save_table(pd.DataFrame(rows), "table_a1_families_of_profiles")

    #Lorenzo Giannuzzo: A2 isolates the in-sample and out-of-sample readings of the two catalogs
    # built from the data, on the domestic scope where all families are defined
    rows = []
    for scope in ("domestic", "non_domestic"):
        for base in ("DD", "ACT"):
            r = {"Population": scope_txt[scope],
                 "Catalog": "Data-driven" if base == "DD" else "Activity-based"}
            for kind in ("in", "cv"):
                f = f"{base}_{kind}"
                x = um[(um["scope"] == scope) & (um["family"] == f)]
                p = pf[(pf["scope"] == scope) & (pf["family"] == f)]
                tag = "in-sample" if kind == "in" else "out-of-sample"
                r[f"Median TV, {tag} [-]"] = fmt(x["tv"].median()) if len(x) else ""
                r[f"Portfolio misallocation, {tag} [%]"] = (
                    pct(np.average(p["tv_hourly"], weights=p["energy_kWh"])) if len(p) else "")
            rows.append(r)
    save_table(pd.DataFrame(rows), "table_a2_out_of_sample")

    #Lorenzo Giannuzzo: the share misallocated by contractual power class, as in Fig. 16, with
    # the activity-based catalog added
    x = um[um["scope"] == "domestic"]
    if len(x):
        rows = []
        for cls, y in x.groupby("power_class", observed=True):
            r = {"Contractual power class [kW]": str(cls),
                 "Points of delivery [-]": int(y["pod"].nunique())}
            for f in ("GSE", "ARERA", "DD_in", "ACT_in", "DD_cv", "ACT_cv"):
                z = y[y["family"] == f]
                r[f"{FAMILY_LABEL[f]} [%]"] = pct((z["tv"] * z["energy_kWh"]).sum()
                                                   / z["energy_kWh"].sum()) if len(z) else ""
            rows.append(r)
        save_table(pd.DataFrame(rows), "table_a1_power_classes")

    #Lorenzo Giannuzzo: A3, only when at least one price series is configured
    if prices:
        rows = []
        for scope, fams in scopes.items():
            for f in fams:
                x = um[(um["scope"] == scope) & (um["family"] == f)]
                p = pf[(pf["scope"] == scope) & (pf["family"] == f)]
                if not len(x):
                    continue
                r = {"Population": scope_txt[scope], "Family of profiles": FAMILY_LABEL[f],
                     "Misallocated energy, user-months [MWh]":
                         fmt((x["tv"] * x["energy_kWh"]).sum() / 1000, 1)}
                for pk in prices:
                    #Lorenzo Giannuzzo: the misallocated energy of Eq. 13 is half the absolute
                    # discrepancy, so its value is half the absolute valuation, which counts
                    # every misplaced kilowatt-hour once where it is missing and once where
                    # it is in excess
                    r[f"Value of the misallocated energy, user-months, {pk} [EUR]"] = fmt(
                        x[f"value_abs_EUR_{pk}"].sum() / 2, 0)
                    r[f"Value of the misallocated energy, portfolio, {pk} [EUR]"] = fmt(
                        p[f"portfolio_value_abs_EUR_{pk}"].sum() / 2, 0)
                    r[f"Signed cost, portfolio, {pk} [EUR]"] = fmt(
                        p[f"portfolio_value_signed_EUR_{pk}"].sum(), 0)
                rows.append(r)
        save_table(pd.DataFrame(rows), "table_a3_valuation")


def run_A1_A2_A3(inp: Inputs) -> None:
    import comparison as cp
    prices = {}
    for key, label in (("price_file", "day-ahead"), ("imbalance_price_file", "imbalance")):
        series, note = cp.load_price_series(key, inp.hcal)
        print(f"    prices [{label}]: {note}")
        if series is not None:
            prices[label] = series
    fam = build_families(inp)
    res = evaluate_families(inp, fam, prices)
    tables_families(res, prices)
    if not prices:
        print("    A3 skipped: no price series configured (comparison.price_file)")


# ================================================================== A4 dispersion, Eq. 7
def run_A4(inp: Inputs) -> None:
    """Dispersion of the members around their profile, against the national profiles."""
    import comparison as cp
    arera, gse = inp.national
    hcal = inp.hcal
    labels = inp.grid["labels"]
    key = cp.hcal_cell_key(hcal, labels).to_numpy()
    hod = hcal["hour"].to_numpy()
    groups = inp.groups
    pc = inp.pod_cell
    ix = pc["index"]
    members = {g: np.array([ix[p] for p in x["pod"] if p in ix]) for g, x in groups.groupby("group")}
    cat = catalog(inp, members)
    dd_h = cat["curves"].reshape(*cat["curves"].shape[:2], 24, 4).sum(axis=3)

    #Lorenzo Giannuzzo: the average day of a national profile in each cell of the grid, as a
    # distribution over the hours. Time-band profiles are left out, since their values are
    # normalised within bands and have no daily shape of their own.
    nat = {k: v for k, v in arera.items()}
    nat.update({k: v for k, v in gse.items() if not cp.is_banded(f"GSE {k}")})
    nat_cell = {}
    for name, v in nat.items():
        shapes = np.zeros((len(labels), 24))
        for j, c in enumerate(labels):
            m = key == c
            if m.any():
                prof = np.bincount(hod[m], weights=v[m], minlength=24)
                shapes[j] = prof / prof.sum() if prof.sum() > 0 else 0
        nat_cell[name] = shapes

    def nrmsd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sqrt(((a - b) ** 2).mean(axis=-1)) / (1.0 / 24)

    cell_of = {c: i for i, c in enumerate(inp.grid["cells"])}
    all_days = inp.days.assign(cell=[cell_of.get((p, t), -1) for p, t in
                                     zip(inp.grid["period"], inp.days["daytype"])])
    d = all_days[all_days["has_shape"]].merge(groups[["pod", "group"]], on="pod")
    rng = np.random.default_rng(SEED)
    eq7 = {name: np.zeros((len(cat["keys"]), len(labels))) for name in nat_cell}
    rows = []
    for gi, g in enumerate(cat["keys"]):
        pooled_member, pooled_w = [], []
        for j in range(len(labels)):
            x = d[(d["group"] == g) & (d["cell"] == j)]
            for name, shp in nat_cell.items():
                eq7[name][gi, j] = nrmsd(shp[j], dd_h[gi, j])
            if not len(x):
                continue
            idx = x["shape_idx"].to_numpy()
            if len(idx) > DISP_MAX_DAYS:
                idx = rng.choice(idx, DISP_MAX_DAYS, replace=False)
            mem = np.asarray(inp.shapes[np.sort(idx)], dtype="float64").reshape(-1, 24, 4).sum(axis=2)
            val = nrmsd(mem, dd_h[gi, j][None, :])
            pooled_member.append(val)
            pooled_w.append(np.full(len(val), len(x) / len(val)))
        pm = np.concatenate(pooled_member)
        pw = np.concatenate(pooled_w)
        order = np.argsort(pm)
        cw = np.cumsum(pw[order]) / pw.sum()
        med = float(pm[order][np.searchsorted(cw, 0.5)])
        p90 = float(pm[order][np.searchsorted(cw, 0.9)])
        #Lorenzo Giannuzzo: the nearest national profile by the median of Eq. 7 over the cells,
        # and the share of member days lying farther from the profile than that national curve
        med_nat = {name: float(np.median(eq7[name][gi])) for name in nat_cell}
        best = min(med_nat, key=med_nat.get)
        farther = float(np.sum(pw[pm > med_nat[best]]) / pw.sum())
        rows.append({"Data-driven profile": f"DD-SLP {g}",
                     "Median nRMSD of the member days [-]": fmt(med, 2),
                     "90th percentile nRMSD of the member days [-]": fmt(p90, 2),
                     "Nearest single-rate national profile, Eq. 7": national_name(best),
                     "Median nRMSD of that national profile [-]": fmt(med_nat[best], 2),
                     "Member days farther than that national profile [%]": pct(farther)})
    save_table(pd.DataFrame(rows), "table_a4_dispersion_against_national")

    #Lorenzo Giannuzzo: Eq. 7 for every pair, as the median over the cells of the grid, in the
    # layout of Table 2
    t = pd.DataFrame({f"DD-SLP {g}": [float(np.median(eq7[n][gi])) for n in nat_cell]
                      for gi, g in enumerate(cat["keys"])})
    t.insert(0, "National profile", [national_name(n) for n in nat_cell])
    body = t.iloc[:, 1:].to_numpy()
    t["Nearest"] = [f"DD-SLP {cat['keys'][int(np.argmin(r))]}" for r in body]
    order = np.argsort(body.min(axis=1))
    t = t.iloc[order].reset_index(drop=True)
    body = t.iloc[:, 1:-1].to_numpy()
    bold = np.zeros(t.shape, dtype=bool)
    bold[np.arange(len(t)), 1 + body.argmin(axis=1)] = True
    for c in t.columns[1:-1]:
        t[c] = t[c].map(lambda v: fmt(v, 2))
    save_table(t, "table_a4_eq7_nrmsd", bold)


# ============================================================ A5 partition uncertainty
def run_A5(inp: Inputs) -> None:
    import mapping
    groups = inp.groups
    K = int(groups["group"].nunique())
    base = mapping.build_table(inp.users, groups)
    n_min = base.attrs["n_min_classes"]
    pods_X, X = user_X(inp, groups["pod"].to_numpy())
    rng = np.random.default_rng(SEED + 1)
    n_sub = int(_CFG.get("clustering.k_n_boot", 20))
    frac = float(_CFG.get("clustering.k_frac", 0.8))

    def metrics_on(gframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        m = mapping.build_table(inp.users, gframe, n_min=n_min)
        m1 = mapping.multiplicity(m, "pod", with_null=True).set_index("activity")
        nat = mapping.national_frame(m, inp.users)
        beh = nat.groupby("national").apply(
            lambda x: mapping.effective_number(x.groupby("profile").size().to_numpy()),
            include_groups=False)
        n_nat = nat.groupby("national").size()
        return m1, beh[n_nat >= 50]

    base_m1, base_beh = metrics_on(groups.assign(below_n_min=False))
    m1_runs, beh_runs = [], []
    for b in range(n_sub):
        sel = rng.random(len(pods_X)) < frac
        lab = ward_partition(X[sel], K)
        sizes = pd.Series(lab).value_counts()
        small = set(sizes[sizes < mapping.n_min_groups(K)].index)
        g = pd.DataFrame({"pod": pods_X[sel], "group": lab})
        g["below_n_min"] = g["group"].isin(small)
        m1, beh = metrics_on(g)
        m1_runs.append(m1[["M1_effective", "coherent_at_5pct"]])
        beh_runs.append(beh)
        print(f"    partition {b + 1}/{n_sub}")
    rows = []
    for cls in base_m1.index:
        vals = np.array([r.loc[cls, "M1_effective"] for r in m1_runs if cls in r.index])
        coh = np.array([bool(r.loc[cls, "coherent_at_5pct"]) for r in m1_runs if cls in r.index])
        rows.append({"Activity class": activity_name(cls),
                     "Points of delivery [-]": int(base_m1.loc[cls, "n_pod"]),
                     "Class multiplicity, full partition [-]": fmt(base_m1.loc[cls, "M1_effective"], 1),
                     "Median over subsample partitions [-]": fmt(np.median(vals), 1) if len(vals) else "",
                     "5th to 95th percentile [-]":
                         f"{fmt(np.percentile(vals, 5), 1)} to {fmt(np.percentile(vals, 95), 1)}"
                         if len(vals) else "",
                     "Coherent, full partition": "yes" if base_m1.loc[cls, "coherent_at_5pct"] else "no",
                     "Partitions in which coherent [%]": pct(coh.mean(), 0) if len(coh) else ""})
    save_table(pd.DataFrame(rows), "table_a5_uncertainty_class_multiplicity")
    rows = []
    for nat in base_beh.index:
        vals = np.array([r.get(nat, np.nan) for r in beh_runs], dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append({"National profile": arera_or_gse_name(nat),
                     "Behaviors delivered, full partition [-]": fmt(base_beh[nat], 1),
                     "Median over subsample partitions [-]": fmt(np.median(vals), 1) if len(vals) else "",
                     "5th to 95th percentile [-]":
                         f"{fmt(np.percentile(vals, 5), 1)} to {fmt(np.percentile(vals, 95), 1)}"
                         if len(vals) else ""})
    save_table(pd.DataFrame(rows), "table_a5_uncertainty_behaviors_delivered")


# ===================================================================== A6 information
def _mi(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> tuple[float, float, float, float]:
    ct = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    n = ct.sum()
    p = ct / n
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (px[:, None] * py[None, :])[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    expct = px[:, None] * py[None, :] * n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((ct - expct) ** 2 / np.where(expct > 0, expct, np.nan)))
    k = min((px > 0).sum(), (py > 0).sum()) - 1
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 else np.nan
    return mi, hx, hy, v


def run_A6(inp: Inputs) -> None:
    import mapping
    m = mapping.build_table(inp.users, inp.groups)
    a = inp.assign.set_index("pod")
    power = assignment.arera_class(inp.users.set_index("pod")["D_POTC"]).astype(object)
    m["gse_category"] = m["pod"].map(a["gse_category"])
    m["power_class"] = m["pod"].map(power)
    m["arera_key"] = [f"{c} {r}" if ok else None for c, r, ok in
                      zip(m["pod"].map(a["arera_class"]), m["pod"].map(a["arera_residency"]),
                          m["pod"].map(a["arera_applicable"]))]
    rng = np.random.default_rng(SEED + 2)
    variables = [("Activity class (Section 2.6)", "activity_grouped", "all points"),
                 ("Tariff category, domestic or other uses", "gse_category", "all points"),
                 ("Contractual power class", "power_class", "all points"),
                 ("ARERA power class and residency", "arera_key", "domestic points with an ARERA profile")]
    rows = []
    for label, col, pop in variables:
        x = m[m[col].notna()]
        if not len(x):
            continue
        xc, nx = pd.factorize(x[col].astype(str))[0], x[col].astype(str).nunique()
        yc, ny = pd.factorize(x["profile"])[0], x["profile"].nunique()
        mi, hx, hy, v = _mi(xc, yc, nx, ny)
        null = np.empty(N_PERM_INFO)
        for b in range(N_PERM_INFO):
            null[b] = _mi(rng.permutation(xc), yc, nx, ny)[0]
        p = (np.sum(null >= mi) + 1) / (N_PERM_INFO + 1)
        rows.append({"Variable": label, "Population": pop,
                     "Points of delivery [-]": len(x), "Categories [-]": nx,
                     "Mutual information [nats]": fmt(mi, 3),
                     "Share of the profile entropy explained [%]": pct(mi / hy if hy > 0 else np.nan),
                     "Normalized mutual information [-]": fmt(mi / np.sqrt(hx * hy) if hx * hy > 0 else np.nan, 3),
                     "Cramér's V [-]": fmt(v, 3),
                     "Mutual information of random labels, 95th percentile [nats]": fmt(np.percentile(null, 95), 4),
                     "Permutation p-value [-]": f"{p:.4f}"})
    save_table(pd.DataFrame(rows), "table_a6_information_on_the_profile")


# ================================================================= A7 multiple testing
def run_A7(inp: Inputs) -> None:
    import mapping
    path = _CFG.results_dir("mapping", "multiplicity") / "m1_multiplicity_pod.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the mapping stage first")
    m1 = pd.read_csv(path)
    n = mapping.N_PERMUTATIONS
    #Lorenzo Giannuzzo: the permutation p-value of each class, with the +1 correction that
    # keeps it above zero, then the Benjamini-Hochberg adjustment over the classes tested
    p = (m1["null_quantile"] * n + 1) / (n + 1)
    order = np.argsort(p.to_numpy())
    ranked = p.to_numpy()[order] * len(p) / (np.arange(len(p)) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    q = np.empty(len(p))
    q[order] = q_sorted
    t = pd.DataFrame({"Activity class": [activity_name(c) for c in m1["activity"]],
                      "Points of delivery [-]": m1["n_pod"],
                      "Class multiplicity [-]": m1["M1_effective"].map(lambda v: fmt(v, 1)),
                      "Permutation p-value [-]": p.map(lambda v: f"{v:.4f}"),
                      "Benjamini-Hochberg q-value [-]": [f"{v:.4f}" for v in q],
                      "Coherent at 5%": np.where(p < 0.05, "yes", "no"),
                      "Coherent at 5% after correction": np.where(q < 0.05, "yes", "no")})
    save_table(t, "table_a7_multiple_testing")


# ================================================================== A8 prosumers
def run_A8(inp: Inputs) -> None:
    import mapping
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score
    groups = inp.groups
    users = inp.users
    K = int(groups["group"].nunique())
    pv = set(users.loc[users["prosumer"].fillna(False).astype(bool), "pod"])
    keep = groups[~groups["pod"].isin(pv)]
    pods_X, X = user_X(inp, keep["pod"].to_numpy())
    lab = ward_partition(X, K)
    new = pd.DataFrame({"pod": pods_X, "new": lab}).merge(groups[["pod", "group"]], on="pod")
    ari = adjusted_rand_score(new["group"], new["new"])
    ct = pd.crosstab(new["group"], new["new"])
    r, c = linear_sum_assignment(-ct.to_numpy())
    match = {ct.columns[j]: ct.index[i] for i, j in zip(r, c)}
    new["matched"] = new["new"].map(match)
    dom = users.set_index("pod")["ateco_l1"].astype(str).str.startswith(("DO",))
    rows = []
    for g in sorted(groups["group"].unique()):
        b = groups[groups["group"] == g]
        n_ = new[new["matched"] == g]
        rows.append({"Data-driven profile": f"DD-SLP {g}",
                     "Points of delivery, all [-]": len(b),
                     "Prosumers in the profile [%]": pct(b["pod"].isin(pv).mean()),
                     "Points of delivery, prosumers excluded [-]": len(n_),
                     "Domestic share, all [%]": pct(b["pod"].map(dom).mean(), 0),
                     "Domestic share, prosumers excluded [%]": pct(n_["pod"].map(dom).mean(), 0)
                     if len(n_) else "",
                     "Points kept in the same profile [%]":
                         pct((n_["group"] == g).sum() / b["pod"].isin(keep["pod"]).sum())})
    t = pd.DataFrame(rows)
    save_table(t, "table_a8_prosumers_excluded_profiles")

    gnew = new[["pod"]].assign(group=new["matched"], below_n_min=False)
    base_m = mapping.build_table(users, groups)
    m_new = mapping.build_table(users, gnew, n_min=base_m.attrs["n_min_classes"])
    m1_b = mapping.multiplicity(base_m, "pod", with_null=True)
    m1_n = mapping.multiplicity(m_new, "pod", with_null=True)
    s = pd.DataFrame([
        {"Quantity": "Adjusted Rand index against the adopted partition [-]",
         "All points": "1.000", "Prosumers excluded": fmt(ari)},
        {"Quantity": "Prosumers removed [-]", "All points": "0",
         "Prosumers excluded": str(int(groups["pod"].isin(pv).sum()))},
        {"Quantity": "Median class multiplicity [-]",
         "All points": fmt(m1_b["M1_effective"].median(), 2),
         "Prosumers excluded": fmt(m1_n["M1_effective"].median(), 2)},
        {"Quantity": "Classes more coherent than a random label [-]",
         "All points": str(int(m1_b["coherent_at_5pct"].sum())),
         "Prosumers excluded": str(int(m1_n["coherent_at_5pct"].sum()))},
    ])
    save_table(s, "table_a8_prosumers_excluded_summary")


# ============================================================ A9 quarter-hourly error
def run_A9(inp: Inputs) -> None:
    """The user-month error at quarter-hourly against hourly resolution, on a sample."""
    import comparison as cp
    arera, gse = inp.national
    hcal = inp.hcal
    groups = inp.groups
    a = inp.assign.set_index("pod")
    pc = inp.pod_cell
    ix = pc["index"]
    members = {g: np.array([ix[p] for p in x["pod"] if p in ix]) for g, x in groups.groupby("group")}
    cat = catalog(inp, members)
    dd_q = expand(inp, cat, per_hour=4)
    grp = dict(zip(groups["pod"], groups["group"]))
    elig = [p for p in groups["pod"] if p in a.index and bool(a.loc[p, "arera_applicable"])
            and (a.loc[p, "arera_class"], a.loc[p, "arera_residency"]) in arera
            and a.loc[p, "gse_column"] in gse]
    rng = np.random.default_rng(SEED + 3)
    sample = set(rng.choice(elig, size=min(QH_SAMPLE, len(elig)), replace=False))
    d = inp.days[(inp.days["pod"].isin(sample)) & inp.days["has_shape"]
                 & (inp.days["date"].dt.year == cp.REFERENCE_YEAR)]
    cal_days = hcal.drop_duplicates("date")["date"].reset_index(drop=True)
    slot = {(t.month, t.day): i for i, t in enumerate(cal_days)}
    month_q = np.repeat(hcal["month"].to_numpy(), 4)
    band_q = np.repeat(hcal["band"].to_numpy(), 4)
    hour_q = np.repeat(hcal["hour"].to_numpy(), 4)
    rows = []
    port: dict[tuple[str, str], list] = {}
    for pod, x in d.groupby("pod"):
        obs = np.zeros(len(cal_days) * 96)
        seen = np.zeros(len(cal_days), dtype=bool)
        sl = np.array([slot.get((t.month, t.day), -1) for t in x["date"]])
        ok = sl >= 0
        idx = x["shape_idx"].to_numpy()[ok]
        o_idx = np.argsort(idx)
        shp = np.asarray(inp.shapes[idx[o_idx]], dtype="float64")
        e = x["energy"].to_numpy()[ok][o_idx]
        for s_, row, en in zip(sl[ok][o_idx], shp, e):
            obs[s_ * 96:(s_ + 1) * 96] = row * en
            seen[s_] = True
        rec = a.loc[pod]
        refs = {"GSE": np.repeat(gse[rec["gse_column"]], 4) / 4,
                "ARERA": np.repeat(arera[(rec["arera_class"], rec["arera_residency"])], 4) / 4,
                "DD_in": dd_q[cat["keys"].index(grp[pod])]}
        seen_q = np.repeat(seen, 96)
        for m in range(1, 13):
            q = (month_q == m) & seen_q
            if q.sum() < cp.MIN_HOURS_PER_MONTH * 4:
                continue
            o = obs[q]
            if o.sum() <= 0:
                continue
            for name, ref in refs.items():
                banded = name == "GSE" and cp.is_banded(f"GSE {rec['gse_column']}")
                r = cp.scale_reference(ref[q], o, band_q[q], banded)
                oh = o.reshape(-1, 4).sum(axis=1)
                rh = r.reshape(-1, 4).sum(axis=1)
                rows.append({"family": name, "tv_hour": tv(oh, rh), "tv_quarter": tv(o, r),
                             "energy": float(o.sum())})
                acc = port.setdefault((name, m), [np.zeros(len(month_q)), np.zeros(len(month_q))])
                acc[0][q] += o
                acc[1][q] += r
    x = pd.DataFrame(rows)
    out = []
    for name in ("GSE", "ARERA", "DD_in"):
        y = x[x["family"] == name]
        if not len(y):
            continue
        ph, pq, w = [], [], []
        for (f, m), (so, sr) in port.items():
            if f != name or so.sum() <= 0:
                continue
            ph.append(tv(so.reshape(-1, 4).sum(1), sr.reshape(-1, 4).sum(1)))
            pq.append(tv(so, sr))
            w.append(so.sum())
        out.append({"Family of profiles": FAMILY_LABEL[name],
                    "User-months in the sample [-]": len(y),
                    "Median TV, hourly [-]": fmt(y["tv_hour"].median()),
                    "Median TV, quarter-hourly [-]": fmt(y["tv_quarter"].median()),
                    "Increase of the median [percentage points]":
                        fmt(100 * (y["tv_quarter"].median() - y["tv_hour"].median()), 1),
                    "Portfolio misallocation, hourly [%]": pct(np.average(ph, weights=w)),
                    "Portfolio misallocation, quarter-hourly [%]": pct(np.average(pq, weights=w))})
    t = pd.DataFrame(out)
    t.attrs["sample"] = len(sample)
    save_table(t, "table_a9_quarter_hourly_resolution")
    print(f"    sample of {len(sample)} domestic points; national profiles spread "
          f"uniformly within each hour")


# ==================================================================== A10 time bands
def run_A10(inp: Inputs) -> None:
    import comparison as cp
    arera, gse = inp.national
    hcal = inp.hcal
    bands = hcal["band"].to_numpy()
    groups = inp.groups
    pc = inp.pod_cell
    ix = pc["index"]
    members = {g: np.array([ix[p] for p in x["pod"] if p in ix]) for g, x in groups.groupby("group")}
    cat = catalog(inp, members)
    dd = expand(inp, cat)

    def shares(v: np.ndarray) -> np.ndarray:
        s = np.array([v[bands == b].sum() for b in (1, 2, 3)])
        return s / s.sum() if s.sum() > 0 else s

    rows = []
    vecs = {}
    for gi, g in enumerate(cat["keys"]):
        s = shares(dd[gi])
        vecs[f"DD-SLP {g}"] = dd[gi]
        rows.append({"Profile": f"DD-SLP {g}", "Type": "Data-driven",
                     "F1 [%]": pct(s[0]), "F2 [%]": pct(s[1]), "F3 [%]": pct(s[2])})
    for k, v in list(arera.items()) + [(k, v) for k, v in gse.items()
                                       if not cp.is_banded(f"GSE {k}")]:
        s = shares(v)
        rows.append({"Profile": national_name(k),
                     "Type": "ARERA" if isinstance(k, tuple) else "GSE",
                     "F1 [%]": pct(s[0]), "F2 [%]": pct(s[1]), "F3 [%]": pct(s[2])})
    tilp = EXT.get("tilp_file")
    if tilp:
        p = (ROOT / str(tilp)).resolve()
        if p.exists():
            tt = pd.read_csv(p)
            for r in tt.itertuples():
                tot = r.F1 + r.F2 + r.F3
                rows.append({"Profile": str(r.profile), "Type": "ARERA conventional load profiling",
                             "F1 [%]": pct(r.F1 / tot), "F2 [%]": pct(r.F2 / tot),
                             "F3 [%]": pct(r.F3 / tot)})
        else:
            print(f"    ! extensions.tilp_file not found: {p}")
    save_table(pd.DataFrame(rows), "table_a10_time_band_shares")

    #Lorenzo Giannuzzo: how much of the difference between two data-driven profiles survives
    # once each is reduced to its three band shares
    names = list(vecs)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_, b_ = vecs[names[i]], vecs[names[j]]
            rows.append({"Pair of data-driven profiles": f"{names[i]} and {names[j]}",
                         "TV between the hourly profiles [-]": tv(a_, b_),
                         "TV between the band shares [-]": tv(shares(a_), shares(b_))})
    t = pd.DataFrame(rows).sort_values("TV between the hourly profiles [-]", ascending=False)
    t["Difference retained by the bands [%]"] = (
        100 * t["TV between the band shares [-]"] / t["TV between the hourly profiles [-]"]
    ).map(lambda v: fmt(v, 0))
    for c in ("TV between the hourly profiles [-]", "TV between the band shares [-]"):
        t[c] = t[c].map(fmt)
    save_table(t.reset_index(drop=True), "table_a10_band_discrimination")


# ============================================================ A11 representativeness
def run_A11(inp: Inputs) -> None:
    users = inp.users
    groups = inp.groups
    u = users[users["pod"].isin(groups["pod"])].copy()
    a = inp.assign.set_index("pod")
    u["cat"] = u["pod"].map(a["gse_category"])
    u["res"] = u["pod"].map(a["arera_residency"])
    u["cls"] = assignment.arera_class(u["D_POTC"]).astype(object)
    dom = u[u["cat"] == "domestic"]
    ref = EXT.get("national_reference", {}) or {}
    rows = [("share_domestic", "Domestic points among all points [%]", pct((u["cat"] == "domestic").mean())),
            ("share_resident", "Resident among domestic points [%]",
             pct(dom["res"].astype(str).str.lower().str.startswith("resid").mean())),
            ("share_prosumer", "Prosumers among all points [%]",
             pct(u["prosumer"].fillna(False).astype(bool).mean()))]
    for c in assignment.ARERA_CLASSES:
        rows.append((f"share_power_{c}", f"Domestic points in the {c} kW class [%]",
                     pct((dom["cls"] == c).mean())))
    for key, lab, sel in (("median_kwh_resident", "Median annual consumption, resident domestic [kWh]",
                           dom["res"].astype(str).str.lower().str.startswith("resid")),
                          ("median_kwh_nonresident", "Median annual consumption, non-resident domestic [kWh]",
                           dom["res"].astype(str).str.lower().str.startswith("non"))):
        rows.append((key, lab, fmt(float(dom.loc[sel, "E"].median()), 0) if sel.any() else ""))
    rows.append(("median_kwh_other", "Median annual consumption, other uses [kWh]",
                 fmt(float(u.loc[u["cat"] == "other", "E"].median()), 0)))
    t = pd.DataFrame([{"Indicator": lab, "Dataset": val,
                       "National reference": "" if ref.get(k) is None else str(ref.get(k)),
                       "Source of the reference": str(ref.get("source", "")) if ref.get(k) is not None else ""}
                      for k, lab, val in rows])
    save_table(t, "table_a11_representativeness")
    if not ref:
        print("    national reference not declared: fill extensions.national_reference in config.yaml")


# ============================================================================== main
ANALYSES = {
    "A1": ("activity-based catalog, out-of-sample validation and valuation", run_A1_A2_A3),
    "A4": ("dispersion and Eq. 7", run_A4),
    "A5": ("uncertainty of the partition", run_A5),
    "A6": ("information on the profile", run_A6),
    "A7": ("multiple testing", run_A7),
    "A8": ("prosumers excluded", run_A8),
    "A9": ("quarter-hourly resolution", run_A9),
    "A10": ("time-band shares", run_A10),
    "A11": ("representativeness", run_A11),
}


def main(only: list[str] | None = None) -> None:
    t0 = time.time()
    print(f"\n{'='*78}\n  EXTENSIONS, tables completing Section 3\n{'='*78}")
    inp = Inputs()
    todo = only or list(ANALYSES)
    failed = []
    for key in todo:
        label, fn = ANALYSES[key]
        print(f"\n  [{key}] {label}")
        t1 = time.time()
        try:
            fn(inp)
            print(f"    done ({time.time() - t1:.0f}s)")
        except Exception as exc:
            #Lorenzo Giannuzzo: one analysis that cannot run must not cost the others their tables
            failed.append(key)
            print(f"    ! {key} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n  tables under {OUT / 'tables'}   ({time.time()-t0:.0f}s)")
    if failed:
        print(f"  ! failed: {', '.join(failed)}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="extensions of Section 3")
    ap.add_argument("--only", nargs="*", choices=list(ANALYSES), help="run these analyses only")
    main(ap.parse_args().only)
