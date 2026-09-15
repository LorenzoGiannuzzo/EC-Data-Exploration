"""Numerical results, every number the paper quotes, in one table.

    python main.py --stage numbers        after mapping
    python numerical_results.py

Each stage writes its own tables in its own folder, and the paper quotes numbers from
eleven of them. This stage reads those tables and the cache, computes the handful of
quantities no stage computes on its own (the positioning counts, the uncovered profiles,
the per-cell stability of the match, the Sunday ratio, the structural zeros), and writes
one long table with the section and the figure each number belongs to. The text of the
paper is written against this file and against nothing else.

Output
    paper_results/numerical_results.csv
        section, figure, quantity, subject, value, unit, source, definition

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
from common.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
_CFG = load_config()
CACHE = _CFG.cache_dir
RES_PRE = _CFG.results_dir("preprocessing")
RES_CLU = _CFG.results_dir("clustering")
RES_GEN = _CFG.results_dir("generation")
RES_CMP = _CFG.results_dir("comparison")
RES_MAP = _CFG.results_dir("mapping")
RES_M1 = _CFG.results_dir("mapping", "multiplicity")
RES_M2 = _CFG.results_dir("mapping", "aggregation")
RES_M3 = _CFG.results_dir("mapping", "coverage")
OUT = RES_CMP.parent / "numerical_results.csv"
DOMESTIC_PREFIX = ("DO",)


class Table:
    """Accumulates rows; a missing input becomes a row that says so, never a silent gap."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section: str, figure: str, quantity: str, value, unit: str = "-",
            subject: str = "", source: str = "", definition: str = "") -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and np.isfinite(value):
            value = round(value, 6)
        self.rows.append({"section": section, "figure": figure, "quantity": quantity,
                          "subject": subject, "value": value, "unit": unit,
                          "source": source, "definition": definition})

    def missing(self, section: str, what: str, path: Path) -> None:
        self.add(section, "", what, "not available", "", "",
                 str(path.name), f"input not found: {path}")


def _read(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def _effective(counts) -> float:
    c = np.asarray(counts, float)
    c = c[c > 0]
    if c.sum() <= 0:
        return float("nan")
    p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


# ------------------------------------------------------------------ Section 2.2 / 3.1
def preprocessing_numbers(t: Table) -> None:
    sec = "2.2 / 3.1"
    pf = _read(RES_PRE / "preprocessing_facts.csv")
    if pf is not None:
        for r in pf.itertuples():
            t.add(sec, "", r.quantity, r.value, str(r.unit),
                  source="preprocessing_results/preprocessing_facts.csv", definition=str(r.definition))
    cont = _read(RES_PRE / "pod_continuity.csv")
    if cont is not None:
        for r in cont.itertuples():
            t.add(sec, "", "share of the previous month's PODs found in this month",
                  float(r.share_of_previous_month_found), "-", str(r.folder),
                  "preprocessing_results/pod_continuity.csv",
                  f"codes matching IT###E########: {r.codes_matching_pattern}; "
                  f"examples not matching: {r.examples_not_matching}")
    cov = _read(RES_PRE / "coverage_by_month.csv")
    if cov is not None:
        for r in cov.itertuples():
            for col in ("read_pods", "read_days", "retained_pods", "retained_days"):
                t.add(sec, "", f"coverage by month, {col}", int(getattr(r, col)),
                      "PODs" if "pods" in col else "days", str(r.ym),
                      "preprocessing_results/coverage_by_month.csv",
                      "before (read) and after (retained) the completeness thresholds")
    funnel = _read(RES_PRE / "funnel.csv")
    if funnel is not None and len(funnel):
        t.add(sec, "", "PODs read from the archive", int(funnel.iloc[0]["pods"]), "PODs",
              source="preprocessing_results/funnel.csv",
              definition="first step of the funnel, withdrawn active energy rows only")
        for r in funnel.itertuples():
            t.add(sec, "", "funnel, PODs after step", int(r.pods), "PODs", r.step,
                  "preprocessing_results/funnel.csv", str(r.note))
            t.add(sec, "", "funnel, user-days after step", int(r.user_days), "days", r.step,
                  "preprocessing_results/funnel.csv", str(r.note))
    else:
        t.missing(sec, "funnel", RES_PRE / "funnel.csv")

    days = pd.read_parquet(CACHE / "days.parquet")
    users = pd.read_parquet(CACHE / "users.parquet")
    dates = pd.to_datetime(days["date"])
    t.add(sec, "", "PODs retained after pre-processing", int(users["pod"].nunique()), "PODs",
          source="cache/users.parquet")
    uv_p = CACHE / "user_vectors.parquet"
    if uv_p.exists():
        n_cl = int(pd.read_parquet(uv_p, columns=["pod"])["pod"].nunique())
        t.add(sec, "", "PODs clustered (at least one day with a shape)", n_cl, "PODs",
              source="cache/user_vectors.parquet")
        t.add(sec, "", "retained PODs that cannot be clustered", int(users["pod"].nunique()) - n_cl,
              "PODs", definition="retained on zero days alone, no daily shape")
    t.add(sec, "", "valid user-days", int(len(days)), "days", source="cache/days.parquet")
    t.add(sec, "Figure 2", "daily curves with a shape", int(days["has_shape"].sum()), "days",
          source="cache/days.parquet",
          definition="days of non-null energy and at least min_nonzero_quarters readings")
    t.add(sec, "", "days without a shape", int((~days["has_shape"]).sum()), "days",
          source="cache/days.parquet")
    t.add(sec, "", "calendar days spanned", int(dates.dt.normalize().nunique()), "days")
    t.add(sec, "", "first day", str(dates.min().date()), "date")
    t.add(sec, "", "last day", str(dates.max().date()), "date")
    ym = sorted(set(zip(dates.dt.year, dates.dt.month)))
    t.add(sec, "", "months present", len(ym), "months",
          definition=" ".join(f"{y}-{m:02d}" for y, m in ym))
    t.add(sec, "", "mean valid days per POD", len(days) / max(users["pod"].nunique(), 1), "days")
    t.add(sec, "", "annualized energy of the retained PODs", users["E"].sum() / 1e6, "GWh/year",
          source="cache/users.parquet", definition="sum of Eq. 1 annualized over the valid days")
    if "prosumer" in users:
        t.add(sec, "", "PODs flagged as prosumers", int(users["prosumer"].sum()), "PODs",
              definition="PODs with at least one row of injected active energy (AN)")
        t.add(sec, "", "share of prosumers", float(users["prosumer"].mean()), "-")
    if "ateco_l1" in users:
        dom = users["ateco_l1"].astype(str).str.startswith(DOMESTIC_PREFIX)
        t.add(sec, "", "share of domestic PODs (activity label DO.*)", float(dom.mean()), "-",
              definition="over the retained PODs, before clustering")


# ------------------------------------------------------------------ Section 2.3 / 3.1 / 3.5
def clustering_numbers(t: Table) -> None:
    sec = "2.3 / 3.1"
    facts = _read(RES_CLU / "clustering_facts.csv")
    if facts is None:
        t.missing(sec, "clustering facts", RES_CLU / "clustering_facts.csv")
    else:
        unit = {"n_shapes": "days", "n_pods_clustered": "PODs", "pods_below_n_min": "PODs",
                "pods_in_published_profiles": "PODs", "n_min_group": "PODs"}
        for r in facts.itertuples():
            v = r.value
            try:
                v = float(v)
            except (TypeError, ValueError):
                pass
            t.add(sec, "", r.quantity, v, unit.get(r.quantity, "-"),
                  source="clustering_results/clustering_facts.csv")

    dic = _read(RES_CLU / "dictionary.csv")
    if dic is not None:
        code = np.load(CACHE / "day_codeword.npy") if (CACHE / "day_codeword.npy").exists() else None
        for r in dic.itertuples():
            n = int((code == r.codeword - 1).sum()) if code is not None else np.nan
            t.add("3.1", "Figure 2", "share of daily curves", float(r.share_of_days), "-",
                  f"form {r.codeword}", "clustering_results/dictionary.csv")
            t.add("3.1", "Figure 2", "number of daily curves", n, "days", f"form {r.codeword}")
            t.add("3.1", "Figure 2", "peak hour of the centroid", float(r.peak_hour), "h",
                  f"form {r.codeword}", "clustering_results/dictionary.csv")

    grp = _read(RES_CLU / "groups.csv")
    if grp is not None:
        for r in grp.itertuples():
            subj = f"DD-SLP {int(r.group)}"
            t.add("3.1", "Figure 3", "PODs in group", int(r.n_pods), "PODs", subj,
                  "clustering_results/groups.csv")
            t.add("3.1", "Figure 3", "group below n_min (not published)", bool(r.below_n_min),
                  "-", subj, "clustering_results/groups.csv")
            if hasattr(r, "prosumer_share"):
                t.add("4.4", "Figure 3", "share of prosumers in group", float(r.prosumer_share),
                      "-", subj, "clustering_results/groups.csv",
                      "share of the group's PODs flagged as prosumers (AN rows)")

    gfe = _read(RES_CLU / "group_features.csv")
    if gfe is not None:
        for r in gfe.to_dict("records"):
            gid = int(r.pop("group"))
            for k, v in r.items():
                t.add("3.1", "Figure 3", f"median of {k} in the group", v if isinstance(v, str) else float(v),
                      "-", f"DD-SLP {gid}", "clustering_results/group_features.csv")

    sweep = _read(RES_CLU / "validity_K_dispersion.csv")
    if sweep is not None:
        for r in sweep.itertuples():
            subj = f"K = {int(r.K)}"
            for col, q in (("nrmsd_p50_pod_weighted", "median nRMSD weighted by PODs"),
                           ("nrmsd_p50_unweighted", "median nRMSD unweighted"),
                           ("nrmsd_p95_pod_weighted", "95th percentile nRMSD weighted by PODs"),
                           ("pods_below_n_min", "PODs left without a profile"),
                           ("explained_share", "share of the day-shape variance explained by the groups (exact, energy-weighted)"),
                           ("nrmsd_pooled", "pooled energy-weighted nRMSD of the days from their group curve (exact)"),
                           ("n_profiles", "published profiles")):
                if not hasattr(r, col):
                    continue
                t.add("3.5", "Figure 16", q, float(getattr(r, col)), "-", subj,
                      "clustering_results/validity_K_dispersion.csv")
    valk = _read(RES_CLU / "validity_K.csv")
    if valk is not None and "stability_mean" in valk:
        for r in valk.itertuples():
            t.add("2.3 / 3.5", "", "stability ARI, mean over replicas", float(r.stability_mean),
                  "-", f"K = {int(r.K)}", "clustering_results/validity_K.csv")
            t.add("2.3 / 3.5", "", "stability ARI, least favourable replica",
                  float(r.stability_min), "-", f"K = {int(r.K)}",
                  "clustering_results/validity_K.csv")
    vald = _read(RES_CLU / "validity_D.csv")
    if vald is not None:
        for r in vald.itertuples():
            t.add("2.3", "", "silhouette of the dictionary", float(r.silhouette), "-",
                  f"D = {int(r.D)}", "clustering_results/validity_D.csv",
                  "computed on the micro-clusters")
    sens = _read(RES_CLU / "sensitivity_users.csv")
    if sens is not None:
        for r in sens.itertuples():
            t.add("3.5", "", "ARI of the user partition against the base partition",
                  float(r.ARI_vs_base), "-", f"{r.parameter} = {r.value}",
                  "clustering_results/sensitivity_users.csv",
                  f"base value {r.base_value}, same K and same dictionary")
            if r.parameter == "scale_weight" and np.isfinite(getattr(r, "scale_share_of_distance", np.nan)):
                t.add("3.5", "", "share of the distance carried by the scale block",
                      float(r.scale_share_of_distance), "-", f"scale_weight = {r.value}",
                      "clustering_results/sensitivity_users.csv")

    #Lorenzo Giannuzzo: the structural zeros of the frequency vectors, as in zeros_check.py
    uv_path = CACHE / "user_vectors.parquet"
    gp_path = CACHE / "groups.parquet"
    if uv_path.exists() and gp_path.exists():
        uv = pd.read_parquet(uv_path)
        g = pd.read_parquet(gp_path)
        fcols = [c for c in uv.columns if str(c).startswith("f_")]
        filled = (uv[fcols].to_numpy() > 0).sum(axis=1)
        t.add("2.3", "", "codewords realized per user, median", float(np.median(filled)), "-",
              definition=f"non-empty coordinates of the frequency vector, out of {len(fcols)}")
        t.add("2.3", "", "share of users with more than half the vector empty",
              float((filled <= len(fcols) / 2).mean()), "-")
        d = pd.DataFrame({"pod": uv["pod"].astype(str), "filled": filled}).merge(
            g[["pod", "group"]].assign(pod=lambda x: x["pod"].astype(str)), on="pod")
        ss_tot = float(((d["filled"] - d["filled"].mean()) ** 2).sum())
        ss_w = float(sum(((x["filled"] - x["filled"].mean()) ** 2).sum()
                         for _, x in d.groupby("group")))
        t.add("2.3", "", "share of the variation in realized codewords explained by the groups",
              1.0 - ss_w / ss_tot if ss_tot > 0 else np.nan, "-",
              definition="eta squared of the count of non-empty coordinates on the groups")


# ------------------------------------------------------------------ Section 2.4
def generation_numbers(t: Table) -> None:
    disp = _read(RES_GEN / "dispersion.csv")
    if disp is None:
        t.missing("2.4", "dispersion", RES_GEN / "dispersion.csv")
        return
    t.add("2.4", "", "median over the cells of the member nRMSD, 50th percentile",
          float(disp["nrmsd_p50"].median()), "-", source="generation_results/dispersion.csv")
    t.add("2.4", "", "median over the cells of the member nRMSD, 95th percentile",
          float(disp["nrmsd_p95"].median()), "-", source="generation_results/dispersion.csv")
    t.add("2.4", "", "cells per profile", int(disp.groupby("group")["cell"].nunique().max()), "-")
    t.add("2.4", "", "curve weighting within the cell",
          str(_CFG.get("generation.curve_weighting", "energy")), "-",
          definition="energy: normalized aggregate curve of the group; day: mean of the day shapes")


# ------------------------------------------------------------------ Section 3.1 / 3.4
def composition_numbers(t: Table) -> None:
    comp = _read(RES_CMP / "ddslp_composition.csv")
    if comp is None:
        t.missing("3.1", "composition", RES_CMP / "ddslp_composition.csv")
        return
    for r in comp.itertuples():
        subj = f"DD-SLP {int(r.group)}"
        src = "comparison_results/ddslp_composition.csv"
        t.add("3.1", "Figures 9, 12", "PODs", int(r.n_pod), "PODs", subj, src)
        t.add("3.1", "Figures 9, 12", "domestic share by points, activity label", float(r.share_domestic_pod),
              "-", subj, src, "share of points with ateco_l1 DO.*; used in every figure")
        t.add("3.1", "", "domestic share by energy, activity label", float(r.share_domestic_energy),
              "-", subj, src, "annualized energy of DO.* points over the energy of the profile")
        if hasattr(r, "share_domestic_tariff_pod"):
            t.add("3.1", "", "domestic share by points, tariff category",
                  float(r.share_domestic_tariff_pod), "-", subj, src,
                  "share of points with a domestic tariff, as read by the GSE assignment rule")
        t.add("3.4", "Figure 12", "residential (activity share >= threshold)", bool(r.residential),
              "-", subj, src)
    t.add("3.1", "", "domestic share of the published catalogue by points, activity label",
          float((comp["share_domestic_pod"] * comp["n_pod"]).sum() / comp["n_pod"].sum()), "-")


def comparison_numbers(t: Table) -> None:
    sec = "3.4"
    b1 = _read(RES_CMP / "b1_ddslp_vs_national.csv")
    comp = _read(RES_CMP / "ddslp_composition.csv")
    if b1 is None:
        t.missing(sec, "B1", RES_CMP / "b1_ddslp_vs_national.csv")
    else:
        d = b1[b1["setting"] == "S1_monthly"]
        piv = d.pivot_table(index="national", columns="ddslp", values="total_variation")
        nearest = piv.idxmin(axis=1)
        for nat, dd in nearest.items():
            t.add(sec, "Figure 8", "nearest data-driven profile", dd.replace("DDSLP_", "DD-SLP "),
                  "-", nat, "comparison_results/b1_ddslp_vs_national.csv",
                  "argmin over the columns of the total variation, S1 monthly setting")
            t.add(sec, "Figure 8", "total variation to the nearest data-driven profile",
                  float(piv.loc[nat, dd]), "-", nat)
        counts = nearest.value_counts()
        t.add(sec, "Figure 8", "national profiles positioned", int(len(nearest)), "-")
        for dd in piv.columns:
            t.add(sec, "Figure 8", "national profiles for which it is the nearest",
                  int(counts.get(dd, 0)), "-", dd.replace("DDSLP_", "DD-SLP "))
        mins = piv.min(axis=0)
        argmins = piv.idxmin(axis=0)
        for dd in piv.columns:
            t.add(sec, "Figure 8", "total variation to its nearest national profile",
                  float(mins[dd]), "-", dd.replace("DDSLP_", "DD-SLP "),
                  definition=f"nearest national profile: {argmins[dd]}")
        if comp is not None:
            res = comp.set_index("profile")["residential"]
            dom_cols = [c for c in piv.columns if bool(res.get(c, False))]
            if dom_cols:
                threshold = float(mins[dom_cols].max())
                t.add(sec, "Figure 8", "uncovered threshold", threshold, "-",
                      definition="largest distance between a residential data-driven profile "
                                 "and its nearest national profile (Section 2.5)")
                for dd in piv.columns:
                    t.add(sec, "Figure 8", "uncovered", bool(mins[dd] > threshold), "-",
                          dd.replace("DDSLP_", "DD-SLP "))
        for dd in piv.columns:
            for nat in piv.index:
                t.add(sec, "Figure 8", "total variation", float(piv.loc[nat, dd]), "-",
                      f"{nat} | {dd.replace('DDSLP_', 'DD-SLP ')}",
                      "comparison_results/b1_ddslp_vs_national.csv")

    #Lorenzo Giannuzzo: the nearest national profile cell by cell, as in the day-type figures
    try:
        import mapping_figures as MF
        for daytype in MF.DAYTYPES:
            cells = MF.cells_for(daytype)
            per_cell = MF._curves_by_profile(cells)
            near = MF._nearest_per_cell(per_cell, cells, np.inf)
            for g, by_cell in near.items():
                names = [by_cell[j][0] for j in sorted(by_cell)]
                t.add(sec, "", "distinct nearest national profiles across the seasons",
                      len(set(names)), "-", f"DD-SLP {g}, {daytype}",
                      definition="; ".join(f"{cells[j][0]}: {by_cell[j][0]} ({by_cell[j][1]:.3f})"
                                           for j in sorted(by_cell)))
    except Exception as exc:
        t.add(sec, "", "per-cell nearest national profile", "not available", "",
              definition=f"{type(exc).__name__}: {exc}")

    cs = _read(RES_CMP / "b2_summary_common_set.csv", index_col=0)
    if cs is None:
        t.missing(sec, "B2 common set", RES_CMP / "b2_summary_common_set.csv")
    else:
        for src, r in cs.iterrows():
            for col, unit in (("n", "user-months"), ("n_pods", "PODs"), ("tv_p10", "-"),
                              ("tv_p25", "-"), ("tv_median", "-"), ("tv_p75", "-"),
                              ("tv_p90", "-"), ("misallocated_kWh", "kWh"),
                              ("month_energy_kWh", "kWh"), ("misallocated_share", "-"),
                              ("months_covered", "month numbers")):
                if col in r.index:
                    v = r[col]
                    t.add(sec, "Figures 14, 15", f"B2 common set, {col}",
                          v if isinstance(v, str) else float(v), unit, src,
                          "comparison_results/b2_summary_common_set.csv",
                          "user-months of the reference year on which GSE, ARERA and the "
                          "data-driven profile are all evaluated; S1 monthly setting")
            for col in [c for c in r.index if str(c).startswith("cost_")]:
                t.add(sec, "", f"Eq. 14, {col}", float(r[col]), "EUR", src,
                      "comparison_results/b2_summary_common_set.csv")

    b2p = RES_CMP / "b2_pod_month.csv"
    if b2p.exists():
        b2 = pd.read_csv(b2p, dtype={"pod": str})
        if "common_set" in b2:
            b2 = b2[b2["common_set"]]
        users = pd.read_parquet(CACHE / "users.parquet")
        if "D_POTC" in users:
            b2 = b2.merge(users[["pod", "D_POTC"]], on="pod", how="left")
            b2["class"] = pd.cut(b2["D_POTC"], [0, 1.5, 3, 4.5, 6, np.inf],
                                 labels=["0-1.5", "1.5-3", "3-4.5", "4.5-6", ">6"])
            for (cls, src), x in b2.groupby(["class", "source"], observed=True):
                subj = f"{cls} kW | {src}"
                t.add(sec, "Figure 15", "Eq. 13 misallocated energy", float(x["misallocated_kWh"].sum()) / 1000,
                      "MWh", subj, "comparison_results/b2_pod_month.csv")
                t.add(sec, "Figure 15", "Eq. 13 share of the energy misallocated",
                      float(x["misallocated_kWh"].sum() / x["month_energy_kWh"].sum()), "-", subj)
                t.add(sec, "Figure 15", "PODs in the power class", int(x["pod"].nunique()), "PODs", subj)

    for scope in ("reference", "system"):
        b3 = _read(RES_CMP / f"b3_summary_{scope}.csv", index_col=0)
        if b3 is None:
            continue
        for src, r in b3.iterrows():
            for col in b3.columns:
                t.add(sec, "", f"B3 {scope} scope, {col}", float(r[col]),
                      "EUR" if "cost" in col else ("kWh" if "kWh" in col else "-"), src,
                      f"comparison_results/b3_summary_{scope}.csv",
                      "aggregate of the portfolio (reference) or of the whole family (system)")

    prices = _read(RES_CMP / "price_series.csv")
    if prices is not None:
        for r in prices.itertuples():
            t.add(sec, "", "Eq. 14 price series available", bool(r.available), "-", r.series,
                  "comparison_results/price_series.csv", str(r.note))

    #Lorenzo Giannuzzo: whether a single-rate GSE profile distinguishes the days of a month at
    # all. The coefficient of variation of the daily totals within each month is zero for a
    # profile that gives every day of the month the same energy.
    gp = CACHE / "gse.parquet"
    if gp.exists():
        gse = pd.read_parquet(gp)
        for col in ("PDMM", "PAUM"):
            if col not in gse:
                continue
            daily = gse.groupby(["month", "day"])[col].sum().reset_index()
            cv = daily.groupby("month")[col].agg(lambda x: x.std(ddof=0) / x.mean() if x.mean() else np.nan)
            t.add(sec, "Figure 13", "GSE profile, coefficient of variation of the daily energy within the month, maximum over months",
                  float(cv.max()), "-", col, "cache/gse.parquet",
                  "zero means every day of a month receives the same energy")
            t.add(sec, "Figure 13", "GSE profile, coefficient of variation of the daily energy within the month, median over months",
                  float(cv.median()), "-", col, "cache/gse.parquet")

    #Lorenzo Giannuzzo: the ratio of Figure 13, on every point for which it is defined
    try:
        import figures as F
        ratio = F.sunday_ratio_per_pod()
        t.add(sec, "Figure 13", "PODs with a defined Sunday to working-day ratio", int(len(ratio)), "PODs")
        t.add(sec, "Figure 13", "median Sunday to working-day energy ratio", float(ratio.median()), "-")
        t.add(sec, "Figure 13", "5th percentile of the ratio", float(ratio.quantile(0.05)), "-")
        t.add(sec, "Figure 13", "95th percentile of the ratio", float(ratio.quantile(0.95)), "-")
        t.add(sec, "Figure 13", "share of PODs with the ratio above 2.2 (outside the axis)",
              float((ratio > 2.2).mean()), "-")
    except Exception as exc:
        t.add(sec, "Figure 13", "Sunday ratio", "not available", "",
              definition=f"{type(exc).__name__}: {exc}")
    g = _read(RES_CMP / "gse_normalisation.csv")
    try:
        from comparison import GSE_PROFILES
    except Exception:
        GSE_PROFILES = ("PDMM", "PAUM", "PDMF", "PAUF")
    if g is not None:
        g = g[g["profile"].isin(GSE_PROFILES)]
        for r in g.itertuples():
            if not bool(getattr(r, "ratio_meaningful", str(r.profile).endswith("M"))):
                continue
            t.add(sec, "Figure 13", "Sunday to working-day energy ratio of the GSE profile",
                  float(r.sunday_over_weekday_energy), "-", r.profile,
                  "comparison_results/gse_normalisation.csv",
                  "holidays counted as Sundays; single-rate profiles only")


# ------------------------------------------------------------------ Section 2.6 / 3.2 / 3.3
def mapping_numbers(t: Table) -> None:
    for weight in ("pod", "energy"):
        m1 = _read(RES_M1 / f"m1_multiplicity_{weight}.csv")
        if m1 is None:
            t.missing("3.2", f"M1 {weight}", RES_M1 / f"m1_multiplicity_{weight}.csv")
            continue
        src = f"mapping_results/multiplicity/m1_multiplicity_{weight}.csv"
        t.add("2.6 / 3.2", "Figure 5", f"uninformative reference exp(H(A)), weighted by {weight}",
              float(m1["reference_uninformative"].iloc[0]), "-", source=src,
              definition="effective number of published profiles of the whole population")
        for r in m1.itertuples():
            subj = str(r.activity)
            t.add("3.2", "Figure 5", f"M1 effective number, weighted by {weight}", float(r.M1_effective),
                  "-", subj, src)
            t.add("3.2", "Figure 5", f"M1 95% bootstrap interval low, {weight}", float(r.M1_ci_low), "-", subj, src)
            t.add("3.2", "Figure 5", f"M1 95% bootstrap interval high, {weight}", float(r.M1_ci_high), "-", subj, src)
            t.add("3.2", "Figure 5", "PODs in the class", int(r.n_pod), "PODs", subj, src)
            if hasattr(r, "null_p05"):
                t.add("3.2", "Figure 5", f"M1 null 5th percentile, {weight}", float(r.null_p05), "-", subj, src,
                      "effective number over the same number of points drawn at random")
                t.add("3.2", "Figure 5", f"M1 null 95th percentile, {weight}", float(r.null_p95), "-", subj, src)
                t.add("3.2", "Figure 5", f"M1 null quantile of the observed value, {weight}",
                      float(r.null_quantile), "-", subj, src,
                      "share of random classes of the same size with M1 <= observed")
                t.add("3.2", "Figure 5", f"coherent at 5%, {weight}", bool(r.coherent_at_5pct), "-", subj, src)
        if weight == "pod" and "null_quantile" in m1:
            t.add("3.2", "Figure 5", "classes coherent at 5% (weighted by points)",
                  int(m1["coherent_at_5pct"].sum()), "classes", source=src)
            t.add("3.2", "Figure 5", "median M1 over the classes (weighted by points)",
                  float(m1["M1_effective"].median()), "-", source=src)

    ct = _read(RES_MAP / "contingency_pod.csv", index_col=0)
    if ct is not None:
        share = ct.div(ct.sum(axis=1), axis=0)
        for cls, row in share.iterrows():
            for prof, v in row.items():
                if v > 0:
                    t.add("3.2", "Figure 4", "share of the class in the profile", float(v), "-",
                          f"{cls} | {prof.replace('DDSLP_', 'DD-SLP ')}",
                          "mapping_results/contingency_pod.csv")

    for weight in ("pod", "energy"):
        m2 = _read(RES_M2 / f"m2_aggregation_{weight}.csv")
        if m2 is None:
            continue
        src = f"mapping_results/aggregation/m2_aggregation_{weight}.csv"
        for r in m2.itertuples():
            subj = r.profile.replace("DDSLP_", "DD-SLP ")
            t.add("3.3", "Figure 7", f"M2 effective number, weighted by {weight}", float(r.M2_effective), "-", subj, src)
            if weight == "pod":
                t.add("3.3", "Figure 7", "classes present", int(r.n_classes_present), "classes", subj, src)
                t.add("3.3", "Figure 7", "classes covering 80% of the points", int(r.M2_hard80), "classes", subj, src)

    lift = _read(RES_M2 / "lift.csv")
    if lift is not None:
        lp = lift[(lift["weight"] == "pod") & (lift["count"] > 0)]
        for prof, x in lp.groupby("profile"):
            top = x.sort_values("lift", ascending=False).head(3)
            for r in top.to_dict("records"):
                t.add("3.3", "Figure 7", "lift, top classes of the profile (points)", float(r["lift"]), "-",
                      f"{prof.replace('DDSLP_', 'DD-SLP ')} | {r['activity_grouped']}",
                      "mapping_results/aggregation/lift.csv",
                      f"p(class|profile) = {r['p_class_given_profile']:.3f}, count = {int(r['count'])}")
        dom = lp[lp["activity_grouped"].astype(str).str.startswith(DOMESTIC_PREFIX)]
        if len(dom):
            t.add("3.3", "Figure 7", "largest lift of the domestic class in any profile",
                  float(dom["lift"].max()), "-", source="mapping_results/aggregation/lift.csv")

    cov = _read(RES_M3 / "m3_coverage_pod.csv")
    if cov is not None:
        src = "mapping_results/coverage/m3_coverage_pod.csv"
        for r in cov.itertuples():
            subj = r.national_profile
            t.add("3.4", "Figures 9, 10", "behaviors delivered (effective number of DD-SLPs)",
                  float(r.M3_ddslp_effective), "-", subj, src)
            t.add("3.4", "Figure 10", "activity classes represented (effective number)",
                  float(r.M3_classes_effective), "-", subj, src)
            t.add("3.4", "Figures 9, 10, 11", "PODs the national profile is applied to", int(r.n_pod), "PODs", subj, src)
            t.add("3.4", "Figure 11", "reach outside the declared category", float(r.reach_outside_category),
                  "-", subj, src, "Eq. 12, activity dimension only (lower bound)")
            t.add("3.4", "Figure 11", "domestic share of the reach", float(r.reach_domestic_share), "-", subj, src)
    reach = _read(RES_M3 / "m3_reach_pod.csv")
    if reach is not None:
        q = reach[reach["activity"].astype(str).str.startswith("__profile__")]
        for r in q.itertuples():
            t.add("3.4", "Figure 9", "share of the applied PODs in the data-driven profile",
                  float(r.reach_share), "-",
                  f"{r.national_profile} | {str(r.activity).replace('__profile__DDSLP_', 'DD-SLP ')}",
                  "mapping_results/coverage/m3_reach_pod.csv")

    sk = _read(RES_MAP / "sensitivity_k.csv")
    if sk is not None:
        for r in sk.itertuples():
            subj = f"K = {int(r.K)}"
            for col in ("published_profiles", "pods_in_published", "M1_median",
                        "reference_uninformative", "spearman_M1_vs_base", "PDMM_behaviours"):
                if pd.isna(getattr(r, col, np.nan)):
                    continue
                t.add("3.5", "", f"sensitivity to K, {col}", float(getattr(r, col)), "-", subj,
                      "mapping_results/sensitivity_k.csv",
                      "same tree of the users cut at K; Spearman over the classes common to both")
    l2 = _read(RES_M1 / "m1_multiplicity_pod_ateco_l2.csv")
    if l2 is not None:
        body = l2[l2["activity"] != "other (below n_min)"]
        t.add("3.5", "", "M1 at the ATECO class level, median over classes", float(body["M1_effective"].median()), "-",
              source="mapping_results/multiplicity/m1_multiplicity_pod_ateco_l2.csv")
        t.add("3.5", "", "ATECO classes retained at the class level", int(len(body)), "classes")


def main() -> None:
    t0 = time.time()
    print(f"\n{'='*78}\n  NUMERICAL RESULTS\n{'='*78}")
    t = Table()
    for fn in (preprocessing_numbers, clustering_numbers, generation_numbers,
               composition_numbers, comparison_numbers, mapping_numbers):
        try:
            fn(t)
        except Exception as exc:
            #Lorenzo Giannuzzo: one block failing must not cost the others; the failure is
            # written into the table itself so that it cannot go unnoticed
            t.add("", "", f"{fn.__name__} failed", "not available", "",
                  definition=f"{type(exc).__name__}: {exc}")
            print(f"  ! {fn.__name__}: {type(exc).__name__}: {exc}")
    df = pd.DataFrame(t.rows)
    #Lorenzo Giannuzzo: the source column names the folders as config.py lays them out
    rename = {"preprocessing_results": RES_PRE.name, "clustering_results": RES_CLU.name,
              "generation_results": RES_GEN.name, "comparison_results": RES_CMP.name,
              "mapping_results": RES_MAP.name}
    for a, b in rename.items():
        df["source"] = df["source"].astype(str).str.replace(a, b, regex=False)
    df.to_csv(OUT, index=False)
    print(f"  {len(df)} numbers -> {OUT}   ({time.time()-t0:.0f}s)\n")


if __name__ == "__main__":
    main()
