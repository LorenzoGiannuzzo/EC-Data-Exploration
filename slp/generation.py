"""Stage 3 — Standard load profile generation (Section 2.4).

A group of users is not yet a standard profile. The step that turns it into one
is the same the regulator performs: the metered days of the members are averaged
over a grid of seasons and day types.

Two quantities come out per cell (Eq. 6). The curves say how the energy of a day
is spread across its ninety-six intervals, each summing to one and therefore
carrying nothing about how much that day consumes; the weights say how much of
the year falls on each kind of day. A profile is the pair, and neither suffices
alone.

The profile is then rescaled to 1000 kWh a year, which is what makes a standard
profile reusable: a group holds users of every size, and what they share is the
shape of their consumption, not its magnitude. Whoever applies the profile
restores the magnitude from the one quantity always known, the billed annual
consumption.

One quantity is reported that no national catalogue publishes: the dispersion of
the members around the mean. A profile is a claim about a population, and the
dispersion says how well the claim holds. Section 2.5 needs it, since a distance
between a national profile and a data-driven one is a number without a scale
until it can be told whether the national curve lies further from the centroid
than the members themselves do.

Outputs
    cache/profiles.npy              (K, n_cells, 96) the typical curves
    cache/profile_weights.parquet   group, cell, weight, dispersion
    paper_results/generation_results/
        profiles.csv                every curve, in the national format
        profiles.png                the curves, per group and season
        weights.csv / weights.png   how much of the year each cell carries
        dispersion.csv              how far the members lie from their profile
        summary.txt

Run
    python generation.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.cache import check_manifest
from common.config import load_config
from preprocessing import day_type, italian_holidays, season

DAYTYPE_ORDER = ["weekday", "saturday", "sunday"]

#Lorenzo Giannuzzo: the internal keys are lower case and mid stands for the two
#half seasons the grid merges into one cell, which is unreadable on a figure.
#The mapping is applied at drawing time only, so nothing downstream of the plots
#has to know about it and the cell keys written to the CSV files stay as they are.
SEASON_LABEL = {"mid": "Autumn/Spring"}


def pretty(word: object) -> str:
    w = str(word)
    if w.lower() in SEASON_LABEL:
        return SEASON_LABEL[w.lower()]
    return w[:1].upper() + w[1:] if w else w


def pretty_cell(cell: object) -> str:
    return " | ".join(pretty(p) for p in str(cell).split("|"))
MONTH_LABELS = [f"M{m:02d}" for m in range(1, 13)]


# ── the regulatory grid ──────────────────────────────────────────────────────
def period_of(dates: pd.Series, smap: dict, grid: str) -> pd.Series:
    """The first coordinate of the grid, which the config chooses.

    Two grids are admissible because the two national catalogues do not use the
    same one. The ARERA tables are published per month, the GSE workbook is a
    calendar year in its own right, and a profile averaged over three seasons
    carries fewer degrees of freedom than either. Comparing objects built on
    different grids measures the grid before it measures the classification, so
    the season grid stays available and the month grid is what the comparison
    against the published tables is run on.
    """
    if grid == "season":
        return season(dates, smap)
    return dates.dt.strftime("M%m")


def calendar_days_per_cell(seasons: dict, year: int = 2025,
                           grid: str = "season") -> pd.Series:
    """How many days of each kind a standard year holds.

    The weights are taken over the calendar rather than over the observed days.
    The period covered has whole months missing, so the observed share of the
    year spent in mid-season would be the share of mid-season that happens to be
    in the archive, which is a property of the export and not of the users.
    Averaging within a cell and weighting by the calendar separates the two.
    """
    smap = {m: lab for lab, months in seasons.items() for m in months}
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    df = pd.DataFrame({"date": days})
    df["period"] = period_of(df["date"], smap, grid)
    df["daytype"] = day_type(df["date"], italian_holidays(year))
    return df.groupby(["period", "daytype"]).size().rename("calendar_days")


def cell_index(days: pd.DataFrame, cells: list[tuple[str, str]]) -> np.ndarray:
    """The cell each day falls in. `days` carries the period column main() adds."""
    lookup = {c: i for i, c in enumerate(cells)}
    key = list(zip(days["period"], days["daytype"]))
    return np.array([lookup.get(k, -1) for k in key], dtype="int32")


# ── Eq. 6 ────────────────────────────────────────────────────────────────────
def typical_curves(shapes: np.ndarray, days: pd.DataFrame, groups: pd.DataFrame,
                   cells: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Average the members' days over the grid.

    Returns (curves, counts, per-cell daily energy), where curves is
    (K, n_cells, 96) with each curve summing to one.
    """
    d = days[days["has_shape"]].merge(groups[["pod", "group"]], on="pod", how="inner")
    gcodes = pd.Categorical(d["group"], categories=sorted(groups["group"].unique()))
    K = len(gcodes.categories)
    ci = cell_index(d, cells)
    ok = (ci >= 0) & (gcodes.codes >= 0)

    key = (gcodes.codes[ok].astype("int64") * len(cells) + ci[ok])
    sub = np.asarray(shapes[d.loc[ok, "shape_idx"].to_numpy()], dtype="float64")

    sums = np.zeros((K * len(cells), 96))
    np.add.at(sums, key, sub)
    cnt = np.bincount(key, minlength=K * len(cells)).astype("float64")
    curves = np.divide(sums, cnt[:, None], out=np.zeros_like(sums), where=cnt[:, None] > 0)

    # each curve is a distribution over the day; renormalise against rounding
    s = curves.sum(axis=1, keepdims=True)
    curves = np.divide(curves, s, out=np.zeros_like(curves), where=s > 0)

    # mean daily energy per cell, which is what the weights are built on
    e = np.zeros(K * len(cells))
    np.add.at(e, key, d.loc[ok, "energy"].to_numpy())
    mean_e = np.divide(e, cnt, out=np.zeros_like(e), where=cnt > 0)

    energy = pd.DataFrame({
        "group": np.repeat(list(gcodes.categories), len(cells)),
        "cell": [f"{s_}|{t_}" for _ in range(K) for s_, t_ in cells],
        "n_days_observed": cnt.astype(int),
        "mean_daily_kWh": mean_e,
    })
    return (curves.reshape(K, len(cells), 96), cnt.reshape(K, len(cells)), energy)


def calendar_weights(energy: pd.DataFrame, cal: pd.Series,
                     cells: list[tuple[str, str]]) -> pd.DataFrame:
    """Eq. 6, second quantity: the share of the year each cell carries.

    Built as mean daily energy times the number of days the calendar holds for
    that cell, then closed to one. Weighting by the calendar rather than by the
    observed days is what keeps the missing months out of the profile.
    """
    cal_map = {f"{s}|{t}": int(cal.get((s, t), 0)) for s, t in cells}
    e = energy.copy()
    e["calendar_days"] = e["cell"].map(cal_map)
    e["unnormalised"] = e["mean_daily_kWh"] * e["calendar_days"]
    tot = e.groupby("group")["unnormalised"].transform("sum")
    e["weight"] = np.where(tot > 0, e["unnormalised"] / tot, 0.0)
    return e


# ── dispersion ───────────────────────────────────────────────────────────────
def dispersion(shapes: np.ndarray, days: pd.DataFrame, groups: pd.DataFrame,
               curves: np.ndarray, cells: list[tuple[str, str]],
               rng: np.random.Generator, n_max: int = 4000) -> pd.DataFrame:
    """How far the members lie from the curve that stands in for them.

    nRMSD between each member day and the curve of its cell, normalised on the
    mean of the curve, which is 1/96 since the curves are distributions. The
    95th percentile is the criterion of Section 2.5: a national profile is a
    legitimate representative only where it lies no further from the centroid
    than the members themselves do.
    """
    d = days[days["has_shape"]].merge(groups[["pod", "group"]], on="pod", how="inner")
    cats = sorted(groups["group"].unique())
    gc = pd.Categorical(d["group"], categories=cats).codes
    ci = cell_index(d, cells)
    ok = (ci >= 0) & (gc >= 0)
    d, gc, ci = d[ok], gc[ok], ci[ok]

    rows = []
    for gi, g in enumerate(cats):
        for cj, (s_, t_) in enumerate(cells):
            m = (gc == gi) & (ci == cj)
            n = int(m.sum())
            if n == 0:
                continue
            idx = d.loc[m, "shape_idx"].to_numpy()
            if n > n_max:
                idx = rng.choice(idx, n_max, replace=False)
            mem = np.asarray(shapes[np.sort(idx)], dtype="float64")
            ref = curves[gi, cj]
            nrmsd = np.sqrt(((mem - ref) ** 2).mean(axis=1)) / (1.0 / 96)
            rows.append({
                "group": g, "cell": f"{s_}|{t_}", "n_days": n,
                "nrmsd_p50": float(np.percentile(nrmsd, 50)),
                "nrmsd_p90": float(np.percentile(nrmsd, 90)),
                "nrmsd_p95": float(np.percentile(nrmsd, 95)),
            })
    return pd.DataFrame(rows)


# ── plots ────────────────────────────────────────────────────────────────────
def plot_profiles(curves: np.ndarray, weights: pd.DataFrame, sizes: pd.Series,
                  cells: list[tuple[str, str]], groups_list: list, path: Path,
                  kwh_year: float) -> None:
    """One row per profile, one column per season; the three day types overlaid.

    Drawn in kW, which is the form in which a profile is used: the curve of a
    cell scaled by the energy that cell carries over the days it holds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seasons = list(dict.fromkeys(s for s, _ in cells))
    K = len(groups_list)
    # a monthly grid puts twelve columns on the page, so the column narrows
    col_w = 3.3 if len(seasons) <= 4 else 1.9
    fig, axes = plt.subplots(K, len(seasons),
                             figsize=(col_w * len(seasons), 1.9 * K),
                             sharex=True, squeeze=False)
    x = np.arange(96) / 4.0
    colors = {"weekday": "#0d1f3c", "saturday": "#1565c0", "sunday": "#c62828"}

    wmap = {(r["group"], r["cell"]): (r["weight"], r["calendar_days"])
            for _, r in weights.iterrows()}

    for i, g in enumerate(groups_list):
        top = 0.0
        for j, s_ in enumerate(seasons):
            ax = axes[i][j]
            for t_ in DAYTYPE_ORDER:
                if (s_, t_) not in cells:
                    continue
                cj = cells.index((s_, t_))
                w, nd = wmap.get((g, f"{s_}|{t_}"), (0.0, 0))
                if nd == 0 or w == 0:
                    continue
                # kWh on one such day, spread over the quarter-hours -> kW
                kwh_day = kwh_year * w / nd
                y = curves[i, cj] * kwh_day * 4.0
                ax.plot(x, y, color=colors[t_], lw=1.2, label=t_ if i == 0 and j == 0 else None)
                top = max(top, y.max())
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(pretty(s_), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Standard Profile {g}\n({int(sizes.get(g, 0))} PODs)",
                              fontsize=8)
        for j in range(len(seasons)):
            axes[i][j].set_ylim(0, top * 1.12 if top else 1)

    handles = [plt.Line2D([], [], color=colors[t], lw=1.4, label=pretty(t))
               for t in DAYTYPE_ORDER]
    #Lorenzo Giannuzzo: the band reserved for the legend is a fixed height in
    #inches rather than a fraction of the figure, because the figure grows with
    #the number of profiles and a fraction would leave a hand of white space on a
    #tall grid and none at all on a short one.
    fig.legend(handles=handles, loc="upper center", ncol=len(DAYTYPE_ORDER),
               fontsize=8, frameon=True, framealpha=1.0, edgecolor="black",
               bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3)
    fig.supxlabel("Time of day [h]", fontsize=11)
    #Lorenzo Giannuzzo: the label runs along the short side of the figure, and the
    #figure is only as tall as the number of profiles makes it. Broken over two
    #lines and sized against the height, it fits a grid of two rows as well as one
    #of ten instead of being clipped on the first.
    fig.supylabel(f"Power normalized to an annual\nconsumption of "
                  f"{kwh_year:,.0f} kWh [kW]",
                  fontsize=min(11.0, max(7.0, 2.2 * fig.get_figheight())))
    fig_h = fig.get_figheight()
    fig.tight_layout(rect=[0, 0, 1, max(0.80, 1.0 - 0.45 / fig_h)])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights(weights: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = weights.pivot(index="group", columns="cell", values="weight")
    fig, ax = plt.subplots(figsize=(1.4 + 0.85 * piv.shape[1], 1.4 + 0.42 * piv.shape[0]))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(piv.shape[1]), [pretty_cell(c) for c in piv.columns],
                  rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0]),
                  [f"Standard Profile {g}" for g in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if v >= 0.02:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > piv.to_numpy().max() * 0.55 else "#0d1f3c")
    ax.set_title("Share of the annual energy carried by each cell [%]",
                 fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── pipeline ─────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    cfg = load_config()
    out = cfg.results_dir("generation")
    rng = np.random.default_rng(int(cfg.get("clustering.random_state", 42)))
    kwh_year = float(cfg.get("generation.normalise_kwh_year", 1000))

    print(f"\n{'='*78}\nSTAGE 3 — STANDARD LOAD PROFILE GENERATION (Section 2.4)\n{'='*78}\n")

    # what wrote cache/ has to be the configuration now in force, or this stage
    # reports profiles built on a dictionary that no longer exists
    man = check_manifest(cfg.cache_dir, cfg["clustering"], "generation")
    print(f"  cache written with D = {man.get('n_codewords')}, "
          f"K = {man.get('n_profiles')}, unit {man.get('shape_unit')}\n")

    shapes = np.load(cfg.cache_dir / "shapes.npy", mmap_mode="r")
    days = pd.read_parquet(cfg.cache_dir / "days.parquet")
    groups = pd.read_parquet(cfg.cache_dir / "groups.parquet")
    users = pd.read_parquet(cfg.cache_dir / "users.parquet")

    # groups below n_min are reported but do not carry a profile
    keep = groups[~groups["below_n_min"]] if "below_n_min" in groups else groups
    dropped = len(groups) - len(keep)
    glist = sorted(keep["group"].unique())
    sizes = keep.groupby("group").size()
    print(f"  {len(glist)} groups, {len(keep):,} PODs"
          + (f"   ({dropped:,} PODs in groups below n_min, no profile)" if dropped else ""))
    if not glist:
        raise SystemExit(
            f"\n  No group carries a profile: all {len(groups):,} PODs sit in groups\n"
            f"  below n_min. Either K is too high for the population, or\n"
            f"  clustering.min_group_size is too demanding. It defaults to\n"
            f"  max(30, 3K); set it explicitly in config.yaml to override.\n")

    # ── the grid ─────────────────────────────────────────────────────────────
    seasons_cfg = cfg.get("preprocessing.seasons")
    grid = str(cfg.get("generation.grid", "season")).lower()
    if grid not in ("season", "month"):
        raise SystemExit(
            f"\n  generation.grid is '{grid}'; it must be 'season' or 'month'.\n")
    smap = {m: lab for lab, months in seasons_cfg.items() for m in months}
    days["period"] = period_of(days["date"], smap, grid)
    periods = list(seasons_cfg) if grid == "season" else MONTH_LABELS
    present = days.groupby(["period", "daytype"]).size()
    cells = [(p, t) for p in periods for t in DAYTYPE_ORDER
             if (p, t) in present.index]
    cal = calendar_days_per_cell(seasons_cfg, grid=grid)
    missing = [p for p in periods if not any(c[0] == p for c in cells)]
    print(f"  grid: {len(periods)} {grid}s x {len(DAYTYPE_ORDER)} day types "
          f"= {len(cells)} cells")
    if missing:
        print(f"    WARNING: no observed day falls in {', '.join(missing)}; "
              "those cells carry no curve and their calendar weight is lost")

    # ── Eq. 6 ────────────────────────────────────────────────────────────────
    print("\n  averaging the members' days over the grid...")
    curves, counts, energy = typical_curves(np.asarray(shapes), days, keep, cells)
    weights = calendar_weights(energy, cal, cells)
    print(f"    {curves.shape[0]} profiles x {curves.shape[1]} cells x 96 values")

    thin = weights[weights["n_days_observed"] < 10]
    if len(thin):
        print(f"    WARNING: {len(thin)} cells rest on fewer than 10 observed days")

    # ── dispersion ───────────────────────────────────────────────────────────
    print("  measuring how far the members lie from their profile...")
    disp = dispersion(np.asarray(shapes), days, keep, curves, cells, rng)
    med = disp["nrmsd_p95"].median()
    print(f"    nRMSD of the members from their own curve: "
          f"median p95 = {med:.2f}")
    print(f"      this is the yardstick of Section 2.5: a national profile is a")
    print(f"      legitimate representative only if it lies closer than this")

    # ── write ────────────────────────────────────────────────────────────────
    np.save(cfg.cache_dir / "profiles.npy", curves.astype("float32"))
    weights.to_parquet(cfg.cache_dir / "profile_weights.parquet", index=False)

    rows = []
    for i, g in enumerate(glist):
        for j, (s_, t_) in enumerate(cells):
            w = weights[(weights["group"] == g) &
                        (weights["cell"] == f"{s_}|{t_}")]
            wv = float(w["weight"].iloc[0]) if len(w) else 0.0
            nd = int(w["calendar_days"].iloc[0]) if len(w) else 0
            kwh_day = kwh_year * wv / nd if nd else 0.0
            rows.append({
                # `period` is the grid coordinate, a season or a month; `season`
                # repeats it so that readers written against the season grid keep
                # working unchanged
                "profile": g, "period": s_, "season": s_, "daytype": t_,
                "weight": round(wv, 5),
                "calendar_days": nd,
                "n_days_observed": int(counts[i, j]),
                "kWh_per_day": round(kwh_day, 4),
                # q1..q96 are the shape, summing to one over the day, which is
                # what Eq. 6 defines and what the comparison of Section 2.5
                # consumes. kW1..kW96 are the same curve in the unit the figure
                # is drawn in, the power a user drawing kwh_year over the year
                # would take in that quarter-hour, so that the file can be read
                # without reconstructing the conversion from kWh_per_day.
                **{f"q{k+1}": round(float(curves[i, j, k]), 6) for k in range(96)},
                **{f"kW{k+1}": round(float(curves[i, j, k]) * kwh_day * 4.0, 6)
                   for k in range(96)},
            })
    pd.DataFrame(rows).to_csv(out / "profiles.csv", index=False)
    weights.to_csv(out / "weights.csv", index=False)
    disp.to_csv(out / "dispersion.csv", index=False)

    plot_profiles(curves, weights, sizes, cells, glist, out / "profiles.png", kwh_year)
    plot_weights(weights, out / "weights.png")

    with open(out / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Profiles                     {len(glist)}\n")
        fh.write(f"Grid                         {grid}\n")
        fh.write(f"Cells per profile            {len(cells)}  "
                 f"({len(periods)} {grid}s x {len(DAYTYPE_ORDER)} day types)\n")
        fh.write(f"PODs carrying a profile      {len(keep):,}\n")
        if dropped:
            fh.write(f"PODs in groups below n_min   {dropped:,} (no profile)\n")
        fh.write(f"Normalised to                {kwh_year:,.0f} kWh per year\n")
        fh.write(f"\nDispersion of the members from their own curve, nRMSD:\n")
        fh.write(f"  median p50                 {disp['nrmsd_p50'].median():.3f}\n")
        fh.write(f"  median p95                 {disp['nrmsd_p95'].median():.3f}\n")
        fh.write(f"  worst cell p95             {disp['nrmsd_p95'].max():.3f}\n")
        fh.write(
            "\nThe weights are taken over the calendar, not over the observed days.\n"
            "The period has whole months missing, so an observed share would be the\n"
            "share of that season which happens to sit in the archive, a property of\n"
            "the export rather than of the users. Averaging within the cell and\n"
            "weighting by the calendar keeps the two apart.\n")
        if len(thin):
            fh.write(f"\nCells resting on fewer than 10 observed days: {len(thin)}\n")

    print(f"\n{'='*78}")
    print(f"  {len(glist)} profiles, each {len(cells)} curves + {len(cells)} weights")
    print(f"  normalised to {kwh_year:,.0f} kWh/year, in the format of the national catalogue")
    print(f"\n  cache/   profiles.npy, profile_weights.parquet")
    print(f"  results/ {out.name}   ({time.time()-t0:.0f}s)")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()