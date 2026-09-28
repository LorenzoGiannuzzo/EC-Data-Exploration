"""Figures for the comparison stage (Section 2.5).

Run after `python main.py --stage comparison`:

    python figures.py

Every figure is written both as PNG for the manuscript and as PDF for the camera-ready
version, at the resolution declared under output.figure_dpi in config.yaml. Nothing is computed here that is not already in the comparison
outputs or in the cache, so the figures cannot disagree with the tables.

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import calendar as C  # noqa: E402

from common.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
_CFG = load_config()
CACHE = _CFG.cache_dir
RES = _CFG.results_dir("comparison")
FIG = _CFG.figures_dir("comparison")

INK = "#1b1b1b"
GRID = "#d9d9d9"
ACCENT = "#0b3c5d"
WARM = "#c1440e"
NEUTRAL = "#7d8491"
PALETTE = ["#0b3c5d", "#328cc1", "#c1440e", "#e2a33c", "#4c8055", "#7d5ba6"]

#Lorenzo Giannuzzo: one colour per data-driven profile for every figure of the paper,
# keyed on the profile number and not on its rank within a figure, so that DD-SLP 3 is
# the same colour wherever it appears.
DDSLP_PALETTE = ["#6b6b6b", "#c1440e", "#328cc1", "#e2a33c", "#7d5ba6", "#4c9a6a",
                 "#a8577e", "#0b3c5d", "#8c6c3f", "#5f7d95", "#d98c3f", "#2e7d4f"]


def ddslp_color(group: object) -> str:
    g = int(str(group).replace("DDSLP_", "").replace("DD-SLP", "").strip())
    return DDSLP_PALETTE[(g - 1) % len(DDSLP_PALETTE)]


def ddslp_label(group: object) -> str:
    return "DD-SLP " + str(group).replace("DDSLP_", "").replace("DD-SLP", "").strip()


#Lorenzo Giannuzzo: the published codes are spelled out in every label, since a reader of
# the paper should not have to know that PAUM means other uses at a single rate.
GSE_NAME = {"PDMM": "GSE domestic, single rate", "PDMF": "GSE domestic, time bands",
            "PAUM": "GSE other uses, single rate", "PAUF": "GSE other uses, time bands"}
RESIDENCY_EN = {"non residente": "non-resident", "residente": "resident", "tutti": "all"}


def national_label(name: object) -> str:
    """Full name of a national profile, from the spelling the pipeline keys on."""
    txt = str(name).strip()
    for code, full in GSE_NAME.items():
        if txt in (code, f"GSE {code}"):
            return full
    if txt.upper().startswith("ARERA"):
        rest = txt[5:].strip()
        parts = rest.split(" ", 1)
        cls = parts[0]
        res = parts[1].strip().lower() if len(parts) > 1 else ""
        res = RESIDENCY_EN.get(res, res)
        return f"ARERA {cls} kW, {res}" if res else f"ARERA {cls} kW"
    return txt


def legend_top(target, handles=None, labels=None, ncol: int = 1, **kw):
    """Legend centred at the top, inside a box with a black border."""
    args = [] if handles is None else ([handles] if labels is None else [handles, labels])
    opts = dict(loc="upper center", ncol=ncol, frameon=True, framealpha=1.0,
                edgecolor="black", fontsize=7.5)
    opts.update(kw)
    return target.legend(*args, **opts)


def legend_below(ax, handles=None, labels=None, ncol: int = 1, pad_in: float = 0.50, **kw):
    """Legend centred under an axis, clear of its tick labels and axis title.

    The offset is given in inches below the axis and converted to axes fraction, so the
    legend sits at the same distance from the axis whatever the height of the panel.
    Call it after the layout is final (after tight_layout), since it reads the position
    of the axis in the figure.
    """
    h_in = ax.get_position().height * ax.figure.get_figheight()
    args = [] if handles is None else ([handles] if labels is None else [handles, labels])
    opts = dict(loc="upper center", bbox_to_anchor=(0.5, -pad_in / max(h_in, 1e-6)),
                ncol=ncol, frameon=True, framealpha=1.0, edgecolor="black", fontsize=7.5)
    opts.update(kw)
    return ax.legend(*args, **opts)


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "figure.dpi": 120,
})


def save(fig, name: str) -> None:
    #Lorenzo Giannuzzo: one directory per format, as in the mapping figures, so that the
    #whole PNG set or the whole PDF set can be selected at once instead of by extension.
    for ext in ("png", "pdf"):
        out = FIG / ext
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.{ext}", dpi=_CFG.figure_dpi, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def pretty_season(s: str) -> str:
    return {"mid": "Autumn/Spring"}.get(str(s), str(s)[:1].upper() + str(s)[1:])


# ------------------------------------------------------------------------- table B1
def save_table(table: pd.DataFrame, name: str, folder: Path, bold: np.ndarray | None = None) -> None:
    """Write a table of the paper as CSV and as a formatted Excel sheet.

    The tables live in a `tables` folder next to the `figures` folder of the stage, so that a
    table and the figures of the same section are found in the same place. `bold` is a mask
    with the shape of the table body marking the cells set in bold in the Excel sheet, since
    a CSV cannot carry it.
    """
    out = Path(folder) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{name}.csv", index=False)
    try:
        from openpyxl.styles import Alignment, Border, Font, Side
        with pd.ExcelWriter(out / f"{name}.xlsx", engine="openpyxl") as xw:
            table.to_excel(xw, index=False, sheet_name="table")
            ws = xw.sheets["table"]
            thin = Side(style="thin", color="000000")
            for j, col in enumerate(table.columns, start=1):
                c = ws.cell(row=1, column=j)
                c.font = Font(bold=True)
                c.border = Border(top=thin, bottom=thin)
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                width = max(len(str(col)), *(len(str(v)) for v in table[col])) + 2
                ws.column_dimensions[c.column_letter].width = min(width, 45)
            for i in range(len(table)):
                for j in range(len(table.columns)):
                    c = ws.cell(row=i + 2, column=j + 1)
                    c.alignment = Alignment(horizontal="left" if j == 0 else "center")
                    if bold is not None and bold[i, j]:
                        c.font = Font(bold=True)
                    if i == len(table) - 1:
                        c.border = Border(bottom=thin)
    except ImportError:
        pass
    print(f"  {name} (table)")


def _drop_retired_figure(folder: Path, name: str) -> None:
    #Lorenzo Giannuzzo: a figure replaced by a table is removed from the results, so that the
    # collection step does not carry the old image into paper_results/figures
    for ext in ("png", "pdf"):
        p = Path(folder) / ext / f"{name}.{ext}"
        if p.exists():
            p.unlink()


def table_b1_distance() -> None:
    """How far each national profile sits from each data-driven profile, as a table.

    Rows are the national profiles, ordered by the distance to their nearest data-driven
    profile, and columns the data-driven profiles; every cell is the total variation of Eq. 8
    in the S1 setting. The nearest data-driven profile of every row, the positioning Section
    2.5 defines, is named in the last column and set in bold in the Excel sheet.
    """
    b1 = pd.read_csv(RES / "b1_ddslp_vs_national.csv")
    d = b1[b1["setting"] == "S1_monthly"]
    m = d.pivot_table(index="national", columns="ddslp", values="total_variation")
    m = m.loc[m.min(axis=1).sort_values().index]
    order = sorted(m.columns, key=lambda c: int(c.split("_")[1]))
    m = m[order]
    vals = m.to_numpy()
    best = np.nanargmin(vals, axis=1)

    table = pd.DataFrame({"National profile": [national_label(n) for n in m.index]})
    for j, c in enumerate(m.columns):
        table[ddslp_label(c)] = [f"{v:.3f}" if np.isfinite(v) else "" for v in vals[:, j]]
    table["Nearest data-driven profile"] = [ddslp_label(m.columns[j]) for j in best]
    bold = np.zeros(table.shape, dtype=bool)
    for i, j in enumerate(best):
        bold[i, j + 1] = True
    save_table(table, "table_b1_total_variation", RES, bold=bold)
    _drop_retired_figure(FIG, "fig1_b1_distance_heatmap")


# ------------------------------------------------------------------------ figure 2
def fig_daily_shapes(cell: str = "winter|weekday") -> None:
    """The shapes themselves, each family against the national profile meant for it.

    A data-driven profile is residential or not by who ended up in it, not by
    construction, so the split is read from the composition table the comparison stage
    writes, on the activity label that Section 2.6 also uses. The share in the legend is
    the share of points with a domestic activity label.
    """
    arr = np.load(CACHE / "profiles.npy")
    w = pd.read_parquet(CACHE / "profile_weights.parquet")
    groups = sorted(w["group"].unique())
    cells = list(w.loc[w["group"] == groups[0], "cell"])
    if cell not in cells:
        #Lorenzo Giannuzzo: on the monthly grid the cell is rebuilt from the first month
        # of the season rather than failing on a key that does not exist there
        season, daytype = cell.split("|")
        cand = [c for c in cells if c.endswith("|" + daytype)]
        cell = cand[0] if cand else cells[0]
    ci = cells.index(cell)
    hourly = arr.reshape(arr.shape[0], arr.shape[1], 24, 4).sum(axis=3)

    comp = pd.read_csv(RES / "ddslp_composition.csv").set_index("group")
    gse = pd.read_parquet(CACHE / "gse.parquet")
    days = pd.read_parquet(CACHE / "days.parquet")
    cal = C.build_calendar(int(_CFG.get("comparison.reference_year", 2025)),
                           C.season_map_from_days(days))
    season, daytype = cell.split("|")
    sel_days = cal[(cal["season"] == season) & (cal["daytype"] == daytype)]
    sel = gse[pd.to_datetime(dict(year=gse.year, month=gse.month, day=gse.day))
              .dt.date.isin(sel_days["date"].dt.date)]

    panels = [("Residential data-driven profiles", True, "PDMM"),
              ("Remaining data-driven profiles", False, "PAUM")]
    #Lorenzo Giannuzzo: two vertical scales. The residential shapes vary between 2 and 7 per cent
    # of the day per hour, a daytime-only activity reaches 9, and a shared scale flattened the
    # left panel into the lower half of its height
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=False)
    x = np.arange(24)
    for ax, (title, want, gse_col) in zip(axes, panels):
        for i, g in enumerate(groups):
            if int(g) not in comp.index or bool(comp.loc[int(g), "residential"]) != want:
                continue
            ax.plot(x, hourly[i, ci] * 100, color=ddslp_color(g), lw=1.7,
                    label=f"{ddslp_label(g)} "
                          f"({comp.loc[int(g), 'share_domestic_pod']*100:.0f}% domestic)")
        if gse_col in sel.columns:
            #Lorenzo Giannuzzo: the published daily shape is the ratio of the sums over the
            # days of the cell, which is the shape of the energy the profile allocates to
            # that cell, and not the mean of daily ratios
            prof = sel.groupby("hour")[gse_col].sum()
            prof = prof / prof.sum() * 100
            ax.plot(x, prof.reindex(x).to_numpy(), lw=2.4, ls="--", color=INK,
                    label=GSE_NAME[gse_col])
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Time of day [h]")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of the daily energy [%]")
    for a in axes:
        lo, hi = a.get_ylim()
        a.set_ylim(0, hi * 1.05)
    fig.tight_layout()
    #Lorenzo Giannuzzo: the legends go under the panels, so that no curve runs behind them
    for a in axes:
        legend_below(a, fontsize=7.0, ncol=2, pad_in=0.55)
    save(fig, "fig2_daily_shapes")


# ------------------------------------------------------------------------ figure 3
def fig_b2_distributions() -> None:
    """The audit: how the per-user error is distributed, not just its median."""
    #Lorenzo Giannuzzo: POD codes contain an "E" and are read as floats unless forced to string:
    # 99999E00010600 is a valid float literal and silently becomes inf.
    b2 = pd.read_csv(RES / "b2_pod_month.csv", dtype={"pod": str})
    if "common_set" in b2.columns:
        b2 = b2[b2["common_set"]]
    order = ["GSE", "ARERA", "DDSLP"]
    labels = {"GSE": "GSE", "ARERA": "ARERA", "DDSLP": "Data-driven"}
    colors = {"GSE": WARM, "ARERA": NEUTRAL, "DDSLP": ACCENT}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))

    ax = axes[0]
    for s in order:
        v = np.sort(b2.loc[b2["source"] == s, "total_variation"].dropna().to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=colors[s], lw=1.8,
                label=labels[s])
    ax.set_xlabel("Total variation per user-month [-]")
    ax.set_ylabel("Cumulative share of user-months [-]")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    #Lorenzo Giannuzzo: at the top left, where the cumulative curves have not risen yet, one
    # entry per line
    legend_top(ax, ncol=1, loc="upper left")

    ax = axes[1]
    data = [b2.loc[b2["source"] == s, "total_variation"].dropna() for s in order]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for pc, s in zip(parts["bodies"], order):
        pc.set_facecolor(colors[s])
        pc.set_alpha(0.35)
        pc.set_edgecolor(colors[s])
    bp = ax.boxplot(data, widths=0.12, showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor=INK, linewidth=0.8)
    for med in bp["medians"]:
        med.set(color=INK, linewidth=1.4)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([labels[s] for s in order])
    ax.set_ylabel("Total variation per user-month [-]")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fig3_b2_distributions")


# ------------------------------------------------------------------------ figure 4
def sunday_ratio_per_pod() -> pd.Series:
    """Mean Sunday energy over mean working-day energy, per point of delivery.

    Sunday includes the national holidays, as in the day types of Section 2.2; the
    working day is Monday to Friday. Days at zero energy count, since a closure is a
    legitimate day of the week. A point needs at least one day of both kinds with a
    positive mean working-day energy for the ratio to be defined.
    """
    days = pd.read_parquet(CACHE / "days.parquet")
    #Lorenzo Giannuzzo: the population of the paper is the clustered one. A POD retained by
    # the completeness filter but without a single day carrying a shape never enters a
    # profile, and counting it here gave a histogram of more points than the paper has.
    uv = CACHE / "user_vectors.parquet"
    if uv.exists():
        days = days[days["pod"].isin(pd.read_parquet(uv, columns=["pod"])["pod"])]
    per = days.groupby(["pod", "daytype"])["energy"].mean().unstack()
    per = per.dropna(subset=["weekday", "sunday"])
    per = per[per["weekday"] > 0]
    return (per["sunday"] / per["weekday"]).replace([np.inf, -np.inf], np.nan).dropna()


def fig_daytype_energy() -> None:
    """The structural finding: what the national profiles do to the week.

    The median is taken over every point for which the ratio is defined; the horizontal
    axis is cut at 2.2 for legibility only.
    """
    ratio = sunday_ratio_per_pod()
    shown = ratio[ratio <= 2.2]

    g = pd.read_csv(RES / "gse_normalisation.csv")
    #Lorenzo Giannuzzo: only the single-rate profiles of the categories the audit covers. A
    # time-band profile is normalised within each band and has no Sunday to working-day
    # ratio of its own, so drawing one would plot an artefact of the normalisation.
    from comparison import GSE_PROFILES
    mono = g[g["profile"].isin(GSE_PROFILES) & g["profile"].str.endswith("M")]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.hist(shown, bins=np.linspace(0, 2.2, 67), color=ACCENT, alpha=0.5,
            edgecolor="none", label=f"Metered points of delivery ({len(ratio)} PODs)")
    ax.axvline(float(ratio.median()), color=ACCENT, lw=1.8,
               label=f"Median of the metered points ({ratio.median():.2f})")
    #Lorenzo Giannuzzo: profiles with the same ratio are drawn once with a joint label; two
    # lines on the same abscissa hid the dashed one under the solid one
    vals = mono["sunday_over_weekday_energy"].round(3)
    styles = ["-", "--"]
    for i, (v, grp) in enumerate(mono.groupby(vals, sort=True)):
        names = [GSE_NAME.get(p, p).replace("GSE ", "").replace(", single rate", "")
                 for p in grp["profile"]]
        label = ("GSE " + " and ".join(names) + ", single rate" if len(names) > 1
                 else GSE_NAME.get(grp["profile"].iloc[0], grp["profile"].iloc[0]))
        ax.axvline(float(v), color=WARM, lw=2.0, ls=styles[i % 2], label=f"{label} ({v:.3f})")
    ax.set_xlabel("Sunday energy over working-day energy [-]")
    ax.set_ylabel("Points of delivery [-]")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    #Lorenzo Giannuzzo: inside the axis on the right, over the thin tail of the distribution
    # and clear of the two vertical lines, one entry per line
    legend_top(ax, ncol=1, fontsize=7.2, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4_daytype_energy")


# ------------------------------------------------------------------------ figure 5
def fig_misallocated() -> None:
    """Eq. 13 by contractual power class, on the common set of user-months."""
    #Lorenzo Giannuzzo: POD codes contain an "E" and are read as floats unless forced to string:
    # 99999E00010600 is a valid float literal and silently becomes inf.
    b2 = pd.read_csv(RES / "b2_pod_month.csv", dtype={"pod": str})
    if "common_set" in b2.columns:
        b2 = b2[b2["common_set"]]
    users = pd.read_parquet(CACHE / "users.parquet")[["pod", "D_POTC"]]
    b2 = b2.merge(users, on="pod", how="left")
    b2["class"] = pd.cut(b2["D_POTC"], [0, 1.5, 3, 4.5, 6, np.inf],
                         labels=["0-1.5", "1.5-3", "3-4.5", "4.5-6", ">6"])

    order = ["GSE", "ARERA", "DDSLP"]
    labels = {"GSE": "GSE", "ARERA": "ARERA", "DDSLP": "Data-driven"}
    colors = {"GSE": WARM, "ARERA": NEUTRAL, "DDSLP": ACCENT}
    piv = (b2.groupby(["class", "source"], observed=True)["misallocated_kWh"]
             .sum().unstack().reindex(columns=order))
    share = (b2.groupby(["class", "source"], observed=True)
               .apply(lambda x: x["misallocated_kWh"].sum() / x["month_energy_kWh"].sum(),
                      include_groups=False)
               .unstack().reindex(columns=order))
    n_pods = b2.groupby("class", observed=True)["pod"].nunique().reindex(piv.index)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    w = 0.26
    xs = np.arange(len(piv.index))
    for k, s in enumerate(order):
        axes[0].bar(xs + (k - 1) * w, piv[s].to_numpy() / 1000, width=w,
                    color=colors[s], label=labels[s])
        axes[1].bar(xs + (k - 1) * w, share[s].to_numpy() * 100, width=w,
                    color=colors[s], label=labels[s])
    axes[0].set_ylabel("Misallocated energy [MWh]")
    axes[1].set_ylabel("Share of the energy misallocated [%]")
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c}\n({int(n)} PODs)" for c, n in zip(piv.index.astype(str), n_pods)])
        ax.set_xlabel("Contractual power class [kW]")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    legend_top(axes[0], ncol=3)
    legend_top(axes[1], ncol=3)
    fig.tight_layout()
    save(fig, "fig5_misallocated_energy")


REQUIRED = ("b1_ddslp_vs_national.csv", "b2_pod_month.csv", "gse_normalisation.csv")


def main(include_mapping: bool = True) -> None:
    print(f"\n{'='*78}\n  FIGURES, Section 2.5\n{'='*78}")
    missing = [f for f in REQUIRED if not (RES / f).exists()]
    if missing:
        #Lorenzo Giannuzzo: Reached when the stage is run before comparison has produced its tables. The
        # figures are a view on those tables and nothing here can be drawn without them.
        print(f"  comparison output not found in {RES}")
        print(f"  missing: {', '.join(missing)}")
        print("  run  python main.py --stage comparison  first\n")
        return
    table_b1_distance()
    fig_daily_shapes()
    fig_b2_distributions()
    fig_daytype_energy()
    fig_misallocated()
    print(f"\n  figures in {FIG}\n")

    #Lorenzo Giannuzzo: The stage is called "figures", so it draws every figure the pipeline has, not only
    # the ones belonging to Section 2.5. The mapping figures are skipped in silence when
    # that stage has not been run, which is the only case in which they cannot exist.
    if not include_mapping:
        #Lorenzo Giannuzzo: called from the comparison stage, which runs before the mapping.
        # Drawing the mapping figures there would draw them from the tables of the previous
        # run, which is what produced the missing m3_reach_pod.csv in the log.
        return
    try:
        import mapping_figures
        if mapping_figures.inputs_ready():
            mapping_figures.main()
        else:
            #Lorenzo Giannuzzo: Named rather than skipped in silence. The tables moved into one folder per
            # metric, and a wrong path here would otherwise look exactly like a mapping
            # stage that has not been run yet.
            print("  mapping figures skipped, missing: "
                  + ", ".join(mapping_figures.missing_inputs()))
    except Exception as exc:
        print(f"  ! mapping figures not produced: {type(exc).__name__}: {exc}")

    #Lorenzo Giannuzzo: the clustering figures are redrawn from its tables, so that a change of
    # style reaches them with the figures stage alone
    try:
        import clustering
        clustering.redraw_figures()
    except Exception as exc:
        print(f"  ! clustering figures not redrawn: {type(exc).__name__}: {exc}")

    #Lorenzo Giannuzzo: the generation figures likewise, from the tables of that stage
    try:
        import generation
        generation.redraw_figures()
    except Exception as exc:
        print(f"  ! generation figures not redrawn: {type(exc).__name__}: {exc}")

    collect_figures()


def collect_figures() -> None:
    """Copy every figure of the pipeline into paper_results/figures, one folder per format.

    Each stage keeps its figures next to its own tables, which is where they are checked
    against the numbers. The manuscript needs them all in one place, so they are also
    copied, flat, into paper_results/figures/png and paper_results/figures/pdf. The folder
    is emptied first, so that a figure no longer produced does not survive from an earlier
    run. Two figures with the same file name in different stages are prefixed with the
    stage folder, so that neither overwrites the other.
    """
    import shutil

    root = RES.parent
    target = root / "figures"
    if target.exists():
        shutil.rmtree(target)
    found: dict[str, list] = {}
    for ext in ("png", "pdf"):
        for p in sorted(root.rglob(f"*.{ext}")):
            if target in p.parents:
                continue
            found.setdefault(p.name, []).append(p)
    n = 0
    for name, paths in found.items():
        for p in paths:
            ext = p.suffix.lstrip(".")
            out = target / ext
            out.mkdir(parents=True, exist_ok=True)
            #Lorenzo Giannuzzo: the stage folder is the first component below paper_results
            stage = p.relative_to(root).parts[0]
            dest = out / (f"{stage}_{name}" if len(paths) > 1 else name)
            shutil.copy2(p, dest)
            n += 1
    print(f"  {n} figure files collected in {target}\n")

    #Lorenzo Giannuzzo: the tables of the paper likewise, from the tables folder of each stage
    tables = root / "tables"
    if tables.exists():
        shutil.rmtree(tables)
    m = 0
    for p in sorted(root.rglob("tables/*")):
        if tables in p.parents or p.suffix not in (".csv", ".xlsx"):
            continue
        tables.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, tables / p.name)
        m += 1
    if m:
        print(f"  {m} table files collected in {tables}\n")


if __name__ == "__main__":
    main()