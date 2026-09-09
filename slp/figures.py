"""Figures for the comparison stage (Section 2.5).

Run after `python main.py --stage comparison`:

    python figures.py

Every figure is written both as PNG at 300 dpi for the manuscript and as PDF for the
camera-ready version. Nothing is computed here that is not already in the comparison
outputs or in the cache, so the figures cannot disagree with the tables.
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

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RES = ROOT / "paper_results" / "comparison_results"
FIG = RES / "figures"

INK = "#1b1b1b"
GRID = "#d9d9d9"
ACCENT = "#0b3c5d"
WARM = "#c1440e"
NEUTRAL = "#7d8491"
PALETTE = ["#0b3c5d", "#328cc1", "#c1440e", "#e2a33c", "#4c8055", "#7d5ba6"]

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
    "legend.frameon": False,
    "figure.dpi": 120,
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  {name}")


# ------------------------------------------------------------------------ figure 1
def fig_b1_heatmap() -> None:
    """How far each national profile sits from each data-driven profile."""
    b1 = pd.read_csv(RES / "b1_ddslp_vs_national.csv")
    d = b1[b1["setting"] == "S1_monthly"]
    m = d.pivot_table(index="national", columns="ddslp", values="total_variation")
    m = m.loc[m.mean(axis=1).sort_values().index]
    order = sorted(m.columns, key=lambda c: int(c.split("_")[1]))
    m = m[order]

    fig, ax = plt.subplots(figsize=(5.6, 8.2))
    im = ax.imshow(m.to_numpy(), aspect="auto", cmap="RdYlBu_r", vmin=0,
                   vmax=float(np.nanpercentile(m.to_numpy(), 98)))
    ax.set_xticks(range(len(m.columns)))
    # Beyond about six columns the labels collide; rotating is cheaper than shortening
    # them, which would cost the reader the profile numbers.
    rot = 0 if len(m.columns) <= 6 else 45
    ax.set_xticklabels([c.replace("DDSLP_", "DD-SLP ") for c in m.columns],
                       rotation=rot, ha="center" if rot == 0 else "right", fontsize=8)
    ax.set_yticks(range(len(m.index)))
    ax.set_yticklabels(m.index, fontsize=7)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if v > np.nanpercentile(m.to_numpy(), 70) else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Total variation (share of energy misallocated)", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title("Distance between national and data-driven profiles\n"
                 "monthly setting, hourly resolution", loc="left", fontsize=9.5, pad=10)
    ax.set_xlabel("")
    save(fig, "fig1_b1_distance_heatmap")


# ------------------------------------------------------------------------ figure 2
def fig_daily_shapes(cell: str = "winter|weekday") -> None:
    """The shapes themselves, each family against the national profile meant for it.

    A data-driven profile is residential or not by who ended up in it, not by
    construction, so the split is read from the composition table the comparison stage
    writes. Putting a mostly residential archetype next to the profile for other uses
    would compare two objects the regulation never intends to meet.
    """
    arr = np.load(CACHE / "profiles.npy")
    w = pd.read_parquet(CACHE / "profile_weights.parquet")
    groups = sorted(w["group"].unique())
    cells = list(w.loc[w["group"] == groups[0], "cell"])
    ci = cells.index(cell)
    hourly = arr.reshape(arr.shape[0], arr.shape[1], 24, 4).sum(axis=3)

    comp = pd.read_csv(RES / "ddslp_composition.csv").set_index("group")
    gse = pd.read_parquet(CACHE / "gse.parquet")
    days = pd.read_parquet(CACHE / "days.parquet")
    cal = C.build_calendar(2025, C.season_map_from_days(days))
    season, daytype = cell.split("|")
    sel_days = cal[(cal["season"] == season) & (cal["daytype"] == daytype)]
    sel = gse[pd.to_datetime(dict(year=gse.year, month=gse.month, day=gse.day))
              .dt.date.isin(sel_days["date"].dt.date)]

    panels = [("Residential", True, "PDMM", "Domestic"),
              ("Other uses", False, "PAUM", "Other uses")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    x = np.arange(24)
    for ax, (title, want, gse_col, gse_label) in zip(axes, panels):
        k = 0
        for i, g in enumerate(groups):
            if bool(comp.loc[int(g), "residential"]) != want:
                continue
            ax.plot(x, hourly[i, ci] * 100, color=PALETTE[k % len(PALETTE)], lw=1.7,
                    label=f"DD-SLP {int(g)} "
                          f"({comp.loc[int(g), 'share_domestic_pod']*100:.0f}% domestic)")
            k += 1
        if gse_col in sel.columns:
            prof = sel.groupby("hour")[gse_col].mean()
            prof = prof / prof.sum() * 100
            ax.plot(x, prof.reindex(x).to_numpy(), lw=2.4, ls="--", color=INK,
                    label=f"GSE {gse_col}, {gse_label}")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("Hour")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.2)
    axes[0].set_ylabel("Share of daily energy (%)")
    fig.suptitle(f"Daily shapes, {season} {daytype}, each family against its own "
                 f"national profile", x=0.005, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig2_daily_shapes")


# ------------------------------------------------------------------------ figure 3
def fig_b2_distributions() -> None:
    """The audit: how the per-user error is distributed, not just its median."""
    # POD codes contain an "E" and are read as floats unless forced to string:
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
    ax.set_xlabel("Total variation per user-month")
    ax.set_ylabel("Cumulative share of user-months")
    ax.set_xlim(0, 0.8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    ax.set_title("Distribution of misallocation", loc="left", fontsize=9.5)

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
    ax.set_ylabel("Total variation")
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Same data, by profile family", loc="left", fontsize=9.5)

    fig.tight_layout()
    save(fig, "fig3_b2_distributions")


# ------------------------------------------------------------------------ figure 4
def fig_daytype_energy() -> None:
    """The structural finding: what the national profiles do to the week.

    For each profile family, the ratio between the mean daily energy of a Sunday and of a
    working day. The metered points are shown as a distribution, the national profiles as
    single values, because a profile has no distribution.
    """
    days = pd.read_parquet(CACHE / "days.parquet")
    d = days[days["energy"] > 0]
    per = (d.groupby(["pod", "daytype"])["energy"].mean().unstack())
    per = per.dropna(subset=["weekday", "sunday"])
    ratio = (per["sunday"] / per["weekday"]).replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio < 3]

    g = pd.read_csv(RES / "gse_normalisation.csv")
    mono = g[g["profile"].str.endswith("M")]
    band = g[g["profile"].str.endswith("F")]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.hist(ratio, bins=70, color=ACCENT, alpha=0.5, edgecolor="none",
            label=f"Metered points (n = {len(ratio):,})")
    ax.axvline(float(ratio.median()), color=ACCENT, lw=1.8,
               label=f"Median of metered points ({ratio.median():.2f})")
    ax.axvline(float(mono["sunday_over_weekday_energy"].mean()), color=WARM, lw=2.0,
               label=f"GSE, single-rate ({mono['sunday_over_weekday_energy'].mean():.3f})")
    for i, row in enumerate(band.itertuples()):
        ax.axvline(row.sunday_over_weekday_energy, color=NEUTRAL, lw=0.9, ls=":",
                   label="GSE, banded" if i == 0 else None)
    ax.set_xlabel("Sunday energy / working-day energy")
    ax.set_ylabel("Points of delivery")
    ax.set_xlim(0, 2.2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5)
    ax.set_title("How much the week is flattened\n"
                 "single-rate national profiles give a Sunday the energy of a Tuesday",
                 loc="left", fontsize=9.5, pad=8)
    fig.tight_layout()
    save(fig, "fig4_daytype_energy")


# ------------------------------------------------------------------------ figure 5
def fig_misallocated() -> None:
    """Eq. 15 and Eq. 16, by tariff category, on the common set of user-months."""
    # POD codes contain an "E" and are read as floats unless forced to string:
    # 99999E00010600 is a valid float literal and silently becomes inf.
    b2 = pd.read_csv(RES / "b2_pod_month.csv", dtype={"pod": str})
    if "common_set" in b2.columns:
        b2 = b2[b2["common_set"]]
    users = pd.read_parquet(CACHE / "users.parquet")[["pod", "D_TIPTA", "D_POTC"]]
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

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    w = 0.26
    xs = np.arange(len(piv.index))
    for k, s in enumerate(order):
        axes[0].bar(xs + (k - 1) * w, piv[s].to_numpy() / 1000, width=w,
                    color=colors[s], label=labels[s])
        axes[1].bar(xs + (k - 1) * w, share[s].to_numpy() * 100, width=w,
                    color=colors[s], label=labels[s])
    axes[0].set_ylabel("Misallocated energy (MWh)")
    axes[0].set_title("Eq. 15, absolute", loc="left", fontsize=9.5)
    axes[1].set_ylabel("Share of energy misallocated (%)")
    axes[1].set_title("Eq. 15, relative to consumption", loc="left", fontsize=9.5)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(piv.index.astype(str))
        ax.set_xlabel("Contractual power class (kW)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend()
    fig.tight_layout()
    save(fig, "fig5_misallocated_energy")


REQUIRED = ("b1_ddslp_vs_national.csv", "b2_pod_month.csv", "gse_normalisation.csv")


def main() -> None:
    print(f"\n{'='*78}\n  FIGURES, Section 2.5\n{'='*78}")
    missing = [f for f in REQUIRED if not (RES / f).exists()]
    if missing:
        # Reached when the stage is run before comparison has produced its tables. The
        # figures are a view on those tables and nothing here can be drawn without them.
        print(f"  comparison output not found in {RES}")
        print(f"  missing: {', '.join(missing)}")
        print("  run  python main.py --stage comparison  first\n")
        return
    fig_b1_heatmap()
    fig_daily_shapes()
    fig_b2_distributions()
    fig_daytype_energy()
    fig_misallocated()
    print(f"\n  figures in {FIG}\n")

    # The stage is called "figures", so it draws every figure the pipeline has, not only
    # the ones belonging to Section 2.5. The mapping figures are skipped in silence when
    # that stage has not been run, which is the only case in which they cannot exist.
    try:
        import mapping_figures
        if all((mapping_figures.RES / f).exists() for f in mapping_figures.REQUIRED):
            mapping_figures.main()
    except Exception as exc:
        print(f"  ! mapping figures not produced: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
