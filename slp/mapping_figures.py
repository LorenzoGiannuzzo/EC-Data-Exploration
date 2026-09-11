"""Figures for the mapping stage (Section 2.6).
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import INK, GRID, ACCENT, WARM, NEUTRAL, save as _save  # noqa: E402

from common import calendar as C  # noqa: E402
from common.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
_CFG = load_config()

#Lorenzo Giannuzzo: every path resolves through the configuration, so that renaming a
#results folder is one edit in config.py and not a hunt through five stages. The three
#mapping metrics read and write under their own sub-folder, matching the sections of the
#paper that report them.
RES = _CFG.results_dir("mapping")
PART_DIR = {p: _CFG.results_dir("mapping", p)
            for p in ("multiplicity", "aggregation", "coverage")}
GEN = _CFG.results_dir("generation")
CLU = _CFG.results_dir("clustering")
CACHE = _CFG.cache_dir


def save(fig, name: str, part: str | None = None) -> None:
    #Lorenzo Giannuzzo: a figure lands in the folder of the metric it illustrates, and the
    #ones built on the contingency table itself land at the root of the stage, because that
    #table is the common origin of all three metrics and belongs to none of them. Within
    #that folder each format has its own directory, so the two sets can be handled whole.
    base = (PART_DIR[part] if part else RES) / "figures"
    for ext in ("png", "pdf"):
        out = base / ext
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


RESIDUAL = "other (below n_min)"


def _order_rows(share: pd.DataFrame) -> pd.DataFrame:
    #Lorenzo Giannuzzo: rows ordered by the profile they mostly land in, and within
    #that by how concentrated they are. Alphabetical order scatters the structure
    #over the whole table, while this brings the blocks onto the diagonal and lets
    #the eye see which classes behave alike. The residual bucket is not a class and
    #is kept out of the ordering, at the bottom, so that it cannot head a ranking it
    #has no business heading.
    body = share.drop(index=[RESIDUAL], errors="ignore")
    key = pd.DataFrame({"dom": body.to_numpy().argmax(axis=1),
                        "conc": -body.to_numpy().max(axis=1)}, index=body.index)
    body = body.loc[key.sort_values(["dom", "conc"]).index]
    tail = share.loc[share.index.intersection([RESIDUAL])]
    return pd.concat([body, tail])


#Lorenzo Giannuzzo: the codes are what the data carries, but a reader of the paper
#should not have to hold the ATECO nomenclature in mind to read a legend. The names
#are short on purpose, since they are axis labels and not definitions, and anything
#the table does not cover falls back to the code itself rather than being dropped or
#guessed at.
ACTIVITY_NAME = {
    "DO": "Domestic",
    "CO": "Common areas",
    "IL": "Public lighting",
    "01": "Agriculture",
    "10": "Food manufacturing",
    "16": "Wood products",
    "22": "Rubber and plastics",
    "25": "Metal products",
    "33": "Machinery repair",
    "35": "Electricity and gas",
    "36": "Water supply",
    "38": "Waste management",
    "41": "Building construction",
    "43": "Specialised construction",
    "45": "Motor vehicle trade",
    "46": "Wholesale trade",
    "47": "Retail trade",
    "49": "Land transport",
    "52": "Warehousing",
    "55": "Accommodation",
    "56": "Food and beverage",
    "60": "Broadcasting",
    "61": "Telecommunications",
    "62": "IT services",
    "64": "Financial services",
    "68": "Real estate",
    "69": "Legal and accounting",
    "81": "Building services",
    "84": "Public administration",
    "85": "Education",
    "86": "Health services",
    "93": "Sport and recreation",
    "94": "Membership organisations",
    "96": "Personal services",
}
POOLED = "Other activities"


def activity_label(code: object) -> str:
    c = str(code)
    if c == RESIDUAL:
        return "Minor classes, pooled"
    if c == POOLED:
        return c
    return ACTIVITY_NAME.get(c, f"class {c}")


def fig_contingency() -> None:
    """The table both metrics are read from, as shares of each activity class."""
    ct = pd.read_csv(RES / "contingency_pod.csv", index_col=0)
    share = _order_rows(ct.div(ct.sum(axis=1), axis=0))

    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(share) + 1.8))
    im = ax.imshow(share.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(share.shape[1]))
    ax.set_xticklabels([c.replace("DDSLP_", "DD-SLP ") for c in share.columns])
    ax.set_yticks(range(share.shape[0]))
    ax.set_yticklabels([activity_label(c) for c in share.index], fontsize=7)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.iat[i, j]
            if v >= 0.02:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Share of the activity class [%]", fontsize=8)
    cb.outline.set_visible(False)
    if RESIDUAL in share.index:
        ax.get_yticklabels()[list(share.index).index(RESIDUAL)].set_color(NEUTRAL)
    ax.set_title("Where each activity class ends up\n"
                 "rows sum to 100 per cent, ordered by the profile they mostly land in",
                 loc="left", fontsize=9.5, pad=10)
    save(fig, "fig6_contingency")


def fig_multiplicity() -> None:
    """M1: how many profiles each activity class spans, with bootstrap intervals."""
    d = pd.read_csv(PART_DIR["multiplicity"] / "m1_multiplicity_pod.csv").sort_values("M1_effective")
    e = pd.read_csv(PART_DIR["multiplicity"] / "m1_multiplicity_energy.csv").set_index("activity")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(7.2, 0.30 * len(d) + 1.8))
    ax.hlines(y, d["M1_ci_low"], d["M1_ci_high"], color=NEUTRAL, lw=1.4, zorder=1)
    ax.scatter(d["M1_effective"], y, s=34, color=ACCENT, zorder=3,
               label="Weighted by points")
    ax.scatter(e.reindex(d["activity"])["M1_effective"], y, s=30, facecolor="white",
               edgecolor=WARM, lw=1.3, zorder=4, label="Weighted by energy")
    ax.axvline(1, color=INK, lw=1.0, ls="--")
    ax.text(1.02, len(d) - 0.4, "one profile per class,\nwhat the classification assumes",
            fontsize=7, va="top", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([activity_label(a) for a in d["activity"]], fontsize=7.5)
    ax.set_xlabel("Effective number of profiles the class spans")
    ax.set_xlim(0.8, None)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("M1, class multiplicity\n"
                 "a class above one is not predicted by its activity code",
                 loc="left", fontsize=9.5, pad=8)
    save(fig, "fig7_m1_multiplicity", "multiplicity")


def fig_aggregation() -> None:
    """M2: how many activity classes each profile subsumes."""
    d = pd.read_csv(PART_DIR["aggregation"] / "m2_aggregation_pod.csv").sort_values("M2_effective")
    e = pd.read_csv(PART_DIR["aggregation"] / "m2_aggregation_energy.csv").set_index("profile")
    y = np.arange(len(d))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.hlines(y, d["M2_ci_low"], d["M2_ci_high"], color=NEUTRAL, lw=1.6, zorder=1)
    ax.scatter(d["M2_effective"], y, s=44, color=ACCENT, zorder=3,
               label="Weighted by points")
    ax.scatter(e.reindex(d["profile"])["M2_effective"], y, s=40, facecolor="white",
               edgecolor=WARM, lw=1.4, zorder=4, label="Weighted by energy")
    ax.axvline(1, color=INK, lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("DDSLP_", "DD-SLP ") for p in d["profile"]])
    ax.set_xlabel("Effective number of activity classes subsumed")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("M2, profile aggregation", loc="left", fontsize=9.5)

    ax = axes[1]
    ax.barh(y, d["n_classes_present"], color=NEUTRAL, alpha=0.45, label="Classes present")
    ax.barh(y, d["M2_hard80"], color=ACCENT, height=0.55,
            label="Classes covering 80% of the profile")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Number of activity classes")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Raw count against effective concentration", loc="left", fontsize=9.5)

    fig.tight_layout()
    save(fig, "fig8_m2_aggregation", "aggregation")


def fig_coverage() -> None:
    """M3: what a published profile declares to represent, against what it does.

    The regulation assigns one national profile per tariff category, which is a claim that
    the category is homogeneous enough to be described by one curve. The figure puts that
    claim, the dashed line at one, next to two measurements: how many activity classes the
    profile is actually applied to, and how many distinct consumption behaviours are
    actually hiding under it. The second is the one that matters, because a profile may
    legitimately span many activity classes if they all consume alike.
    """
    d = pd.read_csv(PART_DIR["coverage"] / "m3_coverage_pod.csv").sort_values("M3_ddslp_effective")
    y = np.arange(len(d))
    off = 0.16

    fig, ax = plt.subplots(figsize=(7.8, 0.95 * len(d) + 1.9))
    ax.axvline(1, color=INK, lw=1.2, ls="--", zorder=0)
    ax.text(1.06, len(d) - 0.35, "declared: one profile\nstands for the category",
            fontsize=7.5, va="top", color=INK)

    ax.hlines(y + off, d["M3_ci_low"], d["M3_ci_high"], color=NEUTRAL, lw=1.6, zorder=1)
    ax.scatter(d["M3_classes_effective"], y + off, s=52, color=NEUTRAL,
               edgecolor=INK, lw=0.6, zorder=3, label="Activity classes represented")
    ax.hlines(y - off, d["M3_ddslp_ci_low"], d["M3_ddslp_ci_high"], color=WARM,
              lw=1.6, alpha=0.5, zorder=1)
    ax.scatter(d["M3_ddslp_effective"], y - off, s=58, color=WARM, zorder=3,
               label="Consumption behaviours subsumed")

    for i, r in enumerate(d.itertuples()):
        cls = "class" if r.n_classes_present == 1 else "classes"
        ax.annotate(f"{r.n_classes_present} {cls}, {r.n_ddslp_present} profiles, "
                    f"{r.n_pod} points",
                    xy=(0.995, i + 0.42), xycoords=("axes fraction", "data"),
                    ha="right", va="center", fontsize=7, color=NEUTRAL)

    ax.set_yticks(y)
    ax.set_yticklabels(d["national_profile"], fontsize=9)
    ax.set_ylim(-0.7, len(d) - 0.1)
    ax.set_xlabel("Effective number, against the one the regulation assumes")
    ax.set_xlim(0.6, None)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("M3, declared coverage against real coverage",
                 loc="left", fontsize=10, pad=8)
    save(fig, "fig9_m3_coverage", "coverage")


#Lorenzo Giannuzzo: the required inputs now live in different folders, so the check
#carries the folder with the name. Checking them all against the stage root would report
#every file as missing the moment the tables moved into their metric sub-folders.
REQUIRED = ((RES, "contingency_pod.csv"),
            (PART_DIR["multiplicity"], "m1_multiplicity_pod.csv"),
            (PART_DIR["aggregation"], "m2_aggregation_pod.csv"),
            (PART_DIR["coverage"], "m3_coverage_pod.csv"))


def missing_inputs() -> list[str]:
    #Lorenzo Giannuzzo: the check lives here and is called from the figures stage rather
    #than reimplemented there. The tables sit in three different folders now, so a caller
    #joining REQUIRED onto a single root gets it wrong, and it did.
    return [str((d / f).relative_to(RES.parents[1]))
            for d, f in REQUIRED if not (d / f).exists()]


def inputs_ready() -> bool:
    return not missing_inputs()


def main() -> None:
    print(f"\n{'='*78}\n  FIGURES, Section 2.6\n{'='*78}")
    missing = missing_inputs()
    if missing:
        print(f"  mapping output not found in {RES}")
        print(f"  missing: {', '.join(missing)}")
        print("  run  python main.py --stage mapping  first\n")
        return
    #Lorenzo Giannuzzo: four figures instead of seven. The stacked-bar renderings of
    #M1 and M2 carried the same numbers as the contingency table and the dot plots
    #they sat next to, and fig9 was fig12 without the filter on the categories built
    #on a handful of points. What the set was missing was not another count but a
    #curve, which is what the last figure supplies.
    fig_contingency()
    fig_multiplicity()
    fig_aggregation()
    fig_coverage_gap()
    try:
        fig_behaviours_under_national("PDMM")
    except FileNotFoundError as exc:
        print(f"  ! fig13 needs the generation output: {exc}")
    fig_declared_against_actual()
    fig_reach_beyond_declared()
    #Lorenzo Giannuzzo: one figure per day type. The published profiles are paired with a
    #behaviour cell by cell, so a catalogue curve that matches on a working day and misses
    #on a Sunday shows up only if the Sunday is drawn.
    for daytype in DAYTYPES:
        fig_profiles_with_classes(daytype=daytype)
    fig_classes_across_profiles()
    fig_national_without_match()
    print(f"\n  figures under {RES}\n")


if __name__ == "__main__":
    main()


# ===========================================================================
#  Alternative renderings of M1, M2 and M3.
#
#  The three above answer the question with an estimate and an interval, which
#  is what a table wants. These three show the composition the estimate is a
#  summary of, which is what a reader wants: the spread is visible directly
#  rather than encoded in a number.
# ===========================================================================
SPREAD = ["#0b3c5d", "#328cc1", "#7fb2d4", "#e2a33c", "#c1440e", "#7d5ba6"]


def fig_multiplicity_composition() -> None:
    """M1 seen as composition: where each activity class actually goes."""
    ct = pd.read_csv(RES / "contingency_pod.csv", index_col=0)
    m1 = pd.read_csv(PART_DIR["multiplicity"] / "m1_multiplicity_pod.csv").set_index("activity")
    share = ct.div(ct.sum(axis=1), axis=0)
    order = m1.reindex(share.index)["M1_effective"].sort_values()
    share = share.loc[order.index]

    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(share) + 2.0))
    left = np.zeros(len(share))
    y = np.arange(len(share))
    for j, col in enumerate(share.columns):
        v = share[col].to_numpy()
        ax.barh(y, v, left=left, height=0.72, color=SPREAD[j % len(SPREAD)],
                edgecolor="white", lw=0.6,
                label=col.replace("DDSLP_", "DD-SLP "))
        for i, (vi, li) in enumerate(zip(v, left)):
            if vi >= 0.10:
                ax.text(li + vi / 2, i, f"{vi*100:.0f}", ha="center", va="center",
                        fontsize=6.8, color="white" if j in (0, 4, 5) else INK)
        left += v
    for i, cls in enumerate(share.index):
        ax.text(1.015, i, f"{order.iloc[i]:.1f}", va="center", fontsize=8,
                color=INK, fontweight="bold")
    ax.text(1.015, len(share) - 0.2, "M1", va="bottom", fontsize=8, color=INK,
            fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(share.index, fontsize=8)
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Share of the activity class")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(sp == "bottom")
    ax.tick_params(axis="y", length=0)
    fig.legend(*ax.get_legend_handles_labels(), ncol=6, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.10 + 1.2 / len(share))
    ax.set_title("M1, where each activity class actually goes\n"
                 "a class that its code predicted would be one solid bar",
                 loc="left", fontsize=10, pad=10)
    save(fig, "fig10_m1_composition", "multiplicity")


def fig_aggregation_composition(top: int = 6) -> None:
    """M2 seen as composition: what each profile is actually made of, by energy."""
    ct = pd.read_csv(RES / "contingency_energy.csv", index_col=0).fillna(0.0)
    m2 = pd.read_csv(PART_DIR["aggregation"] / "m2_aggregation_energy.csv").set_index("profile")
    share = ct.div(ct.sum(axis=0), axis=1)                      # columns sum to 1
    big = share.sum(axis=1).sort_values(ascending=False).index[:top]
    keep = share.loc[big]
    keep.loc[POOLED] = share.drop(index=big).sum()

    cols = [c for c in ct.columns]
    order = m2.reindex(cols)["M2_effective"].sort_values()
    cols = list(order.index)
    y = np.arange(len(cols))
    palette = SPREAD + [NEUTRAL]

    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(cols) + 2.2))
    left = np.zeros(len(cols))
    for j, cls in enumerate(keep.index):
        v = keep.loc[cls, cols].to_numpy(dtype=float)
        ax.barh(y, v, left=left, height=0.62, color=palette[j % len(palette)],
                edgecolor="white", lw=0.6, label=str(cls))
        for i, (vi, li) in enumerate(zip(v, left)):
            if vi >= 0.07:
                ax.text(li + vi / 2, i, f"{vi*100:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if j in (0, 4, 5) else INK)
        left += v
    for i, c in enumerate(cols):
        ax.text(1.015, i, f"{order.iloc[i]:.1f}", va="center", fontsize=8.5,
                color=INK, fontweight="bold")
    ax.text(1.015, len(cols) - 0.35, "M2", va="bottom", fontsize=8.5, color=INK,
            fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("DDSLP_", "DD-SLP ") for c in cols], fontsize=9)
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Share of the profile's energy")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(*ax.get_legend_handles_labels(), ncol=4, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               title="Activity class", title_fontsize=7.5)
    fig.subplots_adjust(bottom=0.16 + 0.9 / len(cols))
    ax.set_title("M2, what each data-driven profile is made of\n"
                 "weighted by energy, so the count of small points cannot dominate",
                 loc="left", fontsize=10, pad=10)
    save(fig, "fig11_m2_composition", "aggregation")


def fig_coverage_gap() -> None:
    """M3 as the gap it is: one profile declared, several behaviours delivered."""
    d = pd.read_csv(PART_DIR["coverage"] / "m3_coverage_pod.csv").sort_values("M3_ddslp_effective")
    d = d[d["n_pod"] >= 50]
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(d) + 2.2))
    for i, r in enumerate(d.itertuples()):
        ax.plot([1, r.M3_ddslp_effective], [i, i], color=WARM, lw=6, alpha=0.30,
                solid_capstyle="butt", zorder=1)
        ax.annotate("", xy=(r.M3_ddslp_effective, i), xytext=(1, i),
                    arrowprops=dict(arrowstyle="-|>", color=WARM, lw=1.6,
                                    shrinkA=0, shrinkB=0), zorder=3)
        ax.text(r.M3_ddslp_effective + 0.10, i, f"{r.M3_ddslp_effective:.1f}",
                va="center", fontsize=8.5, color=WARM, fontweight="bold")
        #Lorenzo Giannuzzo: the grey marker sits exactly on the declared line for
        #almost every ARERA category, because those categories contain domestic
        #points only. Drawn there it says nothing and costs half the legend, so it
        #is shown only where the profile does span more than one activity class.
        if r.M3_classes_effective > 1.05:
            ax.scatter([r.M3_classes_effective], [i], s=30, facecolor="white",
                       edgecolor=NEUTRAL, lw=1.3, zorder=4)
    ax.axvline(1, color=INK, lw=1.4)
    #Lorenzo Giannuzzo: anchored below the top row rather than above it. Placed
    #above, it ran into the title, which is what made the published version of this
    #figure unreadable at the top.
    ax.text(1.04, len(d) - 0.72,
            "what the regulation declares:\none profile for the whole category",
            fontsize=8, va="top", color=INK)
    ax.scatter([], [], s=30, facecolor="white", edgecolor=NEUTRAL, lw=1.3,
               label="Activity classes represented (shown where above one)")
    ax.plot([], [], color=WARM, lw=3, label="Consumption behaviours delivered")
    ax.set_yticks(y)
    ax.set_yticklabels(d["national_profile"], fontsize=8.5)
    ax.set_xlabel("Effective number of categories [-]")
    ax.set_xlim(0.7, float(d[["M3_ddslp_effective", "M3_classes_effective"]]
                           .to_numpy().max()) + 0.9)
    ax.set_ylim(-0.8, len(d) - 0.1)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("M3, the distance between what is declared and what is delivered",
                 loc="left", fontsize=10, pad=10)
    save(fig, "fig12_m3_gap", "coverage")


# ===========================================================================
#  What a published profile actually stands for.
#
#  M1, M2 and M3 count labels. They establish that a national profile is
#  applied to several distinct behaviours, but a reader who has followed the
#  argument still has not seen a single curve, and the question in the title of
#  the paper is about curves. This figure closes that gap: the one curve the
#  regulation applies to a category, drawn over the behaviours that sit beneath
#  it, each as wide as the share of the category it carries.
# ===========================================================================
#Lorenzo Giannuzzo: GEN and CACHE are defined once at the top of the module. They were
#redefined here as well, and since this block runs later it quietly won, sending the
#figure back to the folder name the results tree no longer uses.
SEASON_LABEL = {"mid": "Autumn/Spring"}


def _pretty(word: object) -> str:
    w = str(word)
    return SEASON_LABEL.get(w.lower(), w[:1].upper() + w[1:] if w else w)


def _spread_labels(values: np.ndarray, gap: float) -> np.ndarray:
    #Lorenzo Giannuzzo: labels placed at the height of the curve they name overlap
    #wherever two curves end close together, which on a domestic category is most of
    #them. A forward pass pushes them apart by the minimum gap and a backward pass
    #recentres the block on where it started, so the labels keep the order of the
    #curves and stay near them instead of drifting upward as a group.
    order = np.argsort(values)
    seq = np.asarray(values, float)[order].copy()
    for i in range(1, len(seq)):
        seq[i] = max(seq[i], seq[i - 1] + gap)
    drift = seq.mean() - np.mean(values)
    seq -= drift
    out = np.empty_like(seq)
    out[order] = seq
    return out


#Lorenzo Giannuzzo: the constituents are drawn in colours that stay clear of the ink
#the published curve uses, otherwise the one line the figure is about is the hardest
#to pick out of the bundle.
UNDER = ["#328cc1", "#e2a33c", "#c1440e", "#7d5ba6", "#4c9a6a", "#a8577e"]


def _national_membership(national: str) -> pd.Series:
    """The points a published profile is applied to, and the group each one is in."""
    from common import assignment

    users = pd.read_parquet(CACHE / "users.parquet")
    groups = pd.read_parquet(CACHE / "groups.parquet")

    g = assignment.gse_profile(users)[["pod", "gse_column"]]
    g = g.rename(columns={"gse_column": "national"})
    a = assignment.arera_key(users)
    a = a[a["arera_applicable"]].copy()
    a["national"] = ("ARERA " + a["arera_class"].astype(str) + " "
                     + a["arera_residency"].astype(str))
    both = pd.concat([g[["pod", "national"]], a[["pod", "national"]]], ignore_index=True)

    m = groups.merge(both[both["national"] == national], on="pod", how="inner")
    m = m[~m["below_n_min"]] if "below_n_min" in m else m
    return m.groupby("group").size()


def _group_composition() -> pd.DataFrame:
    """Share of each activity class within every behaviour, over all its members.

    Taken over all the members and not only over those the published profile is applied
    to, because the question the panels answer is what a behaviour is made of, and a
    behaviour exists independently of which national profile happens to reach it.
    """
    users = pd.read_parquet(CACHE / "users.parquet")
    groups = pd.read_parquet(CACHE / "groups.parquet")
    full = groups.merge(users[["pod", "ateco_l1"]].rename(columns={"ateco_l1": "activity"}),
                        on="pod", how="left")
    full = full[full["activity"].notna()]
    if "below_n_min" in full:
        full = full[~full["below_n_min"]]
    comp = pd.crosstab(full["group"], full["activity"])
    return comp.div(comp.sum(axis=1), axis=0)


def _draw_composition_panels(fig, gs, share: pd.Series, palette: dict,
                             top: int = 8, floor: float = 0.1) -> None:
    #Lorenzo Giannuzzo: one small panel per behaviour, bars rather than pies. A pie of
    #ten or more classes turns into a ring of slivers, and two pies side by side cannot
    #be compared at a glance; bars on a shared axis can.
    #
    #The axis is logarithmic because these behaviours are between ninety and ninety-six
    #per cent domestic, and on a linear axis every other class collapses onto the origin
    #and the panel says nothing beyond what the dominant bar already said. The classes
    #that make a behaviour interesting are precisely the ones at one or two per cent, so
    #they have to be legible next to the ninety-five without being inflated into looking
    #comparable to it.
    comp = _group_composition()
    order = list(share.index)
    axes = [fig.add_subplot(gs[0, j]) for j in range(len(order))]

    for ax, g in zip(axes, order):
        row = pd.Series(dtype=float)
        if g in comp.index:
            r = comp.loc[g] * 100.0
            row = r[r >= floor].sort_values(ascending=False).head(top)
        y = np.arange(len(row))[::-1]
        ax.barh(y, row.to_numpy(), height=0.72, left=floor,
                color=palette[g], edgecolor="white", lw=0.4)
        for yy, v in zip(y, row.to_numpy()):
            ax.text(v * 1.25, yy, f"{v:.0f}" if v >= 1 else f"{v:.1f}",
                    va="center", fontsize=6.2, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(floor, 260)
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_xticklabels(["0.1", "1", "10", "100"])
        ax.set_yticks(y)
        ax.set_yticklabels([activity_label(c) for c in row.index], fontsize=6.4)
        ax.set_ylim(-0.7, max(len(row) - 0.3, 0.5))
        ax.tick_params(axis="x", labelsize=6.4)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"DD-SLP {g}  ({share[g]*100:.0f}%)", fontsize=7.6,
                     color=palette[g], fontweight="bold", pad=4)
    fig.text(0.5, 0.012,
             "Share of the points in each behaviour [%], logarithmic scale",
             ha="center", fontsize=8.2, color=INK)



def fig_behaviours_under_national(national: str = "PDMM",
                                  daytype: str = "weekday") -> None:
    """The curves a single published profile is standing in for.

    The published curve is reconstructed as the average of the behaviours beneath it,
    weighted by the points each carries, which is the curve a single profile for the
    category can at best be. Drawn against its own constituents it shows what the
    effective numbers of M3 mean in kilowatts: the category average passes through the
    middle of behaviours that are nowhere near it, and no user is described by it.
    """
    curves = pd.read_csv(GEN / "profiles.csv")
    counts = _national_membership(national)
    counts = counts[counts.index.isin(curves["profile"].unique())]
    if counts.empty:
        print(f"  ! no points under {national}, figure skipped")
        return
    share = (counts / counts.sum()).sort_values(ascending=False)
    effective = float(np.exp(-(share * np.log(share)).sum()))

    kw = [f"kW{i}" for i in range(1, 97)]
    seasons = [s for s in ("winter", "mid", "summer")
               if (curves["season"] == s).any()] or list(curves["season"].unique())
    x = np.arange(96) / 4.0
    palette = {g: UNDER[i % len(UNDER)] for i, g in enumerate(share.index)}

    #Lorenzo Giannuzzo: the descriptor box is given its own height in inches on top of
    #the panels rather than a slice of them. Taking the space out of the panels flattens
    #the curves and, because the labels on the right are spaced in data units, pushes
    #them back on top of one another as soon as the axes get short.
    panel_h = 2.55
    fig_h = 4.2 + panel_h
    fig = plt.figure(figsize=(4.15 * len(seasons), fig_h))
    outer = fig.add_gridspec(2, 1, height_ratios=[4.2, panel_h],
                             hspace=0.46, left=0.075, right=0.93,
                             top=1.0 - 0.72 / fig_h, bottom=0.10)
    top_gs = outer[0].subgridspec(1, len(seasons), wspace=0.12)
    axes = [fig.add_subplot(top_gs[0, j]) for j in range(len(seasons))]
    for a in axes[1:]:
        a.sharey(axes[0])
        a.tick_params(labelleft=False)
    for j, s in enumerate(seasons):
        ax = axes[j]
        sub = curves[(curves["season"] == s) & (curves["daytype"] == daytype)
                     & (curves["profile"].isin(share.index))]
        y = {int(r.profile): sub.loc[r.Index, kw].to_numpy(float)
             for r in sub.itertuples()}
        if not y:
            continue
        stack = np.vstack([y[g] for g in y])
        national_curve = sum(y[g] * share[g] for g in y)

        ax.fill_between(x, stack.min(axis=0), stack.max(axis=0), color=INK,
                        alpha=0.06, lw=0, zorder=1)
        for g, c in y.items():
            ax.plot(x, c, color=palette[g], lw=0.9 + 3.4 * share[g], alpha=0.95, zorder=2)
        ax.plot(x, national_curve, color=INK, lw=2.6, zorder=4)

        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(_pretty(s), fontsize=10)
        ax.set_xlabel("Time of day [h]")

        if j == len(seasons) - 1:
            names = list(y) + ["national"]
            ends = np.array([y[g][-1] for g in y] + [national_curve[-1]])
            #Lorenzo Giannuzzo: the gap is a share of the axis, not of the ends, so it is the
            # same on every panel whatever the curves happen to do at midnight
            span = float(np.ptp(ax.get_ylim()))
            placed = _spread_labels(ends, 0.052 * span)
            for name, pos in zip(names, placed):
                if name == "national":
                    ax.annotate(national, xy=(24.3, pos), fontsize=8, color=INK,
                                va="center", fontweight="bold", annotation_clip=False)
                else:
                    ax.annotate(f"DD-SLP {name} ({share[name]*100:.0f}%)",
                                xy=(24.3, pos), fontsize=7.5, color=palette[name],
                                va="center", annotation_clip=False)

    axes[0].set_ylabel("Power normalized to an annual\nconsumption of 1,000 kWh [kW]",
                       fontsize=9)
    axes[0].plot([], [], color=INK, lw=2.6, label=national)
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=1.0)

    #Lorenzo Giannuzzo: without this box the reader is looking at five coloured lines
    #with numbers on them and no way to tell what any of them is. The descriptors are
    #measured on the curves themselves, so the box cannot fall out of step with what
    #is drawn above it, and they are quantities rather than names because naming a
    #behaviour is an interpretation and belongs to the text.
    fig.suptitle(f"{national}: one published curve, {effective:.0f} effective "
                 f"behaviours beneath it, {int(counts.sum())} points",
                 fontsize=10.5, x=0.012, y=1.0 - 0.24 / fig_h, ha="left")
    bottom_gs = outer[1].subgridspec(1, len(share), wspace=0.75)
    _draw_composition_panels(fig, bottom_gs, share, palette)
    save(fig, "fig13_behaviours_under_national", "coverage")


def _describe_behaviours(curves: pd.DataFrame, share: pd.Series) -> pd.DataFrame:
    """One row per behaviour, with the quantities the table below the panels shows.

    Everything here is measured on the curves that are drawn above, except the annual
    consumption, which the profiles cannot carry because they are normalised. Naming a
    behaviour would be an interpretation and is left to the text; what the figure owes
    the reader is the numbers the interpretation would rest on.
    """
    kw = [f"kW{i}" for i in range(1, 97)]
    path = CLU / "groups.csv"
    real = pd.read_csv(path).set_index("group") if path.exists() else None

    rows = []
    for g in share.index:
        sub = curves[curves["profile"] == g]
        if sub.empty:
            continue
        w = sub["weight"].to_numpy(float)
        annual = (sub[kw].to_numpy(float) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)
        peak = float(annual.max() / annual.mean()) if annual.mean() > 0 else np.nan

        day = sub.set_index(["season", "daytype"])["kWh_per_day"]
        wd = day.xs("weekday", level="daytype").mean()
        we = day[day.index.get_level_values("daytype") != "weekday"].mean()
        seasons_here = day.index.get_level_values("season")
        wint = day.xs("winter", level="season").mean() if "winter" in seasons_here else np.nan
        summ = day.xs("summer", level="season").mean() if "summer" in seasons_here else np.nan

        rows.append({
            "group": g,
            "share": share[g] * 100,
            "annual": (float(real.loc[g, "E"])
                       if real is not None and g in real.index and "E" in real.columns
                       else np.nan),
            "peak": peak,
            "rest": wd / we if we else np.nan,
            "tilt": wint / summ if summ else np.nan,
        })
    return pd.DataFrame(rows)


#Lorenzo Giannuzzo: the header of the first numeric column names the profile itself,
#because "share of the category" left the reader to work out which category and a share
#of what. It reads as the share of the points that profile is applied to, which is the
#quantity the bar widths in the panels above are drawn from.
TABLE_COLUMNS = [
    ("Behaviour", "", 0.000, "left"),
    ("Share of the points\n{national} is applied to [%]", "share", 0.335, "right"),
    ("Mean annual\nconsumption [kWh]", "annual", 0.530, "right"),
    ("Peak over\nmean power [-]", "peak", 0.700, "right"),
    ("Weekday over\nweekend energy [-]", "rest", 0.865, "right"),
    ("Winter over\nsummer energy [-]", "tilt", 1.000, "right"),
]

#Lorenzo Giannuzzo: the table is laid out in row units rather than in figure fractions,
#so the gaps between the title, the header, the rule and the first row stay the same
#whatever the number of behaviours. Fractions were what put the title on top of the
#border the first time round.
ROW_UNIT_IN = 0.20
TITLE_ROWS, HEADER_ROWS, RULE_GAP, BOTTOM_ROWS = 1.15, 1.35, 0.45, 0.55


def _table_height(n_rows: int) -> float:
    """Height in inches the table needs, title and margins included."""
    return ROW_UNIT_IN * (TITLE_ROWS + HEADER_ROWS + RULE_GAP + n_rows + BOTTOM_ROWS)


def _draw_behaviour_table(fig, table: pd.DataFrame, palette: dict,
                          national: str, height: float, caption: str) -> None:
    n = len(table)
    total = TITLE_ROWS + HEADER_ROWS + RULE_GAP + n + BOTTOM_ROWS

    #Lorenzo Giannuzzo: centred on the figure, and no wider than the panels above it
    width = 0.86
    ax = fig.add_axes([(1.0 - width) / 2.0, 0.012, width, max(height - 0.03, 0.05)])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total)

    ax.add_patch(plt.Rectangle((-0.022, 0.0), 1.044, total, fill=False,
                               edgecolor="black", lw=0.9, clip_on=False))

    y_title = total - TITLE_ROWS / 2.0
    y_header = total - TITLE_ROWS - HEADER_ROWS / 2.0
    y_rule = total - TITLE_ROWS - HEADER_ROWS - RULE_GAP / 2.0

    ax.text(0.0, y_title, f"What sits beneath the {caption} curve",
            fontsize=8.8, fontweight="bold", color=INK, va="center", ha="left")
    for label, _key, x, align in TABLE_COLUMNS:
        ax.text(x, y_header, label.format(national=national), fontsize=7.3,
                color=INK, ha=align, va="center", linespacing=1.4)
    ax.plot([-0.022, 1.022], [y_rule, y_rule], color="black", lw=0.7, clip_on=False)

    for i, r in enumerate(table.itertuples()):
        y = y_rule - RULE_GAP / 2.0 - 0.5 - i
        ax.plot([0.000, 0.038], [y, y], color=palette[r.group], lw=3.0,
                solid_capstyle="round")
        ax.text(0.050, y, f"DD-SLP {r.group}", fontsize=7.8, color=INK, va="center")
        for _label, key, x, align in TABLE_COLUMNS[1:]:
            v = getattr(r, key)
            txt = ("n/a" if not np.isfinite(v)
                   else f"{v:.0f}" if key in ("share", "annual") else f"{v:.2f}")
            ax.text(x, y, txt, fontsize=7.8, color=INK, ha=align, va="center")



# ===========================================================================
#  What a published profile says it covers, against who it is applied to.
#
#  A national profile carries a declared meaning: domestic, other uses, a band
#  of contractual power. That label is the only thing a user of the profile
#  sees. This figure puts the label beside the activity classes the profile is
#  actually applied to, so that the effective numbers of M3 acquire names: the
#  curve published for "other uses" is being applied to retail, hospitality,
#  telecommunications and manufacturing at once, and the curve published for
#  the domestic category is applied to points that are not domestic.
# ===========================================================================
DECLARED = {
    "PDMM": "domestic, single rate",
    "PDMF": "domestic, time bands",
    "PAUM": "other uses, single rate",
    "PAUF": "other uses, time bands",
}


#Lorenzo Giannuzzo: the ARERA workbooks spell residency in Italian and the label is built
#straight from them, so the words reached the figures untranslated. They are mapped here,
#at drawing time only: the keys the pipeline matches on stay exactly as the source writes
#them, and nothing upstream has to know about the English.
RESIDENCY_EN = {"residente": "resident", "non residente": "non-resident",
                "tutti": "all"}


SEP = "  -  "

#Lorenzo Giannuzzo: one colour per publishing body, so that a reader who has seen two
#figures knows which catalogue a dashed curve belongs to before reading its label. The
#green is dark enough to stay distinct from the behaviour ink in greyscale print, which
#the accent blues of the palette would not.
GSE_GREEN = "#2e7d4f"
ARERA_RED = WARM


def _national_color(name: str) -> str:
    return ARERA_RED if str(name).upper().startswith("ARERA") else GSE_GREEN


def _national_caption(name: str) -> str:
    """How a published profile is named in a legend: code, and meaning when opaque."""
    meaning = _declared_meaning(name)
    self_describing = str(name).upper().startswith("ARERA")
    return _english_residency(str(name) if self_describing or not meaning
                              else f"{name}{SEP}{meaning}")


def _english_residency(text: str) -> str:
    out = str(text)
    for it, en in sorted(RESIDENCY_EN.items(), key=lambda kv: -len(kv[0])):
        for form in (it, it.title(), it.capitalize()):
            out = out.replace(form, en)
    #Lorenzo Giannuzzo: the power class arrives as a bare interval, "0-1.5", which reads
    #as a range of nothing in particular. The unit is appended here rather than written
    #into the label upstream, because that label is also the key the ARERA table is
    #matched on and it has to keep the spelling of the source.
    parts = out.split()
    if len(parts) >= 2 and parts[0].upper() == "ARERA" and any(c.isdigit() for c in parts[1]):
        parts.insert(2, "kW")
        out = " ".join(parts)
    return out


def _declared_meaning(name: str) -> str:
    #Lorenzo Giannuzzo: the label may arrive prefixed, as "GSE PAUF", so the published
    #code is looked for among its tokens. Matching the label whole returned nothing and
    #the figure then said only "PAUF", which is exactly the opacity being complained of.
    for token in str(name).replace("_", " ").split():
        if token in DECLARED:
            return DECLARED[token]
    if name in DECLARED:
        return DECLARED[name]
    if name.startswith("ARERA"):
        #Lorenzo Giannuzzo: the ARERA key is built in mapping.py as class plus
        #residency, and both halves are already in the name, so the meaning is
        #recovered from it rather than tabulated a second time and left to drift.
        parts = name.split()
        band = parts[1] if len(parts) > 1 else "?"
        res = " ".join(parts[2:]).lower() if len(parts) > 2 else ""
        return f"domestic, {band} kW, {_english_residency(res)}"
    return ""


def _national_frame() -> pd.DataFrame:
    """One row per point: its published profile, its activity class, its group."""
    from common import assignment

    users = pd.read_parquet(CACHE / "users.parquet")
    groups = pd.read_parquet(CACHE / "groups.parquet")

    g = assignment.gse_profile(users)[["pod", "gse_column"]]
    g = g.rename(columns={"gse_column": "national"})
    a = assignment.arera_key(users)
    a = a[a["arera_applicable"]].copy()
    a["national"] = ("ARERA " + a["arera_class"].astype(str) + " "
                     + a["arera_residency"].astype(str))
    both = pd.concat([g[["pod", "national"]], a[["pod", "national"]]], ignore_index=True)

    act = users[["pod", "ateco_l1"]].rename(columns={"ateco_l1": "activity"})
    m = groups.merge(both, on="pod", how="inner").merge(act, on="pod", how="left")
    return m[m["activity"].notna()]


#Lorenzo Giannuzzo: a palette of its own, long enough that the named classes never
#wrap onto a colour already in use. With the six of UNDER the seventh class came back
#as the first blue and sat next to it in the same bar.
CLASSES = ["#328cc1", "#e2a33c", "#c1440e", "#7d5ba6", "#4c9a6a", "#a8577e",
           "#0b3c5d", "#d98c3f", "#5f7d95", "#8c6c3f"]


def fig_declared_against_actual(min_points: int = 50, top_classes: int = 9) -> None:
    """The declared meaning of each published profile, against who it is applied to."""
    m = _national_frame()
    keep = m["national"].value_counts()
    m = m[m["national"].isin(keep[keep >= min_points].index)]
    if m.empty:
        print("  ! no published profile above the point threshold, figure skipped")
        return

    share = (pd.crosstab(m["national"], m["activity"])
             .pipe(lambda t: t.div(t.sum(axis=1), axis=0)))
    #Lorenzo Giannuzzo: the classes are ranked once over the whole table and the tail
    #is pooled, so that a colour means the same class in every bar. Ranking within each
    #bar would put a different class under the same colour on two neighbouring rows.
    ranked = share.sum(axis=0).sort_values(ascending=False)
    head = list(ranked.index[:top_classes])
    tail = [c for c in share.columns if c not in head]
    if tail:
        share[POOLED] = share[tail].sum(axis=1)
        share = share.drop(columns=tail)
    cols = head + ([POOLED] if tail else [])

    effective = share[cols].apply(
        lambda r: float(np.exp(-((p := r[r > 0]) * np.log(p)).sum())), axis=1)
    order = effective.sort_values().index
    share, effective = share.loc[order], effective.loc[order]
    n_pod = m["national"].value_counts().reindex(order)

    palette = {c: CLASSES[i % len(CLASSES)] for i, c in enumerate(head)}
    palette[POOLED] = NEUTRAL

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10.4, 0.62 * len(order) + 2.3))
    left = np.zeros(len(order))
    for c in cols:
        v = share[c].to_numpy() * 100
        ax.barh(y, v, left=left, height=0.62, color=palette[c],
                edgecolor="white", lw=0.6, label=activity_label(c))
        for i, (val, l0) in enumerate(zip(v, left)):
            if val >= 7:
                ax.text(l0 + val / 2, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if c != POOLED else INK)
        left += v

    for i, name in enumerate(order):
        ax.annotate(f"{effective[name]:.0f}", xy=(101.5, i), fontsize=9,
                    fontweight="bold", color=INK, va="center", annotation_clip=False)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{_english_residency(n)}\ndeclared: "
                        f"{_declared_meaning(n)}{SEP}{int(n_pod[n])} PODs"
                        for n in order], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Share of the points the profile is applied to [%]")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.annotate("Effective\nclasses", xy=(101.5, len(order) - 0.35), fontsize=8,
                fontweight="bold", color=INK, va="bottom", annotation_clip=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16 / (0.32 * len(order))),
              ncol=min(len(cols), 3), fontsize=8, frameon=True, framealpha=1.0,
              edgecolor="black", title="Activity class", title_fontsize=8)
    #Lorenzo Giannuzzo: matplotlib has no justified text, so the two lines are set
    #centred over the axes and padded well clear of the top bar. Left alignment made
    #the block hang off the long y labels instead of sitting over the plot.
    ax.set_title("What each published profile declares, against who it is applied to\n"
                 "a profile whose label described its users would be a single bar",
                 loc="center", fontsize=10, pad=22, linespacing=1.5)
    save(fig, "fig14_declared_against_actual", "coverage")


# ===========================================================================
#  How far a published profile reaches beyond the category it is applied to.
#
#  A national profile is applied to a declared category, and within that
#  category it is too coarse, which is what the two figures above establish. The
#  opposite failure is not visible there. The points a profile is applied to sit
#  in behaviours, and those behaviours hold other users the profile is never
#  applied to. If the behaviour is the same, the published curve describes those
#  users too, without being allowed to.
#
#  The reach is read in two steps: the share of a profile's points in each
#  behaviour, then the composition of each behaviour taken over all its members
#  rather than over the profile's own. Composing the two gives the population the
#  curve actually describes, and the part of it that falls outside the declared
#  category is what the assignment rule leaves on the table.
# ===========================================================================
DOMESTIC = ("DO",)


def _declared_classes(name: str, present: list) -> set:
    """The activity classes a published profile is meant for."""
    if name.startswith(("PAU",)):
        #Lorenzo Giannuzzo: the other-uses profiles are defined by exclusion, so their
        #declared scope is everything the domestic ones do not take, rather than a list
        #that would have to be maintained against the activity nomenclature.
        return {c for c in present if not str(c).startswith(DOMESTIC)}
    return {c for c in present if str(c).startswith(DOMESTIC)}


def fig_reach_beyond_declared(min_points: int = 50, top_classes: int = 9) -> None:
    """Who else a published profile would describe, through the behaviours it uses."""
    from common import assignment  # noqa: F401  (kept for the import check)

    m = _national_frame()
    users = pd.read_parquet(CACHE / "users.parquet")
    groups = pd.read_parquet(CACHE / "groups.parquet")
    full = groups.merge(users[["pod", "ateco_l1"]].rename(columns={"ateco_l1": "activity"}),
                        on="pod", how="left")
    full = full[full["activity"].notna()]
    if "below_n_min" in full:
        full = full[~full["below_n_min"]]

    #Lorenzo Giannuzzo: composition of every behaviour, over all its members
    comp = pd.crosstab(full["group"], full["activity"])
    comp = comp.div(comp.sum(axis=1), axis=0)

    keep = m["national"].value_counts()
    m = m[m["national"].isin(keep[keep >= min_points].index)]
    w = pd.crosstab(m["national"], m["group"])
    w = w.div(w.sum(axis=1), axis=0)

    shared = [g for g in w.columns if g in comp.index]
    reach = pd.DataFrame(w[shared].to_numpy() @ comp.loc[shared].to_numpy(),
                         index=w.index, columns=comp.columns)
    if reach.empty:
        print("  ! no overlap between the profiles and the behaviours, figure skipped")
        return

    outside = pd.Series(
        {n: 1.0 - reach.loc[n, list(_declared_classes(n, list(reach.columns)))].sum()
         for n in reach.index})
    order = outside.sort_values().index
    reach, outside = reach.loc[order], outside.loc[order]

    ranked = reach.sum(axis=0).sort_values(ascending=False)
    head = list(ranked.index[:top_classes])
    tail = [c for c in reach.columns if c not in head]
    if tail:
        reach[POOLED] = reach[tail].sum(axis=1)
        reach = reach.drop(columns=tail)
    cols = head + ([POOLED] if tail else [])
    palette = {c: CLASSES[i % len(CLASSES)] for i, c in enumerate(head)}
    palette[POOLED] = NEUTRAL

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10.6, 0.62 * len(order) + 2.6))
    left = np.zeros(len(order))
    for c in cols:
        v = reach[c].to_numpy() * 100
        ax.barh(y, v, left=left, height=0.62, color=palette[c],
                edgecolor="white", lw=0.6, label=activity_label(c))
        for i, (val, l0) in enumerate(zip(v, left)):
            if val >= 6:
                ax.text(l0 + val / 2, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if c != POOLED else INK)
        left += v

    for i, n in enumerate(order):
        ax.annotate(f"{outside[n]*100:.0f}%", xy=(101.5, i), fontsize=9.5,
                    fontweight="bold", color=WARM, va="center", annotation_clip=False)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\napplied to: {_declared_meaning(n)}" for n in order],
                       fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Share of the users whose behaviour the published curve describes [%]")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.annotate("Outside the\ncategory", xy=(101.5, len(order) - 0.32), fontsize=8,
                fontweight="bold", color=WARM, va="bottom", annotation_clip=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20 / (0.30 * len(order))),
              ncol=min(len(cols), 3), fontsize=8, frameon=True, framealpha=1.0,
              edgecolor="black", title="Activity class", title_fontsize=8)
    ax.set_title("Who each published profile would describe, "
                 "through the behaviours it is built on\n"
                 "the figure on the right is the share that falls outside "
                 "the category the profile is applied to",
                 loc="center", fontsize=10, pad=22, linespacing=1.5)
    save(fig, "fig15_reach_beyond_declared", "coverage")


# ===========================================================================
#  The two metrics with the curves beside them.
#
#  M1 and M2 are read from dot plots and a contingency table, and a reader who
#  has followed them still has not seen the behaviour any of the numbers refer
#  to. These two figures put the curve next to the count: what a profile looks
#  like beside the classes it gathers, and what an activity class fragments into
#  beside the profiles it fragments across.
#
#  One cell carries the curves, the winter working day, which holds the largest
#  share of the annual energy in every profile of the case study. Three seasons
#  would triple the height for a point about composition rather than seasonality.
# ===========================================================================
#Lorenzo Giannuzzo: the working day of each season. The weekend is left out because the
#figure is about which users share a behaviour, and the three working days already carry
#most of the annual energy; showing six cells would double the width for a contrast the
#day-type figures of Section 2.5 make better.
CURVE_SEASONS = ["winter", "mid", "summer"]
DAYTYPES = ["weekday", "saturday", "sunday"]
CURVE_CELLS = [(s, "weekday") for s in CURVE_SEASONS]


def cells_for(daytype: str) -> list:
    """The three seasonal cells of one day type.

    Kept as a function rather than a second constant because the figure is drawn once per
    day type. Nine cells in one figure would be four times as wide as the panels are tall
    and unreadable at page size, and the comparison the reader makes is within a day type,
    not across the diagonal of a nine-panel grid.
    """
    return [(s, daytype) for s in CURVE_SEASONS]
YLABEL = "Power normalized to an annual consumption of 1,000 kWh [kW]"


def _effective(counts: np.ndarray) -> float:
    #Lorenzo Giannuzzo: the same exponential of the Shannon entropy the mapping stage
    #computes. It is recomputed here from the same table rather than read back from the
    #CSV so that the figure cannot show a number the panel beside it contradicts.
    c = np.asarray(counts, float)
    c = c[c > 0]
    if c.sum() <= 0:
        return float("nan")
    p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def _curves_by_profile(cells: list = None) -> list:
    """One dictionary of curves per cell, in the order the cells are given."""
    cells = cells or CURVE_CELLS
    curves = pd.read_csv(GEN / "profiles.csv")
    kw = [f"kW{i}" for i in range(1, 97)]
    out = []
    for season, daytype in cells:
        sub = curves[(curves["season"] == season) & (curves["daytype"] == daytype)]
        out.append({int(r.profile): sub.loc[r.Index, kw].to_numpy(float)
                    for r in sub.itertuples()})
    return out


def _membership() -> pd.DataFrame:
    """One row per point: its behaviour and its activity class."""
    users = pd.read_parquet(CACHE / "users.parquet")
    groups = pd.read_parquet(CACHE / "groups.parquet")
    full = groups.merge(users[["pod", "ateco_l1"]].rename(columns={"ateco_l1": "activity"}),
                        on="pod", how="left")
    full = full[full["activity"].notna()]
    if "below_n_min" in full:
        full = full[~full["below_n_min"]]
    return full


_CALENDAR = None


def _calendar() -> pd.DataFrame:
    """The season and day-type calendar of the metering year, built once.

    The year is taken from the metered days rather than from the configuration. The
    configuration key is optional and was empty here, which turned every call into
    int(None) and took the whole figure down with a TypeError that named neither the key
    nor the stage. Reading it from the data cannot be out of step with the days the curves
    were averaged over, which is the property that actually matters.
    """
    global _CALENDAR
    if _CALENDAR is None:
        days = pd.read_parquet(CACHE / "days.parquet")
        year = int(pd.to_datetime(days["date"]).dt.year.mode().iloc[0])
        _CALENDAR = C.build_calendar(year, C.season_map_from_days(days))
    return _CALENDAR


def _cell_months(season: str) -> list[int]:
    cal = _calendar()
    return sorted(cal.loc[cal["season"] == season, "date"].dt.month.unique())


def _national_names() -> list:
    """The published profiles the comparison stage considered, in its own spelling."""
    path = _CFG.results_dir("comparison") / "b1_ddslp_vs_national.csv"
    if not path.exists():
        return []
    return sorted(pd.read_csv(path)["national"].astype(str).unique())


def _total_variation(a: np.ndarray, b: np.ndarray) -> float:
    #Lorenzo Giannuzzo: half the sum of the absolute differences between the two daily
    #shapes once each is normalised to one. It is the share of the day's energy the one
    #would misallocate if used in place of the other, which is the quantity Section 2.5
    #reports, so a distance read here is on the same scale as a distance read there.
    p = np.asarray(a, float)
    q = np.asarray(b, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def _nearest_per_cell(per_cell: list, cells: list, max_distance: float) -> dict:
    """For each behaviour and each cell, the closest published profile in that cell.

    The pairing is recomputed per cell rather than taken once from the annual table. A
    published profile can sit close to a behaviour in summer and far from it in winter,
    and a single pairing carried across the three panels hides exactly that, which is one
    of the things the seasonal panels were added to show.
    """
    names = _national_names()
    if not names:
        print("  fig16: comparison output absent, national curves omitted")
        return {}
    curves = {}
    for j, cell in enumerate(cells):
        for name in names:
            nat = _national_curve(name, *cell)
            if nat is not None:
                curves[(j, name)] = np.repeat(nat, 4) / 4.0

    out, rejected, unresolved = {}, [], set(names) - {n for _, n in curves}
    for g in per_cell[0]:
        for j in range(len(cells)):
            best, best_d = None, np.inf
            for name in names:
                shape = curves.get((j, name))
                if shape is None:
                    continue
                d = _total_variation(per_cell[j][g], shape)
                if np.isfinite(d) and d < best_d:
                    best, best_d = name, d
            if best is None:
                continue
            if best_d <= max_distance:
                out.setdefault(g, {})[j] = (best, best_d, curves[(j, best)])
            else:
                rejected.append(f"DD-SLP {g} in {cells[j][0]}: {best} at {best_d:.3f}")
    if unresolved:
        print("  fig16: national curve not reconstructed for "
              + ", ".join(sorted(unresolved)))
    if rejected:
        #Lorenzo Giannuzzo: printed with the distances so the threshold can be judged
        #against the numbers rather than guessed at.
        print(f"  fig16: nothing within {max_distance:.2f} for " + "; ".join(rejected))
    return out


def _nearest_national(max_distance: float = 0.15) -> dict:
    """For each behaviour, the closest national profile, when there is a close one.

    The distance is the one the comparison stage already computes, so the pairing shown
    here is the pairing Section 2.5 reports and not a second opinion formed in the figure.
    A behaviour with nothing within `max_distance` is left without a national curve rather
    than paired with the least distant of a bad set: drawing the nearest whatever the
    distance would suggest a correspondence exists in every row, which is the claim the
    paper is testing.
    """
    path = _CFG.results_dir("comparison") / "b1_ddslp_vs_national.csv"
    if not path.exists():
        print("  fig16: comparison output absent, national curves omitted")
        return {}
    b1 = pd.read_csv(path)
    if "setting" in b1.columns and (b1["setting"] == "S1_monthly").any():
        b1 = b1[b1["setting"] == "S1_monthly"]
    out, rejected = {}, []
    for ddslp, sub in b1.groupby("ddslp"):
        best = sub.loc[sub["total_variation"].idxmin()]
        g = int(str(ddslp).split("_")[-1])
        if float(best["total_variation"]) <= max_distance:
            out[g] = (str(best["national"]), float(best["total_variation"]))
        else:
            rejected.append(f"DD-SLP {g}: {best['national']} at "
                            f"{float(best['total_variation']):.3f}")
    if rejected:
        #Lorenzo Giannuzzo: the rejections are printed with their distances so that the
        #threshold can be judged against the numbers instead of guessed at.
        print(f"  fig16: nothing within {max_distance:.2f} for " + "; ".join(rejected))
    return out


def _gse_curve(name: str, season: str, daytype: str) -> np.ndarray | None:
    path = CACHE / "gse.parquet"
    if not path.exists():
        return None
    gse = pd.read_parquet(path)
    #Lorenzo Giannuzzo: the published name may arrive carrying a prefix, so the column is
    #looked for among the tokens of the label rather than by matching the label whole.
    col = next((t for t in str(name).replace("_", " ").split() if t in gse.columns), None)
    if col is None:
        return None
    cal = _calendar()
    sel_days = cal[(cal["season"] == season) & (cal["daytype"] == daytype)]
    stamp = pd.to_datetime(dict(year=gse.year, month=gse.month, day=gse.day)).dt.date
    sel = gse[stamp.isin(sel_days["date"].dt.date)]
    if sel.empty:
        return None
    prof = sel.groupby("hour")[col].mean().reindex(range(24))
    if prof.isna().any() or prof.sum() <= 0:
        return None
    return (prof / prof.sum()).to_numpy()


def _arera_curve(name: str, season: str, daytype: str) -> np.ndarray | None:
    """The ARERA curve for the power class and residency named in the label.

    The workbooks are tabulated by month, so the cell is assembled by averaging the months
    the calendar assigns to the season, which is the same grid the generation stage uses.
    """
    cached = [p for p in sorted(CACHE.glob("arera_*.parquet"))
              if "provenance" not in p.name]
    if not cached:
        return None
    tab = pd.read_parquet(cached[0])
    label = str(name).replace("ARERA", "").strip()
    cls = next((c for c in sorted(tab["power_class"].unique(), key=len, reverse=True)
                if label.startswith(str(c))), None)
    if cls is None:
        return None
    residency = label[len(str(cls)):].strip()
    sel = tab[(tab["power_class"] == cls) & (tab["daytype"] == daytype)
              & (tab["month"].isin(_cell_months(season)))]
    if residency:
        match = sel[sel["residency"].str.strip().str.casefold() == residency.casefold()]
        if not match.empty:
            sel = match
    if sel.empty:
        return None
    prof = sel.groupby("hour")["kWh"].mean().reindex(range(24))
    if prof.isna().any() or prof.sum() <= 0:
        return None
    return (prof / prof.sum()).to_numpy()


def _national_curve(name: str, season: str, daytype: str) -> np.ndarray | None:
    """The published curve on the requested cell, as a share of the daily energy."""
    if "ARERA" in str(name).upper():
        return _arera_curve(name, season, daytype)
    return _gse_curve(name, season, daytype)


#Lorenzo Giannuzzo: the effective number is not drawn. It is the subject of the dot plots
#of figures 7 and 8 and repeating it in the margin here competed with the curves for the
#reader's attention without adding anything the tables do not already carry.
def _row_figure(rows: list, title: str, right_title: str,
                name: str, part: str, cells: list, ylabel: str) -> None:
    """One row per subject: the curves on the left, the composition on the right.

    `rows` carries, per subject: its label, one list of curves per cell, an optional
    overlay per cell, the bars, and the effective number printed at the edge. The three
    figures built on this differ in what a subject is and in what the bars count, and in
    nothing else.
    """
    n = len(rows)
    nc = len(cells)
    #Lorenzo Giannuzzo: a short figure needs proportionally more head room, otherwise the
    #title lands on the panel titles. Expressed in inches it would vanish at one row.
    head = 0.22 if n > 2 else 0.42
    fig_h = 1.55 * n + head + 0.55
    #Lorenzo Giannuzzo: the bar column is given both extra width and extra space to its
    #left, because its tick labels are class names and they grow leftwards into whatever
    #panel precedes them. Shrinking the font instead would have cost legibility on the
    #one column the figure is read for.
    #Lorenzo Giannuzzo: the panels take the width back. The bar labels needed room, but
    #buying it with white space between every column shrank the curves, which are what the
    #figure is for. The room comes from a wider figure and from the bar column alone,
    #whose left margin is the only one that has to hold a class name.
    fig, axes = plt.subplots(n, nc + 1, figsize=(3.35 * nc + 6.2, fig_h),
                             gridspec_kw={"width_ratios": [1.0] * nc + [1.55],
                                          "hspace": 0.26, "wspace": 0.10},
                             squeeze=False)
    #Lorenzo Giannuzzo: one vertical scale for every curve in the figure, not per panel.
    #Per-panel scales made a profile that is flat to within two per cent look as
    #structured as one that doubles over the day.
    for i in range(n):
        for j in range(nc):
            if (i, j) != (0, 0):
                axes[i][j].sharey(axes[0][0])
            if j:
                axes[i][j].tick_params(labelleft=False)

    x = np.arange(96) / 4.0
    for i, row in enumerate(rows):
        for j, (season, daytype) in enumerate(cells):
            ax = axes[i][j]
            for curve, _label, colour in row.get("overlay", {}).get(j, []):
                #Lorenzo Giannuzzo: dashed, so that it never reads as one more behaviour.
                #It is published, not measured, and the distinction is the whole point of
                #putting the two in the same panel. The colour is chosen by the caller,
                #which knows whether the dashed line is a catalogue curve and which
                #catalogue it comes from.
                ax.plot(x, curve, color=colour, lw=1.2, ls="--", alpha=0.95, zorder=3)
            for curve, weight, colour in row["curves"][j]:
                ax.plot(x, curve, color=colour, lw=0.6 + 1.0 * weight, alpha=0.95)
            #Lorenzo Giannuzzo: a legend inside the panel rather than a caption above it.
            #The entries are declared by the caller, which is the only place that knows
            #whether the solid line is a behaviour or a published profile, and the night
            #hours leave the upper left corner free in every row of these figures.
            entries = row.get("legend", {}).get(j, [])
            if entries:
                handles = [plt.Line2D([], [], color=c, lw=1.6, ls=st) for _l, c, st in entries]
                ax.legend(handles, [l for l, _c, _s in entries], loc="upper left",
                          fontsize=6.2, frameon=True, framealpha=0.9,
                          edgecolor=NEUTRAL, handlelength=1.7, borderpad=0.35,
                          labelspacing=0.25, borderaxespad=0.3)
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.grid(color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6.5)
            if j == 0:
                #Lorenzo Giannuzzo: the unit repeated on the row, because the label of the
                #whole figure sits far to the left and a reader looking at the fifth row
                #has no reason to travel back to it.
                ax.set_ylabel(row["label"] + "\nNormalized power [kW]",
                              fontsize=7.4, color=INK)
            if i == 0:
                #Lorenzo Giannuzzo: the padding is back to normal. It was opened up to
                #clear the national caption that used to sit above the panel, and that
                #caption is now a legend inside it.
                ax.set_title(f"{_pretty(season)}, {_pretty(daytype)}",
                             fontsize=8.3, color=INK, pad=6)
            if i == n - 1:
                ax.set_xlabel("Time of day [h]", fontsize=7.8)

        ax = axes[i][nc]
        bars = row["bars"]
        y = np.arange(len(bars))[::-1]
        vals = np.array([v for _l, v, _c in bars]) * 100
        ax.barh(y, vals, height=0.68,
                color=[c for _l, _v, c in bars], edgecolor="white", lw=0.4)
        for yy, v in zip(y, vals):
            ax.text(v + 1.6, yy, f"{v:.0f}", va="center", fontsize=6.4, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([l for l, _v, _c in bars], fontsize=6.6)
        ax.set_xlim(0, min(105, vals.max() * 1.22))
        ax.set_ylim(-0.7, max(len(bars) - 0.3, 0.5))
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.set_title(right_title, fontsize=8.5, loc="left", color=INK)
    #Lorenzo Giannuzzo: no figure-wide vertical label. It reserved a column of its own
    #down the whole left side for one line of text, and the row labels already carry the
    #unit. The normalisation it used to state now rides in the title, where it is read
    #once and costs no width.
    fig.tight_layout(rect=[0.004, 0, 0.99, 1.0 - head / fig_h])
    #Lorenzo Giannuzzo: the bar column is shifted right after the layout pass, so its tick
    #labels get their own margin instead of one applied to every gap in the figure. Done
    #before tight_layout it would simply be overwritten by it.
    shift = 0.062
    for i in range(n):
        box = axes[i][nc].get_position()
        axes[i][nc].set_position([box.x0 + shift, box.y0,
                                  max(box.width - shift, 0.05), box.height])
    #Lorenzo Giannuzzo: placed after the layout pass and measured from where the panels
    #actually ended up. tight_layout silently ignores its rect when a figure carries
    #annotations outside the axes, which this one does, so a title positioned beforehand
    #landed on the panel titles and no amount of head room moved it.
    #Lorenzo Giannuzzo: no heading inside the figure. The title and the normalisation it
    #stated belong to the caption, which is where a journal expects them and where they do
    #not compete with the panels for height. The arguments are kept in the signature so
    #that the three callers keep documenting what each figure shows.
    save(fig, name, part)


def fig_profiles_with_classes(top: int = 6, max_distance: float = 0.15,
                              daytype: str = "weekday") -> None:
    """M2 with the behaviour shown: what each profile looks like and who is in it."""
    cells = cells_for(daytype)
    per_cell = _curves_by_profile(cells)
    nearest = _nearest_per_cell(per_cell, cells, max_distance)
    m = _membership()
    comp = pd.crosstab(m["group"], m["activity"])
    comp = comp.div(comp.sum(axis=1), axis=0)
    sizes = m["group"].value_counts()

    ranked = comp.sum(axis=0).sort_values(ascending=False)
    palette = {c: CLASSES[i] for i, c in enumerate(ranked.index[:len(CLASSES)])}

    rows = []
    for g in sorted(c for c in comp.index if c in per_cell[0]):
        share = comp.loc[g]
        share = share[share >= 0.005].sort_values(ascending=False).head(top)

        overlay, legend = {}, {}
        for j in range(len(cells)):
            legend[j] = [(f"DD-SLP {g}", INK, "-")]
        for j, (nat_name, dist, shape) in nearest.get(g, {}).items():
            legend[j].append((_national_caption(nat_name),
                              _national_color(nat_name), "--"))
            #Lorenzo Giannuzzo: put on the same daily energy as the behaviour before being
            #drawn, so the panel compares the shape of the day and not the amplitude,
            #which the comparison stage answers in its own terms.
            daily = per_cell[j][g].sum()
            overlay[j] = [(shape / shape.sum() * daily, nat_name,
                           _national_color(nat_name))]

        rows.append({
            "label": f"DD-SLP {g}\n({int(sizes.get(g, 0))} PODs)",
            "curves": [[(cell[g], 1.0, INK)] for cell in per_cell],
            "overlay": overlay,
            "legend": legend,
            "bars": [(activity_label(c), float(v), palette.get(c, NEUTRAL))
                     for c, v in share.items()],
            "effective": _effective(comp.loc[g].to_numpy()),
        })
    _row_figure(rows,
                "What each behaviour looks like, and which activity classes it gathers",
                "Share of the points in the behaviour [%]",
                f"fig16_profiles_with_classes_{daytype}", "aggregation", cells, YLABEL)


def fig_classes_across_profiles(top_classes: int = 6, min_pods: int = 25) -> None:
    """M1 with the behaviours shown: what a single activity class fragments into."""
    per_cell = _curves_by_profile()
    curves = per_cell[0]
    m = _membership()
    comp = pd.crosstab(m["activity"], m["group"])
    comp = comp[[c for c in comp.columns if c in curves]]
    sizes = comp.sum(axis=1)

    #Lorenzo Giannuzzo: the classes shown are the most fragmented ones, which are the ones
    #the argument is about, but only among those large enough for the fragmentation to be
    #a property of the class rather than of a handful of points.
    eligible = comp.loc[sizes >= min_pods]
    eff = eligible.apply(lambda r: _effective(r.to_numpy()), axis=1)
    chosen = eff.sort_values(ascending=False).head(top_classes).index

    palette = {g: UNDER[i % len(UNDER)] for i, g in enumerate(sorted(curves))}
    rows = []
    for c in chosen:
        row = comp.loc[c]
        share = (row / row.sum()).sort_values(ascending=False)
        share = share[share > 0]
        rows.append({
            "label": f"{activity_label(c)}\n({int(sizes[c])} PODs)",
            "curves": [[(cell[g], float(share[g]), palette[g]) for g in share.index]
                       for cell in per_cell],
            "bars": [(f"DD-SLP {g}", float(v), palette[g]) for g, v in share.items()],
            "effective": float(eff[c]),
        })
    _row_figure(rows,
                "What a single activity class fragments into",
                "Share of the class [%]",
                "fig17_classes_across_profiles", "multiplicity", CURVE_CELLS, YLABEL)


# ===========================================================================
#  The published profiles that match nothing.
#
#  Figure 16 pairs a behaviour with the national profile closest to it and
#  leaves the row bare when nothing is close. That silence is itself a result,
#  and it is read the wrong way round there: a behaviour without a match is a
#  gap in the catalogue, but a *published profile* without a match is a curve
#  the regulator applies to real users while describing none of them.
#
#  Here the national profile is the subject. Each row shows its curve against
#  the nearest behaviour, which is the least distant one available and is drawn
#  however far it is, and beside it the behaviours its own points actually fall
#  into. A profile far from the behaviour it is closest to, whose points then
#  scatter over several, is being applied without describing anything.
# ===========================================================================
def fig_national_without_match(min_distance: float = 0.15, top: int = 6,
                               min_points: int = 25) -> None:
    """National profiles whose nearest behaviour is still far away."""
    path = _CFG.results_dir("comparison") / "b1_ddslp_vs_national.csv"
    if not path.exists():
        print("  fig18: comparison output absent, figure skipped")
        return
    b1 = pd.read_csv(path)
    if "setting" in b1.columns and (b1["setting"] == "S1_monthly").any():
        b1 = b1[b1["setting"] == "S1_monthly"]

    per_cell = _curves_by_profile()
    nat_frame = _national_frame()
    counts = nat_frame["national"].value_counts()

    far = []
    for national, sub in b1.groupby("national"):
        best = sub.loc[sub["total_variation"].idxmin()]
        dist = float(best["total_variation"])
        if dist < min_distance or counts.get(national, 0) < min_points:
            continue
        far.append((str(national), int(str(best["ddslp"]).split("_")[-1]), dist))
    if not far:
        print(f"  fig18: every published profile has a behaviour within "
              f"{min_distance:.2f}, nothing to draw")
        return
    far.sort(key=lambda r: -r[2])

    palette = {g: UNDER[i % len(UNDER)] for i, g in enumerate(sorted(per_cell[0]))}
    rows, unresolved = [], []
    for national, nearest_g, dist in far:
        curves_per_cell, overlay, legend = [], {}, {}
        for j, cell in enumerate(CURVE_CELLS):
            nat = _national_curve(national, *cell)
            if nat is None:
                if j == 0:
                    unresolved.append(national)
                curves_per_cell.append([])
                continue
            #Lorenzo Giannuzzo: the published curve is the subject here, so it is the one
            #drawn solid and the behaviour is the dashed reference. The roles are the
            #reverse of figure 16 and the styling follows them rather than the colours.
            daily = per_cell[j][nearest_g].sum()
            curves_per_cell.append([(np.repeat(nat, 4) / 4.0 * daily, 1.0,
                                     _national_color(national))])
            overlay[j] = [(per_cell[j][nearest_g], f"DD-SLP {nearest_g}", INK)]
            legend[j] = [(_national_caption(national),
                          _national_color(national), "-"),
                         (f"DD-SLP {nearest_g}, closest behaviour", INK, "--")]

        pts = nat_frame[nat_frame["national"] == national]
        spread = pts["group"].value_counts()
        spread = (spread / spread.sum()).sort_values(ascending=False).head(top)
        rows.append({
            "label": f"{_english_residency(national)}\n({int(counts[national])} PODs)",
            "curves": curves_per_cell,
            "overlay": overlay,
            "legend": legend,
            "bars": [(f"DD-SLP {g}", float(v), palette.get(g, NEUTRAL))
                     for g, v in spread.items()],
            "effective": _effective(pts["group"].value_counts().to_numpy()),
        })
    if unresolved:
        print("  fig18: national curve not reconstructed for "
              + ", ".join(sorted(set(unresolved))))
    _row_figure(rows,
                f"Published profiles with no behaviour within {min_distance:.2f},\n"
                f"and the behaviours their own points fall into",
                "Share of the profile's points [%]",
                "fig18_national_without_match", "coverage", CURVE_CELLS, YLABEL)