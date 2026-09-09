"""Figures for the mapping stage (Section 2.6)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import INK, GRID, ACCENT, WARM, NEUTRAL, save as _save  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "paper_results" / "mapping_results"
FIG = RES / "figures"


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
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
    d = pd.read_csv(RES / "m1_multiplicity_pod.csv").sort_values("M1_effective")
    e = pd.read_csv(RES / "m1_multiplicity_energy.csv").set_index("activity")
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
    save(fig, "fig7_m1_multiplicity")


def fig_aggregation() -> None:
    """M2: how many activity classes each profile subsumes."""
    d = pd.read_csv(RES / "m2_aggregation_pod.csv").sort_values("M2_effective")
    e = pd.read_csv(RES / "m2_aggregation_energy.csv").set_index("profile")
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
    save(fig, "fig8_m2_aggregation")


def fig_coverage() -> None:
    """M3: what a published profile declares to represent, against what it does.

    The regulation assigns one national profile per tariff category, which is a claim that
    the category is homogeneous enough to be described by one curve. The figure puts that
    claim, the dashed line at one, next to two measurements: how many activity classes the
    profile is actually applied to, and how many distinct consumption behaviours are
    actually hiding under it. The second is the one that matters, because a profile may
    legitimately span many activity classes if they all consume alike.
    """
    d = pd.read_csv(RES / "m3_coverage_pod.csv").sort_values("M3_ddslp_effective")
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
    save(fig, "fig9_m3_coverage")


REQUIRED = ("contingency_pod.csv", "m1_multiplicity_pod.csv",
            "m2_aggregation_pod.csv", "m3_coverage_pod.csv")


def main() -> None:
    print(f"\n{'='*78}\n  FIGURES, Section 2.6\n{'='*78}")
    missing = [f for f in REQUIRED if not (RES / f).exists()]
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
    print(f"\n  figures in {FIG}\n")


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
    m1 = pd.read_csv(RES / "m1_multiplicity_pod.csv").set_index("activity")
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
    save(fig, "fig10_m1_composition")


def fig_aggregation_composition(top: int = 6) -> None:
    """M2 seen as composition: what each profile is actually made of, by energy."""
    ct = pd.read_csv(RES / "contingency_energy.csv", index_col=0).fillna(0.0)
    m2 = pd.read_csv(RES / "m2_aggregation_energy.csv").set_index("profile")
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
    save(fig, "fig11_m2_composition")


def fig_coverage_gap() -> None:
    """M3 as the gap it is: one profile declared, several behaviours delivered."""
    d = pd.read_csv(RES / "m3_coverage_pod.csv").sort_values("M3_ddslp_effective")
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
    save(fig, "fig12_m3_gap")


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
GEN = ROOT / "paper_results" / "generation_results"
CACHE = ROOT / "cache"
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
            # the gap is a share of the axis, not of the ends, so it is the
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
    save(fig, "fig13_behaviours_under_national")


def _describe_behaviours(curves: pd.DataFrame, share: pd.Series) -> pd.DataFrame:
    """One row per behaviour, with the quantities the table below the panels shows.

    Everything here is measured on the curves that are drawn above, except the annual
    consumption, which the profiles cannot carry because they are normalised. Naming a
    behaviour would be an interpretation and is left to the text; what the figure owes
    the reader is the numbers the interpretation would rest on.
    """
    kw = [f"kW{i}" for i in range(1, 97)]
    real = None
    for folder in ("clustering_results", "clustering"):
        path = ROOT / "paper_results" / folder / "groups.csv"
        if path.exists():
            real = pd.read_csv(path).set_index("group")
            break

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

    # centred on the figure, and no wider than the panels above it
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


def _declared_meaning(name: str) -> str:
    if name in DECLARED:
        return DECLARED[name]
    if name.startswith("ARERA"):
        #Lorenzo Giannuzzo: the ARERA key is built in mapping.py as class plus
        #residency, and both halves are already in the name, so the meaning is
        #recovered from it rather than tabulated a second time and left to drift.
        parts = name.split()
        band = parts[1] if len(parts) > 1 else "?"
        res = " ".join(parts[2:]).lower() if len(parts) > 2 else ""
        return f"domestic, {band} kW, {res}"
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
    ax.set_yticklabels([f"{n}\ndeclared: {_declared_meaning(n)}  ·  {int(n_pod[n])} PODs"
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
    save(fig, "fig14_declared_against_actual")


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

    # composition of every behaviour, over all its members
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
    save(fig, "fig15_reach_beyond_declared")