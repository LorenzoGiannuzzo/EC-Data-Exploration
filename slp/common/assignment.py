"""Which national profile the regulation assigns to each point of delivery.

This is the rule the audit of Section 2.5 rests on: block B2 compares every user with the
profile it is *entitled to* under the tariff and category it actually holds, not with the
closest profile available. The same rule feeds the contingency table of Section 2.6, which
is why it lives in common/ rather than inside either stage.

Two elements of the rule are not fully determined by the published documentation and are
therefore explicit, flagged, and reported in the run summary rather than buried:

  * the GSE categories for public lighting and vehicle charging cannot be told apart from
    the four-letter codes alone. The points concerned are a small share of the sample and
    are marked `uncertain` so that they can be excluded or reported separately;
  * the prefix distinguishing the two families of GSE profiles was tested against the
    prosumer flag on this dataset and the midday signature did not separate the two
    populations, so the P family is assigned throughout and the M family is available
    only as a sensitivity through `use_m_family`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# GSE column = family + category + tariff treatment
GSE_CATEGORY = {"domestic": "DM", "other": "AU", "lighting": "IR", "charging": "AC"}
GSE_TREATMENT = {"monorario": "M", "fasce": "F"}

ARERA_CLASSES = ["0-1.5", "1.5-3", "3-4.5", "4.5-6", ">6"]
_ARERA_EDGES = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf]


def gse_category(users: pd.DataFrame) -> pd.Series:
    """Map the tariff type to one of the four GSE categories.

    D_TIPTA is the authoritative field: TD is domestic, BTIP is public lighting, BTVE is
    public vehicle charging, and the BTA family is non-domestic other uses.
    """
    t = users["D_TIPTA"].astype(str).str.upper().str.strip()
    cat = pd.Series("other", index=users.index, dtype=object)
    cat[t.str.startswith("TD")] = "domestic"
    cat[t == "BTIP"] = "lighting"
    cat[t == "BTVE"] = "charging"
    return cat


def gse_profile(users: pd.DataFrame,
                treatment: str = "monorario",
                use_m_family: bool = False) -> pd.DataFrame:
    """Return the GSE column name per point of delivery, plus an uncertainty flag."""
    if treatment not in GSE_TREATMENT:
        raise ValueError(f"treatment must be one of {list(GSE_TREATMENT)}")
    cat = gse_category(users)
    family = np.where(users["prosumer"].fillna(False) & use_m_family, "M", "P")
    col = pd.Series(
        [f"{fam}{GSE_CATEGORY[c]}{GSE_TREATMENT[treatment]}" for fam, c in zip(family, cat)],
        index=users.index, dtype=object,
    )
    # the M family exists only for the two large categories
    fallback = col.str.startswith("M") & ~col.str[1:3].isin(["DM", "AU"])
    col[fallback] = "P" + col[fallback].str[1:]
    return pd.DataFrame({
        "pod": users["pod"].values,
        "gse_category": cat.values,
        "gse_column": col.values,
        "gse_uncertain": cat.isin(["lighting", "charging"]).values,
    })


def arera_class(power_kw: pd.Series) -> pd.Series:
    """Contractual power to the ARERA power class. Values outside the grid become NaN."""
    return pd.cut(power_kw, bins=_ARERA_EDGES, labels=ARERA_CLASSES, right=True)


def arera_key(users: pd.DataFrame) -> pd.DataFrame:
    """Power class and residency status, the two axes of the ARERA tables.

    The tables cover domestic points only, so every other point is returned with a null
    key and is excluded from that branch of the comparison rather than forced onto it.
    """
    cat = gse_category(users)
    desc = users["FDESC"].astype(str).str.upper()
    residency = np.where(desc.str.contains("NON RESID"), "Non Residente",
                         np.where(desc.str.contains("RESIDENT"), "Residente", None))
    cls = arera_class(users["D_POTC"]).astype(object)
    domestic = (cat == "domestic").values
    return pd.DataFrame({
        "pod": users["pod"].values,
        "arera_class": np.where(domestic, cls, None),
        "arera_residency": np.where(domestic, residency, None),
        "arera_applicable": domestic & pd.notna(cls) & pd.notna(residency),
    })


def build(users: pd.DataFrame, treatment: str = "monorario",
          use_m_family: bool = False) -> pd.DataFrame:
    """Full assignment table, one row per point of delivery."""
    g = gse_profile(users, treatment=treatment, use_m_family=use_m_family)
    a = arera_key(users)
    return g.merge(a, on="pod", how="inner")
