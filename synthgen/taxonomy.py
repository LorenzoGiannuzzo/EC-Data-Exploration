"""User typology.

The activity code lives in the metadata column `CCATETE`, right padded, and its
depth is not uniform across the register: most points carry two blocks (56.2),
some carry three (25.11.00), and the domestic ones carry a placeholder, DO.01 for
non resident and DO.02 for resident.

Levels are named after what they select, not after the column names of the source
database, which do not line up with the NACE hierarchy:

    level 1   division, two digits          47
    level 2   group, four digits            47.71
    level 3   class, six digits             47.71.10

A code can be truncated but not extended, so a point registered at 56.2 answers
at level 1 and level 2 and is simply absent from level 3. That absence is the
reason `available_at` exists: a typology has to be counted on the points that can
actually answer at the requested depth, not on the whole register.

Two codes in the register are not ATECO at all. `IL` marks public lighting and
`DO` marks domestic use. Both are kept as typologies in their own right, since
they name a real and homogeneous population, but they never take part in the
numeric rollup.
"""
from __future__ import annotations

import pandas as pd

DOMESTIC = "domestic"
DOMESTIC_PREFIX = "DO"


def is_domestic(code: object) -> bool:
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return False
    return str(code).strip().upper().startswith(DOMESTIC_PREFIX)


def split_ateco(code: object) -> tuple[str | None, str | None, str | None]:
    """The three levels of one raw code, None where the code does not reach."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return (None, None, None)
    raw = str(code).strip()
    if not raw:
        return (None, None, None)
    if is_domestic(raw):
        return (DOMESTIC, DOMESTIC, DOMESTIC)
    parts = [p for p in raw.split(".") if p != ""]
    out: list[str | None] = []
    for depth in (1, 2, 3):
        out.append(".".join(parts[:depth]) if len(parts) >= depth else None)
    return (out[0], out[1], out[2])


def level_column(level: int) -> str:
    if level not in (1, 2, 3):
        raise ValueError(f"ateco_level must be 1, 2 or 3, got {level!r}")
    return f"ateco_l{level}"


def normalise_typology(typology: str, level: int) -> str:
    """The key a request resolves to. Domestic ignores the level."""
    t = str(typology).strip()
    if t.lower() in (DOMESTIC, "domestico", "domestici") or is_domestic(t):
        return DOMESTIC
    parts = [p for p in t.split(".") if p != ""]
    if len(parts) < level:
        raise ValueError(
            f"typology {t!r} has {len(parts)} block(s) and cannot answer at "
            f"level {level}; ask for level {len(parts)} or a deeper code")
    return ".".join(parts[:level])


def available_at(users: pd.DataFrame, level: int) -> pd.DataFrame:
    """The points whose code reaches the requested depth."""
    col = level_column(level)
    return users[users[col].notna()]


def census(users: pd.DataFrame, level: int,
           min_pods: int = 1) -> pd.DataFrame:
    """How many points, and how much energy, each typology holds at a level."""
    col = level_column(level)
    sub = available_at(users, level)
    agg = {"n_pods": ("pod", "size")}
    if "n_days_valid" in sub:
        agg["days_valid_median"] = ("n_days_valid", "median")
    if "power_kW" in sub:
        agg["power_kW_median"] = ("power_kW", "median")
    out = (sub.groupby(col, dropna=False).agg(**agg)
              .sort_values("n_pods", ascending=False))
    out.index.name = "typology"
    return out[out["n_pods"] >= min_pods]