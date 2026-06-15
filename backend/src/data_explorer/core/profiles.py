"""Profile operations — pure Python, no DB and no UI imports.

A "profile" is a DataFrame indexed by POD with 96 columns q1..q96 (one row
per POD, one column per quarter-hour). All functions here transform such
DataFrames; they don't fetch data or render anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Q_COLS = [f"q{i}" for i in range(1, 97)]


def filter_all_zero(profiles: pd.DataFrame) -> pd.DataFrame:
    """Drop PODs whose entire profile is zero (no usable load).

    Mirrors the "26 PODs excluded from clustering: all-zero measurements"
    behaviour of the original dashboard.
    """
    if profiles.empty:
        return profiles
    nonzero = profiles[Q_COLS].fillna(0).abs().sum(axis=1) > 0
    return profiles[nonzero]


def normalise_profiles(
    profiles: pd.DataFrame,
    method: str = "max",
) -> pd.DataFrame:
    """Row-normalise each profile so that comparison between PODs is on shape,
    not magnitude.

    Methods:
        ``max``    — divide by the row's max ⇒ values in [0, 1]
        ``minmax`` — (x − min) / (max − min) per row ⇒ values in [0, 1]
        ``sum``    — divide by the row's sum ⇒ profile sums to 1

    Returns a DataFrame with the same index and the q1..q96 columns only.
    """
    if profiles.empty:
        return profiles[Q_COLS] if set(Q_COLS).issubset(profiles.columns) else profiles
    X = profiles[Q_COLS].astype(float).fillna(0)
    if method == "max":
        denom = X.max(axis=1).replace(0, np.nan)
        return X.div(denom, axis=0).fillna(0)
    if method == "sum":
        denom = X.sum(axis=1).replace(0, np.nan)
        return X.div(denom, axis=0).fillna(0)
    if method == "minmax":
        row_min   = X.min(axis=1)
        row_max   = X.max(axis=1)
        row_range = (row_max - row_min).replace(0, np.nan)
        return X.sub(row_min, axis=0).div(row_range, axis=0).fillna(0).clip(0.0, 1.0)
    raise ValueError(f"Unknown normalisation method: {method!r} "
                     f"(use 'max', 'minmax' or 'sum')")


def profile_summary(profiles: pd.DataFrame) -> dict:
    """Quick numeric overview of a profile DataFrame — handy for debug/CLI."""
    if profiles.empty:
        return {"n_pods": 0}
    X = profiles[Q_COLS]
    return {
        "n_pods":    int(len(X)),
        "min":       float(X.min().min()),
        "max":       float(X.max().max()),
        "mean":      float(X.mean().mean()),
        "n_allzero": int((X.abs().sum(axis=1) == 0).sum()),
    }
