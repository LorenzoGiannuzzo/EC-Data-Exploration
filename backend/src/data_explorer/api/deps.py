"""Common helpers for the API routers.

`resolve_pod_set` is the single place where a ``PodFilter`` (ATECO + coverage)
becomes a concrete set of POD identifiers — used by clustering, GSE and ARERA
endpoints alike. Keeping it here avoids three subtly different implementations.

`monthly_profile_to_dict` turns a long-form DataFrame indexed by
(month_idx, hour_idx) into the ``{month: [24 floats]}`` JSON shape used in
responses.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from data_explorer.api.schemas import PodFilter
from data_explorer.db.queries import (
    fetch_pods_by_ateco, fetch_pods_by_power, fetch_pods_with_data_coverage,
)


def resolve_pod_set(session: Session, f: PodFilter) -> tuple[set[str], dict]:
    """Apply a PodFilter and return ``(pod_set, summary_dict)``.

    The summary dict is suitable for the ``PodSetSummary`` schema and shows
    how many PODs each filter step removed — useful for diagnostics.
    """
    coverage = fetch_pods_with_data_coverage(
        session, min_months=f.min_months, tipologia=f.tipologia
    )

    ateco_filter_applied = any([f.ateco_l1, f.ateco_l2, f.ateco_l3])
    if ateco_filter_applied:
        ateco_pods: set[str] = set()
        if f.ateco_l1:
            ateco_pods |= fetch_pods_by_ateco(session, f.ateco_l1, level=1)
        if f.ateco_l2:
            ateco_pods |= fetch_pods_by_ateco(session, f.ateco_l2, level=2)
        if f.ateco_l3:
            ateco_pods |= fetch_pods_by_ateco(session, f.ateco_l3, level=3)
        pods = coverage & ateco_pods
        after_ateco = len(ateco_pods)
    else:
        pods = coverage
        after_ateco = None

    # ── Contractual-power filter (legacy sidebar "Contractual Power") ────────
    after_power: int | None = None
    if f.power_ranges:
        ranges = [
            (float(r[0]), (None if len(r) < 2 or r[1] is None else float(r[1])))
            for r in f.power_ranges
        ]
        power_pods = fetch_pods_by_power(
            session, ranges, include_missing=f.include_missing_power
        )
        pods = pods & power_pods
        after_power = len(power_pods)

    sample = sorted(pods)[:10]
    summary = {
        "n_pods":         len(pods),
        "after_ateco":    after_ateco,
        "after_coverage": len(coverage),
        "after_power":    after_power,
        "sample_pods":    sample,
    }
    return pods, summary


def monthly_profile_to_dict(df: pd.DataFrame) -> dict[int, list[float]]:
    """Turn a (month_idx, hour_idx)-indexed DataFrame with a ``value`` column
    into ``{month: [24 floats]}``. Missing hours default to 0.0."""
    out: dict[int, list[float]] = {}
    if df.empty:
        return out
    long = df.reset_index()
    for month, grp in long.groupby("month_idx", sort=True):
        values = [0.0] * 24
        for _, row in grp.iterrows():
            h = int(row["hour_idx"])
            if 0 <= h < 24:
                values[h] = float(row["value"])
        out[int(month)] = values
    return out
