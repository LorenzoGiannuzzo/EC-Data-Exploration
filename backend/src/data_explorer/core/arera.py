"""ARERA profile comparison.

Same idea as the GSE comparison module, but ARERA profiles are stored in kWh
(not %) and indexed by an extra dimension (day type: Weekday/Saturday/Sunday).

Public surface:
    fetch_our_arera_profile(session, pods, day_type, month_idx)
    fetch_arera_reference(session, power_class, market, residenza, day_type,
                          month_idx, province)
    compare_to_arera(our, reference)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

# ── ISO day-of-week → ARERA day_type ─────────────────────────────────────────
# Postgres EXTRACT(ISODOW): 1=Mon … 7=Sun
ARERA_DAYTYPE_TO_ISODOW = {
    "Weekday":  "BETWEEN 1 AND 5",
    "Saturday": "= 6",
    "Sunday":   "= 7",
}


# ── Fetch our profile in SQL ─────────────────────────────────────────────────
def fetch_our_arera_profile(
    session:    Session,
    pods:       list[str] | set[str],
    day_type:   str,                          # Weekday / Saturday / Sunday
    month_idx:  int | None = None,            # 0/None → annual average
    tipologia:  str = "AP",
) -> pd.Series:
    """Average hourly kWh profile of a POD-set for the given day type and month.

    The four quarter-hour columns of each hour are summed (Wh → kWh by /1000),
    then averaged across all PODs and matching days.

    Returns a Series indexed by hour_idx (0..23), values in kWh.
    """
    if not pods or day_type not in ARERA_DAYTYPE_TO_ISODOW:
        return pd.Series(dtype=float)

    hour_sums = ", ".join(
        f"(COALESCE(q{4*h+1},0)+COALESCE(q{4*h+2},0)"
        f"+COALESCE(q{4*h+3},0)+COALESCE(q{4*h+4},0))/1000.0 AS h{h}"
        for h in range(24)
    )
    dow_filter = (
        f"AND EXTRACT(ISODOW FROM data_misura) {ARERA_DAYTYPE_TO_ISODOW[day_type]}"
    )
    month_filter = (
        "AND EXTRACT(MONTH FROM data_misura) = :month"
        if month_idx and month_idx > 0 else ""
    )

    selects = ", ".join(f"AVG(h{h}) AS h{h}" for h in range(24))
    sql = text(
        f"""
        WITH hourly AS (
            SELECT pod, data_misura, {hour_sums}
            FROM measurements
            WHERE tipologia = :tipologia
              AND pod = ANY(:pods)
              {dow_filter}
              {month_filter}
        )
        SELECT {selects} FROM hourly
        """
    )
    params: dict = {"tipologia": tipologia, "pods": list(pods)}
    if month_idx and month_idx > 0:
        params["month"] = int(month_idx)

    row = session.execute(sql, params).first()
    if row is None or all(v is None for v in row):
        return pd.Series(dtype=float)
    s = pd.Series([float(v) if v is not None else 0.0 for v in row],
                  index=range(24), name="kwh")
    s.index.name = "hour_idx"
    # Smooth zero-valued holes via linear interpolation between neighbours.
    zeros = s == 0
    if zeros.any() and not zeros.all():
        s2 = s.where(~zeros, np.nan)
        s = s2.interpolate(method="linear", limit_direction="both").fillna(0.0)
    return s


# ── Fetch ARERA reference ────────────────────────────────────────────────────
def fetch_arera_reference(
    session:     Session,
    power_class: str,
    market:      str,
    residenza:   str,
    day_type:    str,
    month_idx:   int = 0,
    province:    str = "Trento",
) -> pd.Series:
    """ARERA hourly reference profile (kWh) for a fully-qualified key."""
    rows = session.execute(
        text(
            "SELECT hour_idx, value FROM reference_profiles_arera "
            "WHERE power_class = :pc AND market = :mk AND residenza = :res "
            "  AND province = :prov AND day_type = :dt AND month_idx = :m "
            "ORDER BY hour_idx"
        ),
        {
            "pc": power_class, "mk": market, "res": residenza,
            "prov": province, "dt": day_type, "m": month_idx,
        },
    ).all()
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(
        {int(r[0]): float(r[1]) for r in rows}, name="kwh"
    ).reindex(range(24)).fillna(0.0)
    s.index.name = "hour_idx"
    return s


# ── Comparison metrics ───────────────────────────────────────────────────────
def compare_to_arera(our: pd.Series, reference: pd.Series) -> dict:
    """Single-day-type comparison: absolute (kWh) + relative (% of reference mean)."""
    if our.empty or reference.empty:
        return {"rmse": None, "mae": None, "max_abs_err": None, "bias": None,
                "cv_rmse_pct": None, "nmae_pct": None, "rel_bias_pct": None}
    a, b = our.reindex(range(24)).fillna(0), reference.reindex(range(24)).fillna(0)
    err = a - b
    ref_mean = float(b.mean()) if float(b.mean()) > 0 else None
    out = {
        "rmse":         float(np.sqrt(np.mean(err ** 2))),
        "mae":          float(np.mean(np.abs(err))),
        "max_abs_err":  float(np.max(np.abs(err))),
        "bias":         float(np.mean(err)),
    }
    if ref_mean and ref_mean > 0:
        out["cv_rmse_pct"]  = out["rmse"] / ref_mean * 100
        out["nmae_pct"]     = out["mae"]  / ref_mean * 100
        out["rel_bias_pct"] = out["bias"] / ref_mean * 100
    else:
        out["cv_rmse_pct"] = out["nmae_pct"] = out["rel_bias_pct"] = None
    return out
