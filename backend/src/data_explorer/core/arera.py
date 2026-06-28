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

    # Each q_i is sanitised via COALESCE(NULLIF(q_i::text, 'NaN')::float8, 0)
    # so both NULL *and* the IEEE-754 NaN sentinel collapse to 0. PostgreSQL
    # treats NaN as a regular value (not NULL), so plain COALESCE() lets it
    # pass through and a single NaN row poisons the whole AVG() to NaN.
    hour_sums = ", ".join(
        f"NULLIF(("
        f"COALESCE(NULLIF(q{4*h+1}::text,'NaN')::float8,0)"
        f"+COALESCE(NULLIF(q{4*h+2}::text,'NaN')::float8,0)"
        f"+COALESCE(NULLIF(q{4*h+3}::text,'NaN')::float8,0)"
        f"+COALESCE(NULLIF(q{4*h+4}::text,'NaN')::float8,0)"
        f")/1000.0, 0) AS h{h}"
        for h in range(24)
    )
    dow_filter = (
        f"AND EXTRACT(ISODOW FROM data_misura) {ARERA_DAYTYPE_TO_ISODOW[day_type]}"
    )

    # ── 3-level nested mean (legacy parity) ─────────────────────────────────
    # The original dashboard's `compute_our_hourly_kwh_by_daytype` builds the
    # profile as:
    #     1. mean over days within each (POD, month) — one row per POD-month
    #     2. mean over PODs within each month         — one row per month
    #     3. annual = mean of the 12 monthly profiles — equal weight per month
    # A single flat AVG over all rows (what we used to do) instead weights
    # months and PODs proportionally to row counts, which on Lorenzo's
    # dataset systematically pulled the aggregate ~40 % below the ARERA
    # reference. Reproducing the nested mean closes that gap.
    per_pod_month_aggs = ", ".join(f"AVG(h{h}) AS h{h}" for h in range(24))
    per_month_aggs     = ", ".join(f"AVG(h{h}) AS h{h}" for h in range(24))
    annual_aggs        = ", ".join(f"AVG(h{h}) AS h{h}" for h in range(24))
    h_cols             = ", ".join(f"h{h}" for h in range(24))
    final_select = (
        f"SELECT {h_cols} FROM per_month WHERE month_idx = :month"
        if month_idx and month_idx > 0
        else f"SELECT {annual_aggs} FROM per_month"
    )
    sql = text(
        f"""
        WITH hourly AS (
            SELECT pod,
                   EXTRACT(MONTH FROM data_misura)::int AS month_idx,
                   {hour_sums}
            FROM measurements
            WHERE tipologia = :tipologia
              AND pod = ANY(:pods)
              {dow_filter}
        ),
        per_pod_month AS (
            SELECT pod, month_idx, {per_pod_month_aggs}
            FROM hourly
            GROUP BY pod, month_idx
        ),
        per_month AS (
            SELECT month_idx, {per_month_aggs}
            FROM per_pod_month
            GROUP BY month_idx
        )
        {final_select}
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

    # ── DST-style anomaly smoothing ─────────────────────────────────────────
    # In Italian DSO data the daily profile has 96 fixed quarter-hour slots,
    # which means the 02:00 hour disappears on the last Sunday of March and
    # repeats on the last Sunday of October. After averaging across many days
    # this leaves a characteristic notch (Sunday h=2 collapses far below its
    # 01:00 and 03:00 neighbours). Detect ANY hour whose value is less than
    # 30 % of the mean of the 4 hours around it and replace with the local
    # interpolation. The threshold is conservative — a genuine quiet hour
    # in a residential profile is rarely 4× lower than the surrounding two.
    arr = s.to_numpy(dtype=float, copy=True)
    for h in range(24):
        lo, hi = max(0, h - 2), min(24, h + 3)
        neighbours = np.concatenate((arr[lo:h], arr[h + 1:hi]))
        if neighbours.size == 0:
            continue
        local = float(neighbours.mean())
        if local <= 0:
            continue
        if arr[h] < 0.30 * local:
            arr[h] = local
    return pd.Series(arr, index=range(24), name="kwh").rename_axis("hour_idx")


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


# ── Bulk monthly fetchers (used by the "Full Monthly Overview") ─────────────
def fetch_our_arera_profile_monthly_all(
    session:    Session,
    pods:       list[str] | set[str],
    day_type:   str,
    tipologia:  str = "AP",
) -> dict[int, pd.Series]:
    """Return {1..12: Series[24]} of PoliTo hourly kWh profiles, one per month.

    Same NaN-safe sanitisation and per-(POD,month) → per-month nested mean as
    ``fetch_our_arera_profile``, but emitted in a single round-trip so the
    monthly grid renders without firing 12 separate queries.
    """
    if not pods or day_type not in ARERA_DAYTYPE_TO_ISODOW:
        return {}
    hour_sums = ", ".join(
        f"NULLIF(("
        f"COALESCE(NULLIF(q{4*h+1}::text,'NaN')::float8,0)"
        f"+COALESCE(NULLIF(q{4*h+2}::text,'NaN')::float8,0)"
        f"+COALESCE(NULLIF(q{4*h+3}::text,'NaN')::float8,0)"
        f"+COALESCE(NULLIF(q{4*h+4}::text,'NaN')::float8,0)"
        f")/1000.0, 0) AS h{h}"
        for h in range(24)
    )
    per_pod_month_aggs = ", ".join(f"AVG(h{h}) AS h{h}" for h in range(24))
    per_month_aggs     = ", ".join(f"AVG(h{h}) AS h{h}" for h in range(24))
    h_cols             = ", ".join(f"h{h}" for h in range(24))
    sql = text(
        f"""
        WITH hourly AS (
            SELECT pod,
                   EXTRACT(MONTH FROM data_misura)::int AS month_idx,
                   {hour_sums}
            FROM measurements
            WHERE tipologia = :tipologia
              AND pod = ANY(:pods)
              AND EXTRACT(ISODOW FROM data_misura) {ARERA_DAYTYPE_TO_ISODOW[day_type]}
        ),
        per_pod_month AS (
            SELECT pod, month_idx, {per_pod_month_aggs}
            FROM hourly GROUP BY pod, month_idx
        ),
        per_month AS (
            SELECT month_idx, {per_month_aggs}
            FROM per_pod_month GROUP BY month_idx
        )
        SELECT month_idx, {h_cols} FROM per_month ORDER BY month_idx
        """
    )
    rows = session.execute(
        sql, {"tipologia": tipologia, "pods": list(pods)}
    ).all()
    out: dict[int, pd.Series] = {}
    for r in rows:
        m = int(r[0])
        if not 1 <= m <= 12:
            continue
        vals = [float(v) if v is not None else 0.0 for v in r[1:]]
        s = pd.Series(vals, index=range(24), name="kwh")
        s.index.name = "hour_idx"
        out[m] = s
    return out


def fetch_arera_reference_monthly_all(
    session:     Session,
    power_class: str,
    market:      str,
    residenza:   str,
    day_type:    str,
    province:    str = "Trento",
) -> dict[int, pd.Series]:
    """Return {1..12: Series[24]} of ARERA reference profiles per month."""
    rows = session.execute(
        text(
            "SELECT month_idx, hour_idx, value FROM reference_profiles_arera "
            "WHERE power_class = :pc AND market = :mk AND residenza = :res "
            "  AND province = :prov AND day_type = :dt "
            "  AND month_idx BETWEEN 1 AND 12 "
            "ORDER BY month_idx, hour_idx"
        ),
        {"pc": power_class, "mk": market, "res": residenza,
         "prov": province, "dt": day_type},
    ).all()
    out: dict[int, pd.Series] = {}
    if not rows:
        return out
    df = pd.DataFrame(rows, columns=["month_idx", "hour_idx", "value"])
    for m, grp in df.groupby("month_idx"):
        s = pd.Series(
            grp.set_index("hour_idx")["value"].to_dict(), name="kwh"
        ).reindex(range(24)).fillna(0.0).astype(float)
        s.index.name = "hour_idx"
        out[int(m)] = s
    return out


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
