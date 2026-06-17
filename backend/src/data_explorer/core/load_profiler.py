"""Load Profiler — bucket-based hourly profile aggregation.

For a POD set, computes the average daily profile broken down by
(month, day_type, hour) where day_type ∈ {weekday, saturday, sunday}, and
exposes three output granularities:

* **annual**  — 3 day-type profiles × 24 hours (collapsed across months)
* **monthly** — 12 × 3 day-type profiles × 24 hours (legacy "monthly hourly")
* **daily**   — full 8760-hour year (reference 2025, non-leap) where every
  calendar hour inherits the mean + percentiles of its (month, day_type, hour)
  bucket. Output rows therefore expose the empirical distribution of
  consumption for that bucket: mean, std, P5, P25, P50, P75, P95, n_samples.

The heavy lifting is done in PostgreSQL (UNPIVOT via UNION ALL, then a
single `GROUP BY` with `PERCENTILE_CONT`), which is the only way to keep
runtime reasonable on multi-million-row `measurements` tables.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

DAY_TYPES: list[str] = ["weekday", "saturday", "sunday"]
DAY_TYPE_LABELS: dict[str, str] = {
    "weekday": "Weekday", "saturday": "Saturday", "sunday": "Sunday",
}
PERCENTILES: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95]
PERCENTILE_LABELS: list[str] = ["p5", "p25", "p50", "p75", "p95"]
REFERENCE_YEAR: int = 2025   # 365 days × 24 h = 8760 hours, no DST surprises


def _isodow_to_day_type_sql() -> str:
    """SQL CASE that maps EXTRACT(ISODOW) to our day_type label."""
    return (
        "CASE "
        "WHEN EXTRACT(ISODOW FROM data_misura) BETWEEN 1 AND 5 THEN 'weekday' "
        "WHEN EXTRACT(ISODOW FROM data_misura) = 6 THEN 'saturday' "
        "ELSE 'sunday' END"
    )


def _hourly_kwh_sums() -> list[str]:
    """SQL fragments computing the hourly kWh sum h0..h23 from q1..q96."""
    return [
        "(" + " + ".join(f"q{h * 4 + j + 1}" for j in range(4)) + f") / 1000.0 AS h{h}"
        for h in range(24)
    ]


def _unpivot_select() -> str:
    """24 SELECTs (one per hour) UNION'd into one column."""
    return " UNION ALL ".join(
        f"SELECT pod, month_idx, day_type, {h} AS hour, h{h} AS kwh FROM per_row"
        for h in range(24)
    )


def compute_buckets(
    session:    Session,
    pods:       list[str],
    tipologia:  str = "AP",
    with_percentiles: bool = False,
) -> pd.DataFrame:
    """Aggregate (month, day_type, hour) statistics from `measurements`.

    Always returns mean, std, and sample count. When ``with_percentiles`` is
    True also adds P5/P25/P50/P75/P95 (uses ``PERCENTILE_CONT`` — heavier).

    The result is a long DataFrame with at least
    ``[month, day_type, hour, mean, std, n]`` columns and 12 × 3 × 24 = 864
    rows (one row per non-empty bucket).
    """
    if not pods:
        return pd.DataFrame(
            columns=["month", "day_type", "hour", "mean", "std", "n"]
        )

    hourly_kwh = ", ".join(_hourly_kwh_sums())
    day_type_sql = _isodow_to_day_type_sql()

    extra_cols = ""
    if with_percentiles:
        pct_exprs = ", ".join(
            f"PERCENTILE_CONT({p}) WITHIN GROUP (ORDER BY kwh) AS {lbl}"
            for p, lbl in zip(PERCENTILES, PERCENTILE_LABELS)
        )
        extra_cols = ", " + pct_exprs

    sql = text(f"""
        WITH per_row AS (
            SELECT
                pod,
                EXTRACT(MONTH FROM data_misura)::int AS month_idx,
                {day_type_sql} AS day_type,
                {hourly_kwh}
            FROM measurements
            WHERE tipologia = :tipologia AND pod = ANY(:pods)
        ),
        unpivoted AS ({_unpivot_select()})
        SELECT
            month_idx AS month,
            day_type,
            hour,
            AVG(kwh)    AS mean,
            STDDEV(kwh) AS std,
            COUNT(*)    AS n
            {extra_cols}
        FROM unpivoted
        WHERE kwh IS NOT NULL
        GROUP BY month_idx, day_type, hour
        ORDER BY month_idx, day_type, hour
    """)

    rows = session.execute(
        sql, {"tipologia": tipologia, "pods": list(pods)}
    ).all()
    if not rows:
        return pd.DataFrame(
            columns=["month", "day_type", "hour", "mean", "std", "n"]
        )

    cols = ["month", "day_type", "hour", "mean", "std", "n"]
    if with_percentiles:
        cols += PERCENTILE_LABELS
    df = pd.DataFrame(rows, columns=cols)
    # Postgres returns Decimal for AVG/STDDEV/PERCENTILE — make them floats
    # so plotly and the export layer don't choke downstream.
    for c in ("mean", "std", *PERCENTILE_LABELS):
        if c in df.columns:
            df[c] = df[c].astype(float)
    df["n"] = df["n"].astype(int)
    df["hour"] = df["hour"].astype(int)
    df["month"] = df["month"].astype(int)
    return df


# ── Granularity-specific shaping ────────────────────────────────────────────
def to_annual(buckets: pd.DataFrame) -> pd.DataFrame:
    """Collapse (month, day_type, hour) → (day_type, hour) weighted by n.

    Returns 3 × 24 rows with columns [day_type, hour, mean, std, n]. The
    mean is the n-weighted average across months (i.e. equivalent to a
    direct AVG over all source rows); std is recomputed from the
    pooled-variance identity so it reflects the full-year distribution
    rather than a simple mean of monthly stds.
    """
    if buckets.empty:
        return pd.DataFrame(columns=["day_type", "hour", "mean", "std", "n"])

    def _pool(g: pd.DataFrame) -> pd.Series:
        n_total = int(g["n"].sum())
        if n_total == 0:
            return pd.Series({"mean": np.nan, "std": np.nan, "n": 0})
        mu = float((g["mean"] * g["n"]).sum() / n_total)
        # Pooled variance: weighted sum of within-group variances + between-group
        var = float(
            (g["n"] * (g["std"].fillna(0) ** 2 + (g["mean"] - mu) ** 2)).sum()
            / n_total
        )
        return pd.Series({"mean": mu, "std": float(np.sqrt(max(var, 0.0))),
                          "n": n_total})

    out = (buckets.groupby(["day_type", "hour"], as_index=False)
                  .apply(_pool, include_groups=False))
    return out.reset_index(drop=True)


def to_daily_8760(buckets: pd.DataFrame) -> pd.DataFrame:
    """Expand buckets to a 365 × 24 = 8760-row hourly calendar.

    Each calendar hour of ``REFERENCE_YEAR`` (2025) inherits the mean +
    percentiles of its (month, day_type, hour) bucket. Output columns:
    ``[timestamp, month, day_type, hour, mean, std, n, p5, p25, p50,
    p75, p95]``. Missing buckets (e.g. February 29 in a leap year, or
    months with no data) are emitted as NaN rows so the output is always
    exactly 8760 long.
    """
    start = datetime(REFERENCE_YEAR, 1, 1)
    timestamps = [start + timedelta(hours=h) for h in range(8760)]
    calendar = pd.DataFrame({
        "timestamp": timestamps,
        "month": [t.month for t in timestamps],
        "hour":  [t.hour  for t in timestamps],
        "day_type": [
            "weekday"  if t.isoweekday() <= 5 else
            "saturday" if t.isoweekday() == 6 else "sunday"
            for t in timestamps
        ],
    })
    keep = ["month", "day_type", "hour", "mean", "std", "n"]
    if "p50" in buckets.columns:
        keep += PERCENTILE_LABELS
    merged = calendar.merge(buckets[keep], on=["month", "day_type", "hour"],
                            how="left")
    return merged


# ── Wide reshaping for human-friendly export ────────────────────────────────
def annual_to_wide(annual: pd.DataFrame) -> pd.DataFrame:
    """Wide format for annual: rows = day_type, columns = h0..h23 (mean)."""
    piv = annual.pivot(index="day_type", columns="hour", values="mean")
    piv = piv.reindex(DAY_TYPES)
    piv.columns = [f"h{h}" for h in piv.columns]
    return piv.reset_index().rename(columns={"day_type": "Day Type"})


def monthly_to_wide(buckets: pd.DataFrame) -> pd.DataFrame:
    """Wide format for monthly: rows = (month, day_type), cols = h0..h23."""
    piv = buckets.pivot_table(
        index=["month", "day_type"], columns="hour", values="mean",
    )
    piv.columns = [f"h{h}" for h in piv.columns]
    return piv.reset_index().rename(
        columns={"month": "Month", "day_type": "Day Type"}
    )


def daily_8760_to_wide(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily wide is already wide (one column per statistic per row).

    Just emit a clean column order and drop the helper bucket-keys, which
    would otherwise duplicate information already encoded in the timestamp.
    """
    cols = ["timestamp", "mean", "std", "n"]
    if "p50" in daily.columns:
        cols += PERCENTILE_LABELS
    return daily[cols].copy()
