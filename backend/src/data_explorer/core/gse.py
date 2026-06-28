"""GSE profile comparison — replicates the original dashboard methodology.

Two normalisation modes:
    - Monorario (PDMM, PAUM, PIRM, PACM, MDMM, MAUM):
        each hour expressed as % of monthly consumption (sum over 24h × n_days ≈ 100%)
    - Fasce / Time-of-Use (PDMF, PAUF, PIRF, PACF, MDMF, MAUF):
        each hour expressed as % of monthly consumption of *its own ARERA band*
        (F1 = Mon-Fri 08-18, F2 = Mon-Fri 07,19-22 + Sat 07-22, F3 = nights/Sunday)

The dispatch happens automatically based on the profile_code suffix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

Q_COLS = [f"q{i}" for i in range(1, 97)]

# Profile codes that use the in-fasce (time-of-use) normalisation
FASCIA_CODES = {"PDMF", "PAUF", "PIRF", "PACF", "MDMF", "MAUF"}


def _hourly_sum_expr() -> str:
    """SQL expression list: 24 hourly kWh values built from the 96 quarter-hour
    columns. Each h_N = (q[4N+1] + q[4N+2] + q[4N+3] + q[4N+4]) / 1000 (kWh).

    Each q_i goes through `COALESCE(NULLIF(q_i::text, 'NaN')::float8, 0)`,
    which turns both NULL *and* the IEEE-754 NaN sentinel into 0. The plain
    COALESCE() is not enough: in PostgreSQL NaN is not NULL, so it survives
    COALESCE, propagates through any arithmetic, and a single NaN row is
    enough to make the column-wide AVG() collapse to NaN. We learned this
    the hard way on Lorenzo's dataset, where a handful of NaN entries in
    Weekday and Sunday rows poisoned every aggregation for those day-types
    while Saturday (NaN-free in the same bucket) computed fine.

    The result is then wrapped in NULLIF(_, 0) so a quarter-hour that is
    recorded as a true zero (the ingestion layer stores absent measurements
    that way rather than as NULL) is reported as NULL, which makes downstream
    AVG()s skip it instead of dragging the result toward zero.
    """
    def safe(n: int) -> str:
        return f"COALESCE(NULLIF(q{n}::text, 'NaN')::float8, 0)"
    return ", ".join(
        f"NULLIF(({safe(4*h+1)}+{safe(4*h+2)}"
        f"+{safe(4*h+3)}+{safe(4*h+4)})/1000.0, 0) AS h{h}"
        for h in range(24)
    )


# ── Monorario: dashboard's `compute_our_normalized_profiles` ─────────────────
def _dayset_condition(dayset: str) -> str:
    """SQL boolean condition on an ``iso_dow`` column for a dayset."""
    if dayset == "weekday":
        return "iso_dow BETWEEN 1 AND 5"
    if dayset == "weekend":
        return "iso_dow >= 6"
    if dayset == "all":
        return "TRUE"
    raise ValueError(f"Unknown dayset: {dayset!r}")


def _fetch_monorario(
    session: Session,
    pods:    list[str],
    dayset:  str,
    tipologia: str,
) -> pd.DataFrame:
    """Per-month % of monthly consumption per hour (matches GSE monorario scale).

    IMPORTANT (legacy parity / GSE convention): the monthly-total denominator
    is computed over ALL days of the month, exactly like the original
    dashboard's `compute_our_normalized_profiles`. The ``dayset`` selection
    only restricts which rows enter the final hourly average (the numerator).
    Filtering the denominator too would shrink the "monthly total" to the
    weekday-only (or weekend-only) total and inflate every percentage.
    """
    day_cond = _dayset_condition(dayset)

    # SQL plan:
    #   1. per_row: hourly kWh per (pod, day), all days
    #   2. monthly_total: SUM of all hourly energy per (pod, month) — the true
    #      monthly consumption (equals the legacy avg_daily_total * n_days)
    #   3. numer: avg kWh per hour per (pod, month) over dayset-selected days
    #   4. with_pct: avg_h / monthly_total * 100
    #   5. final: average pct across PODs per (month, hour)
    sum_24 = " + ".join(f"COALESCE(h{h}, 0)" for h in range(24))
    avg_per_hour = ", ".join(f"AVG(h{h}) AS avg_h{h}" for h in range(24))
    sql = text(
        f"""
        WITH per_row AS (
            SELECT pod, EXTRACT(MONTH FROM data_misura)::int AS month_idx,
                   EXTRACT(ISODOW FROM data_misura)::int AS iso_dow,
                   {_hourly_sum_expr()}
            FROM measurements
            WHERE tipologia = :tipologia AND pod = ANY(:pods)
        ),
        -- Note: no row-level zero-filter here. The NULLIF inside
        -- _hourly_sum_expr() already gives us hour-level missingness,
        -- so AVG()s downstream naturally skip absent hours instead of
        -- being dragged to zero. Filtering at row level would also
        -- discard partial days where only a subset of hours is missing.
        monthly_total AS (
            SELECT pod, month_idx, SUM({sum_24}) AS tot_mo
            FROM per_row
            GROUP BY pod, month_idx
        ),
        numer AS (
            SELECT pod, month_idx, {avg_per_hour}
            FROM per_row
            WHERE {day_cond}
            GROUP BY pod, month_idx
        ),
        with_pct AS (
            SELECT n.pod, n.month_idx,
                {", ".join(f"n.avg_h{h} / NULLIF(t.tot_mo, 0) * 100 AS p{h}"
                            for h in range(24))}
            FROM numer n
            JOIN monthly_total t USING (pod, month_idx)
        )
        SELECT month_idx, {", ".join(f"AVG(p{h}) AS h{h}" for h in range(24))}
        FROM with_pct
        GROUP BY month_idx
        ORDER BY month_idx
        """
    )
    rows = session.execute(sql, {"tipologia": tipologia, "pods": pods}).all()
    return _rows_to_long(rows)


# ── Fascia: dashboard's `compute_our_fascia_profiles` ────────────────────────
def _hour_to_fascia_pct_expr(h: int) -> str:
    """SQL expression: per-hour pct using the band-monthly denominator.

    Italian ARERA bands:
        F1 = Mon-Fri (iso_dow 1-5) & 08 <= h <= 18
        F2 = Mon-Fri & h == 7 or 19 <= h <= 22; Saturday (6) & 07 <= h <= 22
        F3 = everything else (Sunday, nights)
    """
    if 8 <= h <= 18:
        return (
            f"CASE "
            f"WHEN iso_dow BETWEEN 1 AND 5 THEN h{h} / NULLIF(f1_mo,0) * 100 "
            f"WHEN iso_dow = 6 THEN h{h} / NULLIF(f2_mo,0) * 100 "
            f"ELSE h{h} / NULLIF(f3_mo,0) * 100 END AS p{h}"
        )
    if h == 7 or 19 <= h <= 22:
        return (
            f"CASE "
            f"WHEN iso_dow BETWEEN 1 AND 6 THEN h{h} / NULLIF(f2_mo,0) * 100 "
            f"ELSE h{h} / NULLIF(f3_mo,0) * 100 END AS p{h}"
        )
    # h in {0..6, 23} → always F3
    return f"h{h} / NULLIF(f3_mo,0) * 100 AS p{h}"


def _fetch_fascia(
    session: Session,
    pods:    list[str],
    dayset:  str,
    tipologia: str,
) -> pd.DataFrame:
    """Per-month % of band-monthly consumption per hour (matches GSE fascia scale).

    IMPORTANT (legacy parity / GSE convention): the F1/F2/F3 band-monthly
    denominators are computed over ALL days of the month, exactly like the
    original dashboard's `compute_our_fascia_profiles`. The ``dayset``
    selection only restricts which rows enter the final hourly average.
    Filtering the denominators too would drop Saturday from the F2 total and
    Sunday from the F3 total, inflating evening/night percentages by ~1.7-1.8x
    (while F1 — weekdays only by definition — would deceptively still match).
    """
    day_cond = _dayset_condition(dayset)

    f1_day = "(" + "+".join(f"COALESCE(h{h},0)" for h in range(8, 19)) + ")"
    f2_day_weekday = "(COALESCE(h7,0)+" + "+".join(
        f"COALESCE(h{h},0)" for h in range(19, 23)) + ")"
    f2_day_sat     = "(" + "+".join(
        f"COALESCE(h{h},0)" for h in range(7, 23)) + ")"
    f3_day_weekday = "(" + "+".join(
        f"COALESCE(h{h},0)" for h in list(range(0, 7)) + [23]) + ")"
    f3_day_sat     = f3_day_weekday
    f3_day_sun     = "(" + "+".join(
        f"COALESCE(h{h},0)" for h in range(0, 24)) + ")"

    pct_exprs = ", ".join(_hour_to_fascia_pct_expr(h) for h in range(24))

    sql = text(
        f"""
        WITH per_row AS (
            SELECT pod, EXTRACT(MONTH FROM data_misura)::int AS month_idx,
                   EXTRACT(ISODOW FROM data_misura)::int AS iso_dow,
                   {_hourly_sum_expr()}
            FROM measurements
            WHERE tipologia = :tipologia AND pod = ANY(:pods)
        ),
        -- No row-level filter: NULLIF inside _hourly_sum_expr() already
        -- gives us hour-level missingness, so AVG()s and band SUMs treat
        -- absent hours correctly without us throwing away partial days.
        per_row_with_fascia AS (
            SELECT *,
                CASE WHEN iso_dow BETWEEN 1 AND 5 THEN {f1_day} ELSE 0 END AS f1_d,
                CASE
                    WHEN iso_dow BETWEEN 1 AND 5 THEN {f2_day_weekday}
                    WHEN iso_dow = 6 THEN {f2_day_sat}
                    ELSE 0
                END AS f2_d,
                CASE
                    WHEN iso_dow = 7 THEN {f3_day_sun}
                    WHEN iso_dow = 6 THEN {f3_day_sat}
                    ELSE {f3_day_weekday}
                END AS f3_d
            FROM per_row
        ),
        monthly_fascia AS (
            SELECT pod, month_idx,
                SUM(f1_d) AS f1_mo, SUM(f2_d) AS f2_mo, SUM(f3_d) AS f3_mo
            FROM per_row_with_fascia
            GROUP BY pod, month_idx
        ),
        joined AS (
            SELECT p.*, m.f1_mo, m.f2_mo, m.f3_mo
            FROM per_row_with_fascia p
            JOIN monthly_fascia m USING (pod, month_idx)
        ),
        per_row_pct AS (
            SELECT pod, month_idx, iso_dow, {pct_exprs}
            FROM joined
        )
        SELECT month_idx, {", ".join(f"AVG(p{h}) AS h{h}" for h in range(24))}
        FROM per_row_pct
        WHERE {day_cond}
        GROUP BY month_idx
        ORDER BY month_idx
        """
    )
    rows = session.execute(sql, {"tipologia": tipologia, "pods": pods}).all()
    return _rows_to_long(rows)


def _smooth_holes(long: pd.DataFrame) -> pd.DataFrame:
    """Per-month linear interpolation of zero-valued holes.

    A "hole" is an hour whose value collapsed to zero while neighbouring hours
    are positive — typically a side-effect of sparse data and per-POD averaging.
    We replace those zeros with the linear interpolation of the neighbours.
    """
    if long.empty:
        return long
    out = long.copy().sort_index()
    months = out.index.get_level_values("month_idx").unique()
    for m in months:
        try:
            sub = out.xs(m, level="month_idx")
        except KeyError:
            continue
        s = sub["value"].astype(float)
        zeros = s == 0
        if not zeros.any() or zeros.all():
            continue  # nothing to fill or nothing to interpolate from
        s2 = s.where(~zeros, np.nan)
        s2 = s2.interpolate(method="linear", limit_direction="both").fillna(0.0)
        out.loc[(m, slice(None)), "value"] = s2.values
    return out


# ── Common reshape ───────────────────────────────────────────────────────────
def _rows_to_long(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["value"])
    cols = ["month_idx"] + [f"h{h}" for h in range(24)]
    df = pd.DataFrame(rows, columns=cols)
    long = df.melt(
        id_vars="month_idx",
        value_vars=[f"h{h}" for h in range(24)],
        var_name="hour_idx", value_name="value",
    )
    long["hour_idx"] = long["hour_idx"].str[1:].astype(int)
    long["value"]    = long["value"].astype(float)
    # Drop months where every hour is NaN (no real data after the
    # zero-energy-row filter in the SQL). A blanket `.fillna(0.0)` here would
    # turn those NaN-months into bogus flat zero lines in the chart.
    valid_months = (
        long.groupby("month_idx")["value"]
            .apply(lambda s: s.notna().any())
    )
    long = long[long["month_idx"].isin(
        valid_months.index[valid_months]
    )].copy()
    # Within still-valid months, sporadic NaNs are isolated holes — treat
    # them as zeros so `_smooth_holes` interpolates them.
    long["value"] = long["value"].fillna(0.0)
    long = long.set_index(["month_idx", "hour_idx"]).sort_index()
    return _smooth_holes(long)


# ── Public dispatcher ────────────────────────────────────────────────────────
def fetch_our_gse_normalised_profile(
    session:      Session,
    pods:         list[str] | set[str],
    dayset:       str = "weekday",
    profile_code: str = "PDMM",
    tipologia:    str = "AP",
) -> pd.DataFrame:
    """Pick the right normalisation based on the GSE profile_code:
    F* → in-fasce (TOU) normalisation, others → monorario.
    """
    if not pods:
        return pd.DataFrame(columns=["value"])
    pods = list(pods)
    if profile_code in FASCIA_CODES:
        return _fetch_fascia(session, pods, dayset, tipologia)
    return _fetch_monorario(session, pods, dayset, tipologia)


# ── Reference fetch ──────────────────────────────────────────────────────────
def fetch_gse_reference_profile(
    session:      Session,
    profile_code: str,
) -> pd.DataFrame:
    rows = session.execute(
        text(
            "SELECT month_idx, hour_idx, value "
            "FROM reference_profiles_gse "
            "WHERE profile_code = :code "
            "ORDER BY month_idx, hour_idx"
        ),
        {"code": profile_code},
    ).all()
    if not rows:
        return pd.DataFrame(columns=["value"])
    df = pd.DataFrame(rows, columns=["month_idx", "hour_idx", "value"])
    df["value"] = df["value"].astype(float)
    return df.set_index(["month_idx", "hour_idx"]).sort_index()


# ── Metrics ──────────────────────────────────────────────────────────────────
def compare_to_gse(
    our:       pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    if our.empty or reference.empty:
        return pd.DataFrame(columns=["rmse", "mae", "max_abs_err", "bias"])

    joined = (
        our.rename(columns={"value": "ours"})
        .join(reference.rename(columns={"value": "ref"}), how="inner")
    )
    if joined.empty:
        return pd.DataFrame(columns=["rmse", "mae", "max_abs_err", "bias"])

    joined["err"] = joined["ours"] - joined["ref"]
    return (
        joined.groupby(level="month_idx")
        .agg(
            rmse       =("err", lambda s: float(np.sqrt(np.mean(s ** 2)))),
            mae        =("err", lambda s: float(np.mean(np.abs(s)))),
            max_abs_err=("err", lambda s: float(np.max(np.abs(s)))),
            bias       =("err", lambda s: float(np.mean(s))),
        )
    )
