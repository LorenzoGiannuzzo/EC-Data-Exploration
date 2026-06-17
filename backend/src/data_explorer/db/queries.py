"""SQL query helpers.

All heavy data-shaping is done in PostgreSQL (`GROUP BY`, `AVG`, …) so that
Python only receives the already-aggregated result. This is the central
advantage of moving from Parquet caches to a real database.

Every function takes a SQLAlchemy `Session` as input and returns either a
DataFrame, a Series, or a plain list/dict. No Streamlit, no FastAPI imports.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

Q_COLS = [f"q{i}" for i in range(1, 97)]


# ── Aggregated profiles (the core query) ─────────────────────────────────────
def _view_exists(session: Session) -> bool:
    """Check whether `pod_avg_profile` materialized view exists."""
    return bool(session.execute(text(
        "SELECT to_regclass('public.pod_avg_profile')"
    )).scalar())


def fetch_aggregated_profiles(
    session: Session,
    pod_ids: set[str] | list[str] | None = None,
    tipologia: str = "AP",
    month: int = 0,
) -> pd.DataFrame:
    """Per-POD average daily profile (96 quarter-hour values).

    ``month`` — 0 = annual average (all days); 1..12 = restrict the average
    to a single calendar month (legacy dashboard "Profile period" selector).

    Uses the `pod_avg_profile` materialized view when present (≪1 s for any
    POD subset), otherwise falls back to an on-the-fly GROUP BY on
    `measurements` (~10 s on 4.3 M rows). The view only stores the annual
    average, so single-month requests always go through the GROUP BY path.
    """
    q_cols = ", ".join(Q_COLS)
    if month and not 1 <= month <= 12:
        raise ValueError(f"month must be 0 (annual) or 1..12, got {month}")
    if month:
        q_avg = ", ".join(f"AVG({q}) AS {q}" for q in Q_COLS)
        month_clause = "AND EXTRACT(MONTH FROM data_misura) = :month "
        if pod_ids:
            sql = text(
                f"SELECT pod, {q_avg} FROM measurements "
                "WHERE tipologia = :tipologia AND pod = ANY(:pod_ids) "
                f"{month_clause}GROUP BY pod"
            )
            params = {"tipologia": tipologia, "pod_ids": list(pod_ids),
                      "month": int(month)}
        else:
            sql = text(
                f"SELECT pod, {q_avg} FROM measurements "
                f"WHERE tipologia = :tipologia {month_clause}GROUP BY pod"
            )
            params = {"tipologia": tipologia, "month": int(month)}
        rows = session.execute(sql, params).all()
        if not rows:
            return pd.DataFrame(columns=Q_COLS)
        return pd.DataFrame(rows, columns=["pod", *Q_COLS]).set_index("pod")

    if _view_exists(session):
        if pod_ids:
            sql = text(
                f"SELECT pod, {q_cols} FROM pod_avg_profile "
                "WHERE tipologia = :tipologia AND pod = ANY(:pod_ids)"
            )
            params = {"tipologia": tipologia, "pod_ids": list(pod_ids)}
        else:
            sql = text(
                f"SELECT pod, {q_cols} FROM pod_avg_profile "
                "WHERE tipologia = :tipologia"
            )
            params = {"tipologia": tipologia}
    else:
        q_avg = ", ".join(f"AVG({q}) AS {q}" for q in Q_COLS)
        if pod_ids:
            sql = text(
                f"SELECT pod, {q_avg} "
                "FROM measurements "
                "WHERE tipologia = :tipologia AND pod = ANY(:pod_ids) "
                "GROUP BY pod"
            )
            params = {"tipologia": tipologia, "pod_ids": list(pod_ids)}
        else:
            sql = text(
                f"SELECT pod, {q_avg} "
                "FROM measurements "
                "WHERE tipologia = :tipologia "
                "GROUP BY pod"
            )
            params = {"tipologia": tipologia}

    rows = session.execute(sql, params).all()
    if not rows:
        return pd.DataFrame(columns=Q_COLS)
    df = pd.DataFrame(rows, columns=["pod", *Q_COLS]).set_index("pod")
    return df


# ── Coverage filter (≥ N months of data) ─────────────────────────────────────
def fetch_pods_with_data_coverage(
    session: Session,
    min_months: int = 12,
    tipologia: str = "AP",
) -> set[str]:
    """Return PODs that have at least ``min_months`` distinct calendar months
    of measurements for the given tipologia."""
    sql = text(
        "SELECT pod "
        "FROM measurements "
        "WHERE tipologia = :tipologia "
        "GROUP BY pod "
        "HAVING COUNT(DISTINCT date_trunc('month', data_misura)) >= :min_months"
    )
    rows = session.execute(
        sql, {"tipologia": tipologia, "min_months": min_months}
    ).all()
    return {r[0] for r in rows}


# ── POD selection by contractual power ───────────────────────────────────────
def fetch_pods_by_power(
    session: Session,
    ranges: list[tuple[float, float | None]],
    include_missing: bool = False,
) -> set[str]:
    """Return PODs whose contractual power (``pod_metadata.d_pot1``, kW)
    falls inside at least one of the supplied ``(min_kw, max_kw]`` ranges.

    Bin convention mirrors the legacy dashboard's ``pd.cut(right=True)``:
    a POD belongs to a range when ``min_kw < d_pot1 <= max_kw``.
    ``max_kw = None`` means an open upper bound (e.g. ">110 kW").
    ``include_missing`` additionally keeps PODs with ``d_pot1 IS NULL``.
    """
    conds: list[str] = []
    params: dict = {}
    for i, (lo, hi) in enumerate(ranges):
        if hi is None:
            conds.append(f"(d_pot1 > :lo{i})")
            params[f"lo{i}"] = float(lo)
        else:
            conds.append(f"(d_pot1 > :lo{i} AND d_pot1 <= :hi{i})")
            params[f"lo{i}"] = float(lo)
            params[f"hi{i}"] = float(hi)
    if include_missing:
        conds.append("(d_pot1 IS NULL)")
    if not conds:
        return set()
    sql = text(f"SELECT pod FROM pod_metadata WHERE {' OR '.join(conds)}")
    return {r[0] for r in session.execute(sql, params).all()}


# ── POD selection by market zone ─────────────────────────────────────────────
def fetch_pods_by_zones(
    session: Session, zone_codes: list[str],
) -> set[str]:
    """Return PODs whose Comune resolves to one of the given market zones.

    Resolution: `pod_metadata.d_locfo` → `comuni_centroids.regione` →
    market zone (via the Python map in `core.market_zones`). PODs whose
    Comune is not geocoded yet — or whose region is outside the 7 zones —
    are excluded.
    """
    if not zone_codes:
        return set()
    from data_explorer.core.market_zones import regions_for_zone
    regions: list[str] = []
    for z in zone_codes:
        regions.extend(regions_for_zone(z))
    if not regions:
        return set()
    sql = text(
        """
        SELECT p.pod
        FROM pod_metadata p
        JOIN comuni_centroids c
          ON UPPER(TRIM(p.d_locfo)) = c.comune_raw
        WHERE c.regione = ANY(:regions)
        """
    )
    return {r[0] for r in session.execute(sql, {"regions": regions}).all()}


def fetch_zone_availability(session: Session) -> dict[str, int]:
    """POD count per market zone, across all PODs that have a geocoded Comune.

    Returns a mapping ``{zone_code: n_pods}`` populated only for zones that
    actually have at least one POD. The frontend uses this to render the
    "✓ available" tick next to each zone label and gray out the empty ones.
    """
    from data_explorer.core.market_zones import region_to_zone
    sql = text(
        """
        SELECT c.regione, COUNT(DISTINCT p.pod) AS n
        FROM pod_metadata p
        JOIN comuni_centroids c
          ON UPPER(TRIM(p.d_locfo)) = c.comune_raw
        WHERE c.regione IS NOT NULL
        GROUP BY c.regione
        """
    )
    counts: dict[str, int] = {}
    for region, n in session.execute(sql).all():
        zone = region_to_zone(region)
        if zone is None:
            continue
        counts[zone] = counts.get(zone, 0) + int(n)
    return counts


# ── POD selection by ATECO ────────────────────────────────────────────────────
def fetch_pods_by_ateco(
    session: Session,
    codes: list[str],
    level: int = 1,
) -> set[str]:
    """Return PODs whose ATECO code at the given hierarchy level matches one
    of the supplied codes."""
    if not codes:
        return set()
    col = {1: "ateco_l1", 2: "ateco_l2", 3: "ateco_l3"}.get(level)
    if col is None:
        raise ValueError(f"Invalid ATECO level: {level} (must be 1, 2 or 3)")
    sql = text(f"SELECT pod FROM pod_metadata WHERE {col} = ANY(:codes)")
    rows = session.execute(sql, {"codes": list(codes)}).all()
    return {r[0] for r in rows}


def fetch_ateco_codes(session: Session, level: int = 1) -> list[str]:
    """List all distinct ATECO codes seen at the given level in pod_metadata."""
    col = {1: "ateco_l1", 2: "ateco_l2", 3: "ateco_l3"}.get(level)
    if col is None:
        raise ValueError(f"Invalid ATECO level: {level}")
    sql = text(
        f"SELECT DISTINCT {col} FROM pod_metadata "
        f"WHERE {col} IS NOT NULL ORDER BY {col}"
    )
    return [r[0] for r in session.execute(sql).all()]


def fetch_ateco_pod_counts(
    session:   Session,
    level:     int = 1,
    tipologia: str = "AP",
    min_months: int = 0,
) -> dict[str, int]:
    """POD count per ATECO code at the given level, after coverage filter.

    Returns ``{code: n_pods_with_min_months_of_data}`` — the dict only
    contains codes with at least one matching POD, so the frontend can use
    it both to list "available" codes and to render the count badge inline.
    When ``min_months == 0`` the count is the raw pod_metadata count
    (coverage filter disabled).
    """
    col = {1: "ateco_l1", 2: "ateco_l2", 3: "ateco_l3"}.get(level)
    if col is None:
        raise ValueError(f"Invalid ATECO level: {level}")
    if min_months <= 0:
        sql = text(
            f"SELECT {col} AS code, COUNT(*) AS n FROM pod_metadata "
            f"WHERE {col} IS NOT NULL GROUP BY {col}"
        )
        rows = session.execute(sql).all()
    else:
        # Subquery: PODs with ≥ min_months distinct months of measurements
        sql = text(
            f"""
            WITH covered AS (
                SELECT pod
                FROM measurements
                WHERE tipologia = :tipologia
                GROUP BY pod
                HAVING COUNT(DISTINCT DATE_TRUNC('month', data_misura))
                       >= :min_months
            )
            SELECT p.{col} AS code, COUNT(*) AS n
            FROM pod_metadata p
            JOIN covered c USING (pod)
            WHERE p.{col} IS NOT NULL
            GROUP BY p.{col}
            """
        )
        rows = session.execute(sql, {"tipologia": tipologia,
                                      "min_months": int(min_months)}).all()
    return {r[0]: int(r[1]) for r in rows}


def fetch_ateco_subcodes(
    session:          Session,
    parent_l1:        list[str] | None,
    parent_l2:        list[str] | None,
    target_level:     int,
    tipologia:        str = "AP",
    min_months:       int = 0,
) -> dict[str, int]:
    """ATECO codes at ``target_level`` whose POD descends from the given parents.

    The hierarchy is read directly from ``pod_metadata.ateco_l1`` /
    ``ateco_l2`` (derived during ingestion), so a POD is a child of L1='DO'
    iff its own ateco_l1 column equals 'DO'. Returns ``{code: n_pods}`` after
    the optional coverage filter; only codes with at least one matching POD
    are included.
    """
    if target_level not in (2, 3):
        raise ValueError(f"target_level must be 2 or 3 (got {target_level})")
    parent_filters: list[str] = []
    params: dict = {"tipologia": tipologia, "min_months": int(min_months)}
    if parent_l1:
        parent_filters.append("p.ateco_l1 = ANY(:parents_l1)")
        params["parents_l1"] = list(parent_l1)
    if parent_l2 and target_level == 3:
        parent_filters.append("p.ateco_l2 = ANY(:parents_l2)")
        params["parents_l2"] = list(parent_l2)
    where_parents = " AND ".join(parent_filters) if parent_filters else "TRUE"
    col = f"ateco_l{target_level}"

    if min_months <= 0:
        sql = text(
            f"SELECT p.{col} AS code, COUNT(*) AS n "
            f"FROM pod_metadata p "
            f"WHERE p.{col} IS NOT NULL AND ({where_parents}) "
            f"GROUP BY p.{col}"
        )
    else:
        sql = text(
            f"""
            WITH covered AS (
                SELECT pod
                FROM measurements
                WHERE tipologia = :tipologia
                GROUP BY pod
                HAVING COUNT(DISTINCT DATE_TRUNC('month', data_misura))
                       >= :min_months
            )
            SELECT p.{col} AS code, COUNT(*) AS n
            FROM pod_metadata p
            JOIN covered c USING (pod)
            WHERE p.{col} IS NOT NULL AND ({where_parents})
            GROUP BY p.{col}
            """
        )
    return {r[0]: int(r[1]) for r in session.execute(sql, params).all()}


# ── Metadata fetch ────────────────────────────────────────────────────────────
def fetch_pod_metadata(
    session: Session,
    pod_ids: set[str] | list[str] | None = None,
) -> pd.DataFrame:
    """Return the pod_metadata table as a DataFrame (filtered if pod_ids given)."""
    if pod_ids:
        sql = text("SELECT * FROM pod_metadata WHERE pod = ANY(:pod_ids)")
        rows = session.execute(sql, {"pod_ids": list(pod_ids)}).mappings().all()
    else:
        rows = session.execute(text("SELECT * FROM pod_metadata")).mappings().all()
    return pd.DataFrame(rows)


# ── ATECO lookup ──────────────────────────────────────────────────────────────
def fetch_ateco_descriptions(session: Session) -> dict[str, str]:
    """Return a ``{code: description}`` dict for every ATECO entry."""
    rows = session.execute(
        text("SELECT code, description FROM ateco_lookup")
    ).all()
    return {code: (desc or "") for code, desc in rows}


# ── Tipologie distinte ─────────────────────────────────────────────────────────
def fetch_distinct_tipologie(session: Session) -> list[str]:
    """All distinct measurement type codes present in measurements."""
    sql = text("SELECT DISTINCT tipologia FROM measurements "
               "WHERE tipologia IS NOT NULL ORDER BY tipologia")
    return [r[0] for r in session.execute(sql).all()]


# ── Per-POD data-coverage gap stats ──────────────────────────────────────────
def fetch_pod_data_gaps(
    session: Session, pod_ids: list[str] | set[str],
    tipologia: str = "AP",
) -> pd.DataFrame:
    """For each POD: days with data, expected days (full date range),
    missing days, and missing %."""
    if not pod_ids:
        return pd.DataFrame(columns=[
            "pod", "days_with_data", "expected_days", "missing_days", "missing_pct"
        ])
    sql = text("""
        SELECT pod,
               COUNT(DISTINCT data_misura)                        AS days_with_data,
               (MAX(data_misura)::date - MIN(data_misura)::date) + 1 AS expected_days
        FROM measurements
        WHERE tipologia = :tipologia AND pod = ANY(:pods)
        GROUP BY pod
    """)
    rows = session.execute(
        sql, {"tipologia": tipologia, "pods": list(pod_ids)}
    ).all()
    if not rows:
        return pd.DataFrame(columns=[
            "pod", "days_with_data", "expected_days", "missing_days", "missing_pct"
        ])
    df = pd.DataFrame(rows, columns=["pod", "days_with_data", "expected_days"])
    df["missing_days"] = (df["expected_days"] - df["days_with_data"]).clip(lower=0)
    df["missing_pct"]  = (df["missing_days"] / df["expected_days"].clip(lower=1) * 100).round(2)
    return df


# ── Counts (used by /info, db-check, smoke tests) ────────────────────────────
def fetch_table_counts(session: Session) -> dict[str, int]:
    """Row counts for every main table — useful for health and smoke tests."""
    tables = (
        "pod_metadata", "measurements", "ateco_lookup",
        "reference_profiles_gse", "reference_profiles_arera",
    )
    counts: dict[str, int] = {}
    for t in tables:
        counts[t] = session.execute(text(f"SELECT count(*) FROM {t}")).scalar_one()
    return counts
