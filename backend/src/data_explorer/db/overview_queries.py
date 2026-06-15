"""Pure-SQL helpers backing the /overview/* endpoints."""

from __future__ import annotations

import math
import re

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from data_explorer.db.geocoding import ensure_table as _ensure_centroid_table


# ─────────────────────────────────────────────────────────────────────────────
# ATECO catalogue coverage
# ─────────────────────────────────────────────────────────────────────────────
# pod_metadata stores ATECO at three SEMANTIC levels that don't line up
# 1-to-1 with the catalogue levels — the dataset skips both the SECTION
# (catalogue level=1, alphabetic A..V) and the GROUP (catalogue level=3,
# "01.1"-style). Mapping:
#
#     pod_metadata column   |  semantic level  |  catalogue level
#     ───────────────────── | ──────────────── | ─────────────────
#     ateco_l1  ("25")       |  Division         |  2
#     ateco_l2  ("25.11")    |  Class            |  4
#     ateco_l3  ("25.11.00") |  Subcategory      |  6
_LEVEL_MAP = [
    (1, "ateco_l1", 2, "Division",    "Divisions"),
    (2, "ateco_l2", 4, "Class",       "Classes"),
    (3, "ateco_l3", 6, "Subcategory", "Subcategories"),
]


def fetch_ateco_coverage(session: Session) -> dict:
    out: dict = {}
    for key, col, cat_level, name, name_plural in _LEVEL_MAP:
        present_rows = session.execute(text(
            f"SELECT {col} AS code, COUNT(*) AS n "
            f"FROM pod_metadata "
            f"WHERE {col} IS NOT NULL AND {col} <> '' "
            f"GROUP BY {col} ORDER BY n DESC"
        )).all()
        present_df = pd.DataFrame(present_rows, columns=["code", "n_pods"])

        cat_rows = session.execute(
            text("SELECT code FROM ateco_lookup WHERE level = :lvl"),
            {"lvl": cat_level},
        ).all()
        catalogue = {r[0] for r in cat_rows}

        present_set = set(present_df["code"].tolist())
        in_catalogue_and_present = present_set & catalogue
        missing_in_catalogue     = sorted(catalogue - present_set)
        out_of_catalogue_codes   = sorted(present_set - catalogue)

        coverage_pct = (
            len(in_catalogue_and_present) / len(catalogue) * 100.0
            if catalogue else 0.0
        )

        out[key] = {
            "level_key":        key,
            "catalogue_level":  cat_level,
            "name":             name,
            "name_plural":      name_plural,
            "n_codes_present_total":        len(present_set),
            "n_codes_present_in_catalogue": len(in_catalogue_and_present),
            "n_codes_out_of_catalogue":     len(out_of_catalogue_codes),
            "n_codes_in_catalogue":         len(catalogue),
            "coverage_pct":                 coverage_pct,
            "n_pods_per_code": [
                {"code": str(r.code), "n_pods": int(r.n_pods)}
                for r in present_df.itertuples(index=False)
            ],
            "missing_codes":          missing_in_catalogue[:400],
            "n_missing":              len(missing_in_catalogue),
            "out_of_catalogue_codes": out_of_catalogue_codes[:400],
            "n_out_of_catalogue":     len(out_of_catalogue_codes),
            "n_codes_present":        len(present_set),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Power-class distribution
# ─────────────────────────────────────────────────────────────────────────────
_POWER_BINS = [
    (0.0,  1.5,  "0–1.5 kW"),
    (1.5,  3.0,  "1.5–3 kW"),
    (3.0,  4.5,  "3–4.5 kW"),
    (4.5,  6.0,  "4.5–6 kW"),
    (6.0, 10.0,  "6–10 kW"),
    (10.0, 15.0, "10–15 kW"),
    (15.0, 30.0, "15–30 kW"),
    (30.0, float("inf"), ">30 kW"),
]


def _parse_power(raw) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def fetch_power_class_distribution(session: Session) -> dict:
    sql = text("""
        SELECT DISTINCT ON (pod) pod, potenza_contrattuale
        FROM measurements
        WHERE potenza_contrattuale IS NOT NULL
        ORDER BY pod, data_misura DESC
    """)
    rows = session.execute(sql).all()

    n_pods_total = int(session.execute(
        text("SELECT COUNT(*) FROM pod_metadata")
    ).scalar() or 0)

    counts = {label: 0 for *_x, label in _POWER_BINS}
    unparseable = 0
    for _, raw in rows:
        v = _parse_power(raw)
        if v is None:
            unparseable += 1
            continue
        for lo, hi, label in _POWER_BINS:
            if lo <= v < hi:
                counts[label] += 1
                break

    n_with_meas    = len(rows)
    n_without_meas = max(n_pods_total - n_with_meas, 0)
    return {
        "n_pods_total":         n_pods_total,
        "n_pods_with_meas":     n_with_meas,
        "n_pods_without_meas":  n_without_meas,
        "n_pods_unparseable":   unparseable,
        "by_class":             [{"label": l, "n_pods": n} for l, n in counts.items()],
        "n_pods":               n_with_meas,
        "n_unknown":            n_without_meas,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Consumption distribution
# ─────────────────────────────────────────────────────────────────────────────
def _view_exists(session: Session) -> bool:
    return bool(session.execute(text(
        "SELECT to_regclass('public.pod_avg_profile')"
    )).scalar())


def _empty_consumption() -> dict:
    return {
        "n_pods":      0,
        "daily_kwh":   [], "monthly_kwh": [], "annual_kwh": [],
        "stats": {"daily_mean": 0.0, "daily_median": 0.0,
                   "daily_p10": 0.0, "daily_p90": 0.0},
    }


def _safe_float(x) -> float:
    if x is None:
        return 0.0
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def fetch_consumption_distribution(session: Session, tipologia: str = "AP") -> dict:
    q_sum = " + ".join(f"COALESCE(q{i}, 0)" for i in range(1, 97))
    if _view_exists(session):
        sql = text(f"""
            SELECT pod, ({q_sum})/4.0 AS daily_kwh
            FROM pod_avg_profile WHERE tipologia = :tip
        """)
    else:
        avg = ", ".join(f"AVG(COALESCE(q{i}, 0)) AS q{i}" for i in range(1, 97))
        sql = text(f"""
            WITH per_pod AS (
                SELECT pod, {avg}
                FROM measurements WHERE tipologia = :tip
                GROUP BY pod
            )
            SELECT pod, ({q_sum})/4.0 AS daily_kwh FROM per_pod
        """)

    rows = session.execute(sql, {"tip": tipologia}).all()
    if not rows:
        return _empty_consumption()

    daily_values = [_safe_float(r.daily_kwh) for r in rows]
    daily_values = [v for v in daily_values if v > 0]
    if not daily_values:
        return _empty_consumption()

    daily = pd.Series(daily_values, name="d")
    return {
        "n_pods":      int(len(daily)),
        "daily_kwh":   daily.round(3).tolist(),
        "monthly_kwh": (daily * 30.0).round(2).tolist(),
        "annual_kwh":  (daily * 365.0).round(0).tolist(),
        "stats": {
            "daily_mean":   _safe_float(daily.mean()),
            "daily_median": _safe_float(daily.median()),
            "daily_p10":    _safe_float(daily.quantile(0.10)),
            "daily_p90":    _safe_float(daily.quantile(0.90)),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geographic distribution — every coordinate comes from the centroid table
# (populated automatically by `db.geocoding.geocode_in_background`).
# ─────────────────────────────────────────────────────────────────────────────
def fetch_geography_distribution(session: Session) -> dict:
    # Make sure the lookup table exists, regardless of whether geocoding has
    # ever run. After this the JOIN always works (just returns NULLs when
    # nothing has been geocoded yet).
    _ensure_centroid_table(session)

    rows = session.execute(text("""
        SELECT
            UPPER(TRIM(p.d_locfo))             AS raw_name,
            COUNT(*)                            AS n_pods,
            c.comune_name, c.lat, c.lon,
            c.provincia, c.regione
        FROM pod_metadata p
        LEFT JOIN comuni_centroids c
          ON UPPER(TRIM(p.d_locfo)) = c.comune_raw
        WHERE p.d_locfo IS NOT NULL AND p.d_locfo <> ''
        GROUP BY raw_name, c.comune_name, c.lat, c.lon, c.provincia, c.regione
        ORDER BY n_pods DESC
    """)).all()

    located:   list[dict] = []
    unlocated: list[dict] = []
    for r in rows:
        if r.lat is not None and r.lon is not None:
            located.append({
                "comune":    r.comune_name or r.raw_name.title(),
                "raw_name":  r.raw_name,
                "lat":       float(r.lat),
                "lon":       float(r.lon),
                "provincia": r.provincia,
                "regione":   r.regione,
                "n_pods":    int(r.n_pods),
            })
        else:
            unlocated.append({
                "comune":   r.raw_name.title(),
                "raw_name": r.raw_name,
                "n_pods":   int(r.n_pods),
            })

    if located:
        lats  = [c["lat"] for c in located]
        lons  = [c["lon"] for c in located]
        lat_c = sum(lats) / len(lats)
        lon_c = sum(lons) / len(lons)
        span  = max(max(lats) - min(lats), max(lons) - min(lons), 0.05)
        zoom  = float(max(6.0, min(13.0, 11.0 - 2.0 * span)))
    else:
        lat_c, lon_c, zoom = 42.5, 12.5, 5.5   # Italy fallback

    # Scope label — derived purely from what's in the geocoded table.
    provs = sorted({c["provincia"] for c in located if c.get("provincia")})
    regs  = sorted({c["regione"]   for c in located if c.get("regione")})

    if not located and unlocated:
        scope_label = (f"Geocoding in progress — {len(unlocated)} Comuni "
                        "still being resolved. Refresh in a few seconds.")
    elif not located:
        scope_label = "No geographic data available yet."
    elif len(provs) == 1 and len(regs) == 1:
        scope_label = (f"{len(located)} Comuni in Provincia di {provs[0]} "
                        f"({regs[0]})")
    elif len(provs) == 1:
        scope_label = f"{len(located)} Comuni in Provincia di {provs[0]}"
    elif len(regs) == 1:
        scope_label = (f"{len(located)} Comuni across {len(provs)} "
                        f"provinces in {regs[0]}")
    else:
        scope_label = (f"{len(located)} Comuni across {len(provs)} "
                        f"provinces and {len(regs)} regions")

    return {
        "located":      located,
        "unlocated":    unlocated,
        "n_comuni":     len(located),
        "n_pods_geo":   sum(c["n_pods"] for c in located),
        "n_pods_no_geo": sum(c["n_pods"] for c in unlocated),
        "provinces":    provs,
        "regions":      regs,
        "map_view":     {"center_lat": lat_c, "center_lon": lon_c, "zoom": zoom},
        "scope_label":  scope_label,
    }
