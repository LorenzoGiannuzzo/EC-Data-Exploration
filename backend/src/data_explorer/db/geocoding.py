"""Nominatim-based geocoding of the Comuni found in `pod_metadata.d_locfo`.

Single source of truth for everything geography-related:

  * `ensure_table()`   — create `comuni_centroids` if it does not exist.
  * `geocode_missing()`— geocode any Comune in `pod_metadata` that doesn't
                         already have a centroid (idempotent).
  * `geocode_in_background()` — entry-point for FastAPI BackgroundTasks
                                 (creates its own session, swallows errors).

The Overview endpoint schedules `geocode_in_background()` automatically
whenever it returns `unlocated` Comuni, so the user never needs to touch
anything: ingest data → refresh the dashboard → coordinates fill in.

Nominatim usage policy (https://operations.osmfoundation.org/policies/nominatim/)
requires ≤ 1 request/second and a real User-Agent.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────
USER_AGENT = os.environ.get(
    "GEOCODE_USER_AGENT",
    "data-explorer-polito/0.1 (contact: lorenzo.giannuzzo@polito.it)",
)
SLEEP_S          = float(os.environ.get("GEOCODE_SLEEP_S", "1.1"))
NOMINATIM_URL    = "https://nominatim.openstreetmap.org/search"
MAX_PER_REQUEST  = int(os.environ.get("GEOCODE_MAX_PER_BATCH", "30"))


# ── Table creation ───────────────────────────────────────────────────────────
DDL = """
CREATE TABLE IF NOT EXISTS comuni_centroids (
    comune_raw   VARCHAR(150) PRIMARY KEY,
    comune_name  VARCHAR(200) NOT NULL,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    provincia    VARCHAR(150),
    regione      VARCHAR(150),
    nazione      VARCHAR(80)  DEFAULT 'Italia',
    source       VARCHAR(80)  DEFAULT 'nominatim',
    raw_response TEXT,
    geocoded_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);
"""


def ensure_table(session: Session) -> None:
    """Idempotently create `comuni_centroids` (cheap when it already exists)."""
    session.execute(text(DDL))
    session.commit()


# ── Nominatim helpers ────────────────────────────────────────────────────────
def _nominatim_search(query: str) -> dict | None:
    params = {
        "q":               f"{query}, Italia",
        "format":          "json",
        "addressdetails":  1,
        "limit":           1,
        "countrycodes":    "it",
        "accept-language": "it",
    }
    url = f"{NOMINATIM_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("Nominatim error for %r: %s", query, e)
        return None
    return data[0] if data else None


def _candidate_queries(raw: str) -> list[str]:
    """Try the literal name first, then progressively shorter forms
    (the source data truncates long names at ~24 chars, e.g.
    'PRIMIERO SAN MARTINO DI C')."""
    raw = raw.strip()
    candidates = [raw]
    words = raw.split()
    while words and len(words[-1]) <= 2:
        words = words[:-1]
        if words:
            candidates.append(" ".join(words))
    return candidates


def _geocode_one(raw_name: str) -> dict | None:
    for q in _candidate_queries(raw_name):
        hit = _nominatim_search(q)
        if hit:
            addr      = hit.get("address", {}) or {}
            provincia = addr.get("county") or addr.get("state_district")
            regione   = addr.get("state")
            nazione   = addr.get("country") or "Italia"
            comune    = (addr.get("city") or addr.get("town")
                          or addr.get("village") or addr.get("municipality")
                          or hit.get("display_name", "").split(",")[0].strip())
            return {
                "comune_name":    comune,
                "lat":            float(hit["lat"]),
                "lon":            float(hit["lon"]),
                "provincia":      provincia,
                "regione":        regione,
                "nazione":        nazione,
                "raw_response":   json.dumps(hit, ensure_ascii=False)[:4000],
                "matched_query":  q,
            }
        time.sleep(SLEEP_S)
    return None


def _upsert(session: Session, raw_name: str, result: dict) -> None:
    session.execute(text("""
        INSERT INTO comuni_centroids (
            comune_raw, comune_name, lat, lon,
            provincia, regione, nazione, source, raw_response
        )
        VALUES (:raw, :name, :lat, :lon,
                :prov, :reg, :naz, 'nominatim', :resp)
        ON CONFLICT (comune_raw) DO UPDATE SET
            comune_name  = EXCLUDED.comune_name,
            lat          = EXCLUDED.lat,
            lon          = EXCLUDED.lon,
            provincia    = EXCLUDED.provincia,
            regione      = EXCLUDED.regione,
            nazione      = EXCLUDED.nazione,
            raw_response = EXCLUDED.raw_response,
            geocoded_at  = CURRENT_TIMESTAMP
    """), {
        "raw":  raw_name,
        "name": result["comune_name"],
        "lat":  result["lat"],
        "lon":  result["lon"],
        "prov": result["provincia"],
        "reg":  result["regione"],
        "naz":  result["nazione"],
        "resp": result["raw_response"],
    })
    session.commit()


# ── In-flight coordination ───────────────────────────────────────────────────
# Avoid two concurrent BackgroundTasks geocoding the same Comune. This is
# a per-process set — good enough since the typical deployment runs a small
# number of workers and Nominatim itself enforces global rate limiting.
_IN_FLIGHT: set[str] = set()
_IN_FLIGHT_LOCK = threading.Lock()


def _try_claim(name: str) -> bool:
    with _IN_FLIGHT_LOCK:
        if name in _IN_FLIGHT:
            return False
        _IN_FLIGHT.add(name)
    return True


def _release(name: str) -> None:
    with _IN_FLIGHT_LOCK:
        _IN_FLIGHT.discard(name)


# ── Public API ───────────────────────────────────────────────────────────────
def _list_missing(session: Session) -> list[str]:
    """Comuni present in pod_metadata.d_locfo that aren't yet in
    comuni_centroids."""
    rows = session.execute(text("""
        SELECT DISTINCT UPPER(TRIM(p.d_locfo)) AS raw
        FROM pod_metadata p
        LEFT JOIN comuni_centroids c
          ON UPPER(TRIM(p.d_locfo)) = c.comune_raw
        WHERE p.d_locfo IS NOT NULL
          AND p.d_locfo <> ''
          AND c.comune_raw IS NULL
    """)).all()
    return [r.raw for r in rows]


def geocode_missing(
    session: Session,
    raw_names: list[str] | None = None,
    max_count: int | None = None,
    force: bool = False,
) -> dict:
    """Geocode any Comune that isn't already in `comuni_centroids`.

    Args:
        session:    DB session (already created by the caller).
        raw_names:  Restrict to a specific subset (defaults to "everything
                    missing in pod_metadata").
        max_count:  Cap the number of Nominatim requests per call. Useful
                    when called from an HTTP handler so a single request
                    never spends too long on geocoding.
        force:      Re-geocode even Comuni already present in the table.

    Returns: dict with counts of geocoded / failed / skipped.
    """
    ensure_table(session)

    if raw_names is None:
        candidates = _list_missing(session) if not force else \
                      [r.raw for r in session.execute(text(
                          "SELECT DISTINCT UPPER(TRIM(d_locfo)) AS raw "
                          "FROM pod_metadata WHERE d_locfo IS NOT NULL "
                          "AND d_locfo <> ''")).all()]
    else:
        candidates = [n.strip().upper() for n in raw_names if n and n.strip()]

    if max_count is not None:
        candidates = candidates[:max_count]

    ok, fail, skipped = 0, [], 0
    for raw in candidates:
        # Skip if already geocoded (race-condition-safe re-check)
        if not force:
            already = session.execute(
                text("SELECT 1 FROM comuni_centroids WHERE comune_raw = :n"),
                {"n": raw},
            ).scalar()
            if already:
                skipped += 1
                continue

        if not _try_claim(raw):
            skipped += 1
            continue

        try:
            logger.info("Geocoding %r …", raw)
            result = _geocode_one(raw)
            if result is None:
                fail.append(raw)
                logger.warning("No Nominatim match for %r", raw)
            else:
                _upsert(session, raw, result)
                ok += 1
                logger.info("Geocoded %r → %s (%.4f, %.4f)",
                             raw, result["comune_name"],
                             result["lat"], result["lon"])
        finally:
            _release(raw)
        time.sleep(SLEEP_S)

    return {
        "geocoded": ok,
        "failed":   fail,
        "skipped":  skipped,
        "total":    len(candidates),
    }


def geocode_in_background(raw_names: list[str] | None = None) -> None:
    """Entry-point for FastAPI BackgroundTasks. Creates its own session,
    caps the batch, and swallows any error so the background runner is
    never the source of a failed request."""
    from data_explorer.db.session import SessionLocal
    try:
        with SessionLocal() as session:
            result = geocode_missing(
                session,
                raw_names=raw_names,
                max_count=MAX_PER_REQUEST,
            )
        logger.info("Background geocode finished: %s", result)
    except Exception:
        logger.exception("Background geocode failed")
