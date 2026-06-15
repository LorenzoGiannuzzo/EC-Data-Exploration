"""Overview endpoints — population-level statistics."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session

from data_explorer.db.geocoding import geocode_in_background
from data_explorer.db.overview_queries import (
    fetch_ateco_coverage,
    fetch_consumption_distribution,
    fetch_geography_distribution,
    fetch_power_class_distribution,
)
from data_explorer.db.session import get_session

router = APIRouter(prefix="/overview", tags=["overview"])


@router.get("/ateco-coverage")
def ateco_coverage(db: Session = Depends(get_session)):
    """Per-level ATECO catalogue coverage and per-code POD counts.

    The three "levels" map to the SEMANTIC levels stored in pod_metadata
    (Division / Class / Subcategory), i.e. catalogue levels 2 / 4 / 6.
    """
    return fetch_ateco_coverage(db)


@router.get("/power-class-distribution")
def power_class_distribution(db: Session = Depends(get_session)):
    """POD counts bucketed by contractual power class."""
    return fetch_power_class_distribution(db)


@router.get("/consumption-distribution")
def consumption_distribution(
    tipologia: str = Query(default="AP",
                           description="Measurement type code: AP, AN, RLP…"),
    db:        Session = Depends(get_session),
):
    """Per-POD daily/monthly/annual consumption (in kWh)."""
    return fetch_consumption_distribution(db, tipologia=tipologia)


@router.get("/geography")
def geography(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
):
    """POD counts per Comune with attached centroids.

    Any Comune still missing a centroid triggers a background geocoding
    task (Nominatim, rate-limited per OSM policy). The user never needs to
    run anything manually — the next refresh of the dashboard will show
    the newly-geocoded points.
    """
    payload = fetch_geography_distribution(db)
    unlocated = payload.get("unlocated") or []
    if unlocated:
        # Schedule a non-blocking geocode pass. The task creates its own
        # session, swallows errors, and is capped at GEOCODE_MAX_PER_BATCH
        # Comuni per HTTP request so latency stays bounded.
        background_tasks.add_task(
            geocode_in_background,
            [u["raw_name"] for u in unlocated],
        )
    return payload
