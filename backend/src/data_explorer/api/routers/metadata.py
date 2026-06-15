"""Metadata endpoints — ATECO catalogue and POD-set previews."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from data_explorer.api.deps import resolve_pod_set
from data_explorer.api.schemas import (
    AtecoCodeList, AtecoDescriptions, PodFilter, PodSetSummary, TipologieList,
)
from data_explorer.db.queries import (
    fetch_ateco_codes, fetch_ateco_descriptions, fetch_distinct_tipologie,
)
from data_explorer.db.session import get_session

router = APIRouter(prefix="/metadata", tags=["metadata"])


@router.get("/ateco", response_model=AtecoCodeList)
def list_ateco_codes(
    level: int = Query(default=1, ge=1, le=3, description="1, 2 or 3."),
    db:    Session = Depends(get_session),
):
    """All distinct ATECO codes present in `pod_metadata` at the given level."""
    return AtecoCodeList(level=level, codes=fetch_ateco_codes(db, level=level))


@router.get("/ateco/descriptions", response_model=AtecoDescriptions)
def list_ateco_descriptions(db: Session = Depends(get_session)):
    """Full `code → description` map from the ATECO lookup table."""
    return AtecoDescriptions(descriptions=fetch_ateco_descriptions(db))


@router.get("/tipologie", response_model=TipologieList)
def list_tipologie(db: Session = Depends(get_session)):
    """All distinct measurement-type codes present in measurements."""
    return TipologieList(tipologie=fetch_distinct_tipologie(db))


@router.post("/pod-set", response_model=PodSetSummary)
def preview_pod_set(
    f:  PodFilter,
    db: Session = Depends(get_session),
):
    """How many PODs a filter would select — without running clustering/comparison.

    Useful to populate a "n PODs selected" counter in the UI while the user
    fiddles with filters.
    """
    _, summary = resolve_pod_set(db, f)
    return PodSetSummary(**summary)
