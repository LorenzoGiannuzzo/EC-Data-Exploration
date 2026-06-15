"""GSE reference-profile comparison endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from data_explorer.api.deps import monthly_profile_to_dict, resolve_pod_set
from data_explorer.api.schemas import (
    GseCompareRequest, GseCompareResponse, GseProfileList, MonthlyMetrics,
)
from data_explorer.core.gse import (
    compare_to_gse, fetch_gse_reference_profile, fetch_our_gse_normalised_profile,
)
from data_explorer.db.session import get_session

router = APIRouter(prefix="/gse", tags=["gse"])


@router.get("/profiles", response_model=GseProfileList)
def list_profiles(db: Session = Depends(get_session)):
    """Distinct GSE profile codes currently in the DB (PDMM, PDMF, …)."""
    rows = db.execute(
        text("SELECT DISTINCT profile_code "
             "FROM reference_profiles_gse "
             "ORDER BY profile_code")
    ).all()
    return GseProfileList(profile_codes=[r[0] for r in rows])


@router.post("/compare", response_model=GseCompareResponse)
def compare(req: GseCompareRequest, db: Session = Depends(get_session)):
    """Compare the aggregated PoliTo profile of a POD set against one GSE
    reference column. Returns both profiles (per month) and per-month metrics
    (RMSE, MAE, bias, max |err|) in percentage points."""
    pods, _ = resolve_pod_set(db, req.filter)
    if not pods:
        raise HTTPException(400, detail="No PODs matched the filter.")

    ours = fetch_our_gse_normalised_profile(
        db, pods, dayset=req.dayset, profile_code=req.profile_code,
    )
    ref  = fetch_gse_reference_profile(db, req.profile_code)
    if ref.empty:
        raise HTTPException(
            404,
            detail=f"No GSE reference data for profile {req.profile_code!r}.",
        )

    metrics_df = compare_to_gse(ours, ref)
    metrics = [
        MonthlyMetrics(
            month=int(m),
            rmse=float(r.rmse),
            mae=float(r.mae),
            bias=float(r.bias),
            max_abs_err=float(r.max_abs_err),
        )
        for m, r in metrics_df.iterrows()
    ]

    return GseCompareResponse(
        n_pods=len(pods),
        profile_code=req.profile_code,
        our_profile=monthly_profile_to_dict(ours),
        reference_profile=monthly_profile_to_dict(ref),
        metrics=metrics,
    )
