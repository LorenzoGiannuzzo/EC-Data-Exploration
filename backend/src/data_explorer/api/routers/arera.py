"""ARERA reference-profile comparison endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from data_explorer.api.deps import resolve_pod_set
from data_explorer.api.schemas import (
    AreraCompareAllRequest, AreraCompareAllResponse, AreraCompareRequest,
    AreraCompareResponse, AreraDayPanel, AreraKey, AreraKeyList,
    AreraPowerClassList,
)
from data_explorer.core.arera import (
    compare_to_arera, fetch_arera_reference, fetch_our_arera_profile,
)
from data_explorer.db.session import get_session

router = APIRouter(prefix="/arera", tags=["arera"])


# ── Discovery endpoints ──────────────────────────────────────────────────────
@router.get("/power-classes", response_model=AreraPowerClassList)
def list_power_classes(db: Session = Depends(get_session)):
    """Distinct ARERA power-class labels currently in the DB."""
    rows = db.execute(
        text("SELECT DISTINCT power_class FROM reference_profiles_arera "
             "ORDER BY power_class")
    ).all()
    return AreraPowerClassList(power_classes=[r[0] for r in rows])


@router.get("/keys", response_model=AreraKeyList)
def list_keys(db: Session = Depends(get_session)):
    """All (power_class, market, residenza, province, day_type) combinations
    present in the DB. Used to populate cascading dropdowns in the UI."""
    rows = db.execute(
        text(
            "SELECT DISTINCT power_class, market, residenza, province, day_type "
            "FROM reference_profiles_arera "
            "ORDER BY power_class, market, residenza, province, day_type"
        )
    ).all()
    keys = [
        AreraKey(
            power_class=r[0], market=r[1], residenza=r[2],
            province=r[3], day_type=r[4],
        )
        for r in rows
    ]
    return AreraKeyList(keys=keys)


def _normalise_power_class(s: str) -> str:
    """Power-class strings include an en-dash (–, U+2013) but PowerShell/curl
    sometimes downgrade it to a plain ASCII '-'. Normalise to a single form."""
    return s.replace("-", "–").replace("--", "–") if s else s


# ── Compare ───────────────────────────────────────────────────────────────────
def _filter_pods_by_power_class(
    db: Session, pods: set[str], power_class: str,
) -> set[str]:
    """Apply the ARERA power-class bucket using the latest ``potenza_contrattuale``
    value seen in ``measurements`` (same source as the original dashboard).

    The CSV stores PotenzaContrattuale as a string with Italian decimal comma.
    We pick the most recent value per POD, convert to float, then bucket.
    """
    if not pods:
        return set()
    power_class = _normalise_power_class(power_class)
    rows = db.execute(
        text(
            "SELECT DISTINCT ON (pod) pod, potenza_contrattuale "
            "FROM measurements "
            "WHERE pod = ANY(:pods) AND potenza_contrattuale IS NOT NULL "
            "ORDER BY pod, data_misura DESC"
        ),
        {"pods": list(pods)},
    ).all()
    if not rows:
        return set()
    df = pd.DataFrame(rows, columns=["pod", "potcontr"])
    df["kw"] = pd.to_numeric(
        df["potcontr"].astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )
    df = df.dropna(subset=["kw"])
    buckets = {
        "≤ 1.5 kW": (df["kw"] > 0)   & (df["kw"] <= 1.5),
        "1.5–3 kW": (df["kw"] > 1.5) & (df["kw"] <= 3),
        "3–4.5 kW": (df["kw"] > 3)   & (df["kw"] <= 4.5),
        "4.5–6 kW": (df["kw"] > 4.5) & (df["kw"] <= 6),
        "> 6 kW":   (df["kw"] > 6),
    }
    mask = buckets.get(power_class)
    if mask is None:
        raise HTTPException(400, detail=f"Unknown power_class: {power_class!r}")
    return set(df.loc[mask, "pod"])


@router.post("/compare", response_model=AreraCompareResponse)
def compare(req: AreraCompareRequest, db: Session = Depends(get_session)):
    """Compare the hourly kWh profile of a POD set (filtered by ATECO, coverage
    and ARERA power class) against the ARERA reference for the requested key."""
    pods, _ = resolve_pod_set(db, req.filter)
    power_class = _normalise_power_class(req.power_class)
    pods = _filter_pods_by_power_class(db, pods, power_class)
    if not pods:
        raise HTTPException(400, detail="No PODs match the filter for this power class.")

    ours = fetch_our_arera_profile(
        db, pods, day_type=req.day_type, month_idx=req.month,
        tipologia=req.filter.tipologia,
    )
    ref = fetch_arera_reference(
        db, power_class=power_class, market=req.market,
        residenza=req.residenza, day_type=req.day_type,
        month_idx=req.month, province=req.province,
    )
    if ref.empty:
        raise HTTPException(
            404,
            detail="No ARERA reference data for the requested "
                   "(power_class, market, residenza, province, day_type, month).",
        )

    metrics = compare_to_arera(ours, ref)
    return AreraCompareResponse(
        n_pods=len(pods),
        our_profile=[float(ours.get(h, 0.0))      for h in range(24)],
        reference_profile=[float(ref.get(h, 0.0)) for h in range(24)],
        metrics=metrics,
    )


@router.post("/compare-all-day-types", response_model=AreraCompareAllResponse)
def compare_all_day_types(
    req: AreraCompareAllRequest, db: Session = Depends(get_session),
):
    """Compare the PoliTo profile against ARERA for *all three* day types
    (Weekday, Saturday, Sunday) in a single call. Front-ends use this to
    render three side-by-side charts."""
    pods, _ = resolve_pod_set(db, req.filter)
    power_class = _normalise_power_class(req.power_class)
    pods = _filter_pods_by_power_class(db, pods, power_class)
    if not pods:
        raise HTTPException(400, detail="No PODs match the filter for this power class.")

    panels: list[AreraDayPanel] = []
    for dt in ("Weekday", "Saturday", "Sunday"):
        ours = fetch_our_arera_profile(
            db, pods, day_type=dt, month_idx=req.month,
            tipologia=req.filter.tipologia,
        )
        ref = fetch_arera_reference(
            db, power_class=power_class, market=req.market,
            residenza=req.residenza, day_type=dt,
            month_idx=req.month, province=req.province,
        )
        panels.append(AreraDayPanel(
            day_type=dt,
            our_profile=[float(ours.get(h, 0.0))      for h in range(24)],
            reference_profile=[float(ref.get(h, 0.0)) for h in range(24)],
            metrics=compare_to_arera(ours, ref),
        ))
    return AreraCompareAllResponse(n_pods=len(pods), panels=panels)
