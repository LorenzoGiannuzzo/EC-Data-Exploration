"""/load-profiler/* — bucket-based hourly load profile extraction.

Endpoints:
    GET  /load-profiler/zones           — market zone catalogue + availability
    POST /load-profiler/run             — compute profiles for a POD selection

The "Load Profiler" tab is designed for users who want representative load
profiles for a specific destinazione d'uso × contractual-power × geographic
slice — not exploratory clustering. The output is therefore deterministic
bucket statistics, never k-means/hierarchical labels.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from data_explorer.api.deps import resolve_pod_set
from data_explorer.api.schemas import (
    AtecoAvailability, AtecoAvailabilityResponse,
    LoadProfilerRequest, LoadProfilerResponse, LoadProfilerSelection,
    PodFilter, ZoneAvailability, ZoneListResponse,
)
from data_explorer.core.load_profiler import (
    compute_buckets, to_annual, to_daily_8760,
)
from data_explorer.core.market_zones import ZONE_LABELS, ZONE_ORDER
from data_explorer.db.queries import (
    fetch_ateco_descriptions, fetch_ateco_pod_counts, fetch_ateco_subcodes,
    fetch_pods_by_ateco, fetch_pods_by_zones, fetch_pods_with_data_coverage,
    fetch_zone_availability,
)
from data_explorer.db.session import get_session

router = APIRouter(prefix="/load-profiler", tags=["load-profiler"])


@router.get("/zones", response_model=ZoneListResponse)
def list_zones(db: Session = Depends(get_session)) -> ZoneListResponse:
    counts = fetch_zone_availability(db)
    zones = [
        ZoneAvailability(
            code=z, label=ZONE_LABELS[z],
            n_pods=int(counts.get(z, 0)),
            available=counts.get(z, 0) > 0,
        )
        for z in ZONE_ORDER
    ]
    return ZoneListResponse(zones=zones, n_geocoded=sum(counts.values()))


@router.get("/ateco-availability", response_model=AtecoAvailabilityResponse)
def ateco_availability(
    level:      int = 1,
    tipologia:  str = "AP",
    min_months: int = 12,
    db: Session = Depends(get_session),
) -> AtecoAvailabilityResponse:
    """ATECO codes for ``level`` that actually have PODs after coverage filter.

    The frontend uses this to populate the Load Profiler's searchable
    multiselects with codes that are guaranteed to return at least one POD,
    plus an inline POD count for context. This is what makes the L1
    selection always work — without it, picking a code with all PODs below
    ``min_months`` of coverage yields a confusing "0 PODs" error.
    """
    if level not in (1, 2, 3):
        raise HTTPException(400, "level must be 1, 2 or 3")
    counts = fetch_ateco_pod_counts(
        db, level=level, tipologia=tipologia, min_months=min_months,
    )
    descs = fetch_ateco_descriptions(db)
    rows = sorted(
        [
            AtecoAvailability(
                code=code, level=level, n_pods=n,
                description=descs.get(code),
            )
            for code, n in counts.items() if n > 0
        ],
        key=lambda r: (-r.n_pods, r.code),
    )
    return AtecoAvailabilityResponse(level=level, codes=rows)


@router.get("/ateco-subcodes", response_model=AtecoAvailabilityResponse)
def ateco_subcodes(
    target_level: int,
    parent_l1:    str = "",
    parent_l2:    str = "",
    tipologia:    str = "AP",
    min_months:   int = 12,
    db: Session = Depends(get_session),
) -> AtecoAvailabilityResponse:
    """Sub-codes that descend from selected L1 (and optionally L2) parents.

    ``parent_l1`` / ``parent_l2`` are comma-separated code lists ("DO,47").
    Empty strings mean "no parent constraint at that level". Without any
    parent the response is identical to ``/ateco-availability``.
    """
    if target_level not in (2, 3):
        raise HTTPException(400, "target_level must be 2 or 3")
    p1 = [c.strip() for c in parent_l1.split(",") if c.strip()] or None
    p2 = [c.strip() for c in parent_l2.split(",") if c.strip()] or None
    if not (p1 or p2):
        # Mirror the level-N counts when no parents are constrained.
        counts = fetch_ateco_pod_counts(
            db, level=target_level, tipologia=tipologia, min_months=min_months,
        )
    else:
        counts = fetch_ateco_subcodes(
            db, parent_l1=p1, parent_l2=p2, target_level=target_level,
            tipologia=tipologia, min_months=min_months,
        )
    descs = fetch_ateco_descriptions(db)
    rows = sorted(
        [
            AtecoAvailability(
                code=code, level=target_level, n_pods=n,
                description=descs.get(code),
            )
            for code, n in counts.items() if n > 0
        ],
        key=lambda r: (-r.n_pods, r.code),
    )
    return AtecoAvailabilityResponse(level=target_level, codes=rows)


@router.post("/diagnose")
def diagnose(
    req: LoadProfilerRequest, db: Session = Depends(get_session),
) -> dict:
    """Return POD-set sizes after each filter step independently.

    This is the introspection endpoint behind the "Why is my result empty?"
    expander in the UI: it shows how many PODs survive each filter applied
    in isolation, plus the pairwise and full intersections, so the user can
    pinpoint which constraint is killing the selection.
    """
    coverage = fetch_pods_with_data_coverage(
        db, min_months=req.min_months, tipologia=req.tipologia,
    )

    ateco_pods: set[str] | None = None
    if req.ateco_l1 or req.ateco_l2 or req.ateco_l3:
        ateco_pods = set()
        if req.ateco_l1:
            ateco_pods |= fetch_pods_by_ateco(db, req.ateco_l1, level=1)
        if req.ateco_l2:
            ateco_pods |= fetch_pods_by_ateco(db, req.ateco_l2, level=2)
        if req.ateco_l3:
            ateco_pods |= fetch_pods_by_ateco(db, req.ateco_l3, level=3)

    zone_pods: set[str] | None = None
    zone_pods_no_geocode_drop = 0
    if req.zones:
        zone_pods = fetch_pods_by_zones(db, req.zones)
        # How many candidate PODs would the geocoding gap drop?
        if ateco_pods is not None:
            zone_pods_no_geocode_drop = len(
                coverage & ateco_pods - zone_pods
            )

    result = {
        "coverage_only":     len(coverage),
        "ateco_only":        len(ateco_pods) if ateco_pods is not None else None,
        "zone_only":         len(zone_pods) if zone_pods is not None else None,
        "coverage_ateco":    (len(coverage & ateco_pods)
                              if ateco_pods is not None else len(coverage)),
        "coverage_zone":     (len(coverage & zone_pods)
                              if zone_pods is not None else len(coverage)),
        "ateco_zone":        (len(ateco_pods & zone_pods)
                              if ateco_pods is not None and zone_pods is not None
                              else None),
        "final_intersection": len(
            coverage
            & (ateco_pods if ateco_pods is not None else coverage)
            & (zone_pods  if zone_pods  is not None else coverage)
        ),
        "geocoding_gap_drop": zone_pods_no_geocode_drop,
        "filter_params": {
            "tipologia":  req.tipologia,
            "min_months": req.min_months,
            "ateco_l1":   req.ateco_l1, "ateco_l2": req.ateco_l2,
            "ateco_l3":   req.ateco_l3,
            "zones":      req.zones,
        },
    }
    return result


@router.post("/run", response_model=LoadProfilerResponse)
def run_load_profiler(
    req: LoadProfilerRequest, db: Session = Depends(get_session)
) -> LoadProfilerResponse:
    pod_filter = PodFilter(
        ateco_l1=req.ateco_l1, ateco_l2=req.ateco_l2, ateco_l3=req.ateco_l3,
        min_months=req.min_months, tipologia=req.tipologia,
        power_ranges=req.power_ranges,
        include_missing_power=req.include_missing_power,
    )
    pods, summary = resolve_pod_set(db, pod_filter)

    after_zone: int | None = None
    if req.zones:
        zone_pods = fetch_pods_by_zones(db, req.zones)
        pods = pods & zone_pods
        after_zone = len(zone_pods)

    if not pods:
        raise HTTPException(
            status_code=400,
            detail="No PODs match the supplied filter combination. Relax the "
                   "ATECO, power, or market-zone selection and try again.",
        )

    want_pct = req.granularity == "daily"
    buckets  = compute_buckets(db, list(pods), tipologia=req.tipologia,
                               with_percentiles=want_pct)
    if buckets.empty:
        raise HTTPException(
            status_code=400,
            detail=f"{len(pods)} POD(s) matched but produced no measurement "
                   "rows. Likely all-zero data — check the coverage filter.",
        )

    selection = LoadProfilerSelection(
        n_pods=len(pods),
        after_coverage=summary["after_coverage"],
        after_ateco=summary["after_ateco"],
        after_power=summary["after_power"],
        after_zone=after_zone,
    )

    payload = LoadProfilerResponse(
        selection=selection,
        granularity=req.granularity,
        buckets=buckets.to_dict(orient="records"),
    )

    if req.granularity == "annual":
        payload.annual = to_annual(buckets).to_dict(orient="records")
    elif req.granularity == "daily":
        daily = to_daily_8760(buckets)
        # Serialize timestamps as ISO strings to keep the JSON parseable
        # by the Streamlit client without extra dependencies.
        daily = daily.copy()
        daily["timestamp"] = daily["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        payload.daily_8760 = daily.to_dict(orient="records")
    return payload
