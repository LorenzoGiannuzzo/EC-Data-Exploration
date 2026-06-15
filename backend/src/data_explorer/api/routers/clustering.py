"""Clustering endpoints — runs the full clustering pipeline on filtered PODs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from data_explorer.api.deps import resolve_pod_set
from data_explorer.api.schemas import (
    AtecoBreakdownEntry, ClusterMetrics, ClusteringByLevelRequest,
    ClusteringByLevelResponse, ClusteringLevelResult, ClusteringRequest,
    ClusteringResponse, OutlierPodGap, OutlierRequest, OutlierResponse,
    PodFilter, PodSetSummary,
)
from data_explorer.core.clustering import (
    centroid_pearson_matrix, cluster_ateco_breakdown, cluster_metrics,
    cluster_profiles, compute_centroids, dominant_ateco_per_cluster,
    find_optimal_k,
)
from data_explorer.core.outliers import detect_outliers
from data_explorer.core.profiles import filter_all_zero, normalise_profiles
from data_explorer.db.queries import (
    fetch_aggregated_profiles, fetch_ateco_descriptions, fetch_pod_data_gaps,
    fetch_pod_metadata,
)
from data_explorer.db.session import get_session

router = APIRouter(prefix="/clustering", tags=["clustering"])


@router.post("/preview-size", response_model=PodSetSummary)
def preview_size(f: PodFilter, db: Session = Depends(get_session)):
    """Same as ``/metadata/pod-set`` — exposed under /clustering for UX symmetry."""
    _, summary = resolve_pod_set(db, f)
    return PodSetSummary(**summary)


# ── Helper that runs a single clustering and returns a ClusteringResponse ────
def _do_clustering(
    db: Session,
    pod_filter:    PodFilter,
    n_clusters:    int,
    method:        str,
    normalise:     str,
    top_ateco:     int,
    ateco_level_for_breakdown: int = 1,
    month:         int = 0,
    auto_k:        bool = False,
) -> ClusteringResponse:
    pods, _ = resolve_pod_set(db, pod_filter)
    if len(pods) < n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(pods)} POD(s) match the filter — cannot form "
                   f"{n_clusters} clusters.",
        )

    profiles = fetch_aggregated_profiles(
        db, pod_ids=pods, tipologia=pod_filter.tipologia, month=month,
    )
    profiles = filter_all_zero(profiles)
    if len(profiles) < n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(profiles)} POD(s) have a non-zero profile"
                   + (f" for month {month}." if month else "."),
        )

    normalised = normalise_profiles(profiles, method=normalise)

    auto_k_details: dict | None = None
    if auto_k:
        n_clusters, auto_k_details = find_optimal_k(normalised, method=method)

    clusters   = cluster_profiles(normalised, n_clusters=n_clusters, method=method)
    centroids  = compute_centroids(normalised, clusters)
    metrics    = cluster_metrics(normalised, clusters)
    metadata   = fetch_pod_metadata(db, pod_ids=list(clusters.index))
    breakdown  = cluster_ateco_breakdown(clusters, metadata, ateco_level=ateco_level_for_breakdown)
    if top_ateco > 0:
        breakdown = dominant_ateco_per_cluster(breakdown, top_n=top_ateco)
    descriptions = fetch_ateco_descriptions(db)
    pearson      = centroid_pearson_matrix(centroids)

    return ClusteringResponse(
        n_pods=len(clusters),
        metrics=ClusterMetrics(**metrics),
        centroids={
            int(cl): [float(v) for v in row]
            for cl, row in centroids.iterrows()
        },
        assignments={str(pod): int(cl) for pod, cl in clusters.items()},
        ateco_breakdown=[
            AtecoBreakdownEntry(
                cluster=int(r.cluster), ateco=r.ateco,
                description=(descriptions.get(r.ateco) if r.ateco else None),
                n_pods=int(r.n_pods),
                pct_of_cluster=float(r.pct_of_cluster),
            )
            for r in breakdown.itertuples(index=False)
        ],
        pearson_matrix=(
            pearson.values.tolist() if not pearson.empty else None
        ),
        pearson_labels=[int(c) for c in pearson.index] if not pearson.empty else [],
        auto_k_details=auto_k_details,
    )


@router.post("/run", response_model=ClusteringResponse)
def run_clustering(req: ClusteringRequest, db: Session = Depends(get_session)):
    """Full clustering pipeline on the supplied filter (single panel)."""
    return _do_clustering(
        db, req.filter, n_clusters=req.n_clusters, method=req.method,
        normalise=req.normalise, top_ateco=req.top_ateco_per_cluster,
        month=req.month, auto_k=req.auto_k,
    )


@router.post("/run-by-level", response_model=ClusteringByLevelResponse)
def run_by_level(
    req: ClusteringByLevelRequest, db: Session = Depends(get_session),
):
    """Run a separate clustering for each ATECO level that has a non-empty
    selection. Matches the legacy dashboard, which produced a centroid panel
    per active level."""
    out: list[ClusteringLevelResult] = []

    level_selections: list[tuple[int, list[str]]] = []
    if req.ateco_l1: level_selections.append((1, req.ateco_l1))
    if req.ateco_l2: level_selections.append((2, req.ateco_l2))
    if req.ateco_l3: level_selections.append((3, req.ateco_l3))
    if not level_selections:
        raise HTTPException(
            400, detail="No ATECO codes selected at any level.",
        )

    for level, codes in level_selections:
        f = PodFilter(
            ateco_l1=codes if level == 1 else None,
            ateco_l2=codes if level == 2 else None,
            ateco_l3=codes if level == 3 else None,
            min_months=req.min_months,
            tipologia=req.tipologia,
            power_ranges=req.power_ranges,
            include_missing_power=req.include_missing_power,
        )
        try:
            resp = _do_clustering(
                db, f, n_clusters=req.n_clusters, method=req.method,
                normalise=req.normalise, top_ateco=req.top_ateco_per_cluster,
                ateco_level_for_breakdown=level,
                month=req.month, auto_k=req.auto_k,
            )
        except HTTPException as e:
            # Skip this level if there aren't enough PODs; record an empty result
            resp = ClusteringResponse(
                n_pods=0,
                metrics=ClusterMetrics(
                    n_pods=0, n_clusters=0, silhouette=None,
                    calinski_harabasz=None, davies_bouldin=None, sizes={},
                ),
                centroids={}, assignments={}, ateco_breakdown=[],
                pearson_matrix=None, pearson_labels=[],
            )
        out.append(ClusteringLevelResult(
            level=level, ateco_filter=codes, response=resp,
        ))

    return ClusteringByLevelResponse(results=out)


@router.post("/outliers", response_model=OutlierResponse)
def outliers(req: OutlierRequest, db: Session = Depends(get_session)):
    """Single-linkage outlier detection. Replicates the legacy dashboard
    `outliers_detection_tab` semantics."""
    pods, _ = resolve_pod_set(db, req.filter)
    if len(pods) < req.n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(pods)} POD(s) match the filter — need at least "
                   f"{req.n_clusters} to form that many clusters.",
        )

    profiles = fetch_aggregated_profiles(
        db, pod_ids=pods, tipologia=req.filter.tipologia,
    )
    profiles = filter_all_zero(profiles)
    if len(profiles) < req.n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(profiles)} POD(s) have a non-zero profile.",
        )

    normalised = normalise_profiles(profiles, method=req.normalise)
    result = detect_outliers(
        normalised, k=req.n_clusters, threshold=req.threshold,
    )

    # Per-POD gap stats only for the outlier set
    out_ids   = result["outlier_pods"]
    gaps      = fetch_pod_data_gaps(db, out_ids, tipologia=req.filter.tipologia)
    gaps_map  = gaps.set_index("pod").to_dict("index") if not gaps.empty else {}
    metadata  = fetch_pod_metadata(db, pod_ids=out_ids)
    meta_map  = metadata.set_index("pod").to_dict("index") if not metadata.empty else {}
    assignments = result["cluster_assignments"]

    rows: list[OutlierPodGap] = []
    for pod in out_ids:
        g = gaps_map.get(pod, {})
        m = meta_map.get(pod, {})
        rows.append(OutlierPodGap(
            pod=pod,
            cluster=int(assignments.get(pod, 0)),
            ateco_l1=m.get("ateco_l1"),
            ateco_l2=m.get("ateco_l2"),
            days_with_data=int(g.get("days_with_data", 0)),
            expected_days=int(g.get("expected_days", 0)),
            missing_days=int(g.get("missing_days", 0)),
            missing_pct=float(g.get("missing_pct", 0.0)),
        ))

    return OutlierResponse(
        n_pods=len(normalised),
        n_outliers=result["n_outliers"],
        threshold=req.threshold,
        cluster_sizes=result["cluster_sizes"],
        outlier_clusters=result["outlier_clusters"],
        normal_clusters=result["normal_clusters"],
        outlier_pods=rows,
        outlier_centroids=result.get("outlier_centroids", {}),
        outlier_stds=result.get("outlier_stds", {}),
    )
