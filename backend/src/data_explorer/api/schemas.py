"""Pydantic request/response schemas for the public API.

Convention:
    *Request  — body validated by FastAPI
    *Response — what the endpoint returns; serialisable to JSON
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ── Common: POD filter (used by clustering, GSE, ARERA) ───────────────────────
class PodFilter(BaseModel):
    """Compose a POD set by filtering on ATECO codes + data coverage."""

    ateco_l1:   list[str] | None = Field(default=None, description="L1 codes (e.g. ['DO','47']).")
    ateco_l2:   list[str] | None = Field(default=None, description="L2 codes (e.g. ['DO.R','47.11']).")
    ateco_l3:   list[str] | None = Field(default=None, description="L3 codes.")
    min_months: int              = Field(default=12, ge=0, le=120,
                                          description="Minimum months of data per POD.")
    tipologia:  str              = Field(default="AP",
                                          description="Measurement type (AP, AN, RCN, …).")
    power_ranges: list[list[float | None]] | None = Field(
        default=None,
        description="Contractual-power ranges [min_kw, max_kw] (kW, max=None "
                    "= open bound). A POD matches when min < d_pot1 <= max "
                    "for at least one range. None/empty = no power filter.")
    include_missing_power: bool  = Field(
        default=False,
        description="When power_ranges is set, also keep PODs with no "
                    "contractual-power value (d_pot1 IS NULL).")


# ── /metadata ─────────────────────────────────────────────────────────────────
class AtecoCodeList(BaseModel):
    level: int
    codes: list[str]


class AtecoDescriptions(BaseModel):
    descriptions: dict[str, str]


class TipologieList(BaseModel):
    tipologie: list[str]


class PodSetSummary(BaseModel):
    n_pods:           int
    after_ateco:      int | None = None
    after_coverage:   int
    after_power:      int | None = None
    sample_pods:      list[str]  = []


# ── /clustering ───────────────────────────────────────────────────────────────
class ClusteringRequest(BaseModel):
    filter:     PodFilter        = Field(default_factory=PodFilter)
    n_clusters: int              = Field(ge=2, le=20)
    month:      int              = Field(default=0, ge=0, le=12,
                                          description="0 = annual average profile; 1..12 = single month.")
    auto_k:     bool             = Field(default=False,
                                          description="Ignore n_clusters and pick k via the legacy "
                                                      "multi-metric vote (silhouette, CH, DB, elbow).")
    method:     str              = Field(default="ward",
                                          description="Linkage method: ward, average, complete, single.")
    normalise:  str              = Field(default="max", description="'max' or 'sum'.")
    top_ateco_per_cluster: int   = Field(default=3, ge=0, le=20)


class AtecoBreakdownEntry(BaseModel):
    cluster:        int
    ateco:          str | None
    description:    str | None = None
    n_pods:         int
    pct_of_cluster: float


class ClusterMetrics(BaseModel):
    n_pods:            int
    n_clusters:        int
    silhouette:        float | None
    calinski_harabasz: float | None = None
    davies_bouldin:    float | None = None
    sizes:             dict[int, int]


class ClusteringResponse(BaseModel):
    n_pods:           int
    metrics:          ClusterMetrics
    centroids:        dict[int, list[float]]
    assignments:      dict[str, int]
    ateco_breakdown:  list[AtecoBreakdownEntry]
    pearson_matrix:   list[list[float]] | None = None
    pearson_labels:   list[int]                = []
    auto_k_details:   dict | None              = None


# ── /clustering/run-by-level — separate clustering per selected ATECO level ──
class ClusteringByLevelRequest(BaseModel):
    """Run a separate clustering for each ATECO level that has a non-empty
    selection. Reproduces the legacy dashboard behaviour where Level 1, Level 2
    and Level 3 each produce their own centroid panel."""
    ateco_l1:   list[str] | None = None
    ateco_l2:   list[str] | None = None
    ateco_l3:   list[str] | None = None
    min_months: int              = 12
    tipologia:  str              = "AP"
    n_clusters: int              = Field(ge=2, le=20)
    month:      int              = Field(default=0, ge=0, le=12,
                                          description="0 = annual average profile; 1..12 = single month.")
    auto_k:     bool             = False
    method:     str              = "average"
    normalise:  str              = "minmax"
    top_ateco_per_cluster: int   = 5
    power_ranges: list[list[float | None]] | None = None
    include_missing_power: bool  = False


class ClusteringLevelResult(BaseModel):
    level:           int                       # 1, 2 or 3
    ateco_filter:    list[str]                 # codes that produced this clustering
    response:        ClusteringResponse


class ClusteringByLevelResponse(BaseModel):
    results: list[ClusteringLevelResult]


# ── /clustering/outliers ─────────────────────────────────────────────────────
class OutlierRequest(BaseModel):
    filter:     PodFilter = Field(default_factory=PodFilter)
    n_clusters: int       = Field(default=10, ge=2, le=50,
                                   description="Number of single-linkage clusters.")
    threshold:  int       = Field(default=5, ge=1, le=200,
                                   description="Clusters with fewer PODs are outliers.")
    normalise:  str       = "minmax"


class OutlierPodGap(BaseModel):
    pod:             str
    cluster:         int
    ateco_l1:        str | None = None
    ateco_l2:        str | None = None
    days_with_data:  int
    expected_days:   int
    missing_days:    int
    missing_pct:     float


class OutlierResponse(BaseModel):
    n_pods:            int
    n_outliers:        int
    threshold:         int
    cluster_sizes:     dict[int, int]
    outlier_clusters:  list[int]
    normal_clusters:   list[int]
    outlier_pods:      list[OutlierPodGap]
    outlier_centroids: dict[int, list[float]] = {}
    outlier_stds:      dict[int, list[float]] = {}


# ── /gse ──────────────────────────────────────────────────────────────────────
class GseProfileList(BaseModel):
    profile_codes: list[str]


class GseCompareRequest(BaseModel):
    filter:       PodFilter = Field(default_factory=PodFilter)
    profile_code: str       = Field(description="GSE profile code (PDMM, PDMF, …).")
    dayset:       str       = Field(default="weekday",
                                     description="'weekday', 'weekend' or 'all'.")


class MonthlyMetrics(BaseModel):
    month:       int                         # 1..12
    rmse:        float
    mae:         float
    bias:        float
    max_abs_err: float


class GseCompareResponse(BaseModel):
    n_pods:            int
    profile_code:      str
    our_profile:       dict[int, list[float]]   # month → 24 floats (%)
    reference_profile: dict[int, list[float]]   # month → 24 floats (%)
    metrics:           list[MonthlyMetrics]


# ── /arera ────────────────────────────────────────────────────────────────────
class AreraPowerClass(BaseModel):
    code:  str
    label: str


class AreraPowerClassList(BaseModel):
    power_classes: list[str]


class AreraKey(BaseModel):
    power_class: str
    market:      str
    residenza:   str
    province:    str
    day_type:    str


class AreraKeyList(BaseModel):
    keys: list[AreraKey]


class AreraCompareRequest(BaseModel):
    filter:      PodFilter = Field(default_factory=PodFilter)
    power_class: str
    market:      str
    residenza:   str
    day_type:    str       = Field(description="'Weekday', 'Saturday' or 'Sunday'.")
    month:       int       = Field(default=0, ge=0, le=12,
                                    description="1..12 = month; 0 = annual average.")
    province:    str       = "Trento"


class AreraCompareResponse(BaseModel):
    n_pods:            int
    our_profile:       list[float]            # 24 floats (kWh)
    reference_profile: list[float]            # 24 floats (kWh)
    metrics:           dict[str, float | None]  # rmse, mae, max_abs_err, bias


# ── /arera all-day-types: same request minus day_type, returns 3 panels ──────
class AreraCompareAllRequest(BaseModel):
    filter:      PodFilter = Field(default_factory=PodFilter)
    power_class: str
    market:      str
    residenza:   str
    month:       int       = Field(default=0, ge=0, le=12)
    province:    str       = "Trento"


class AreraDayPanel(BaseModel):
    day_type:          str
    our_profile:       list[float]
    reference_profile: list[float]
    metrics:           dict[str, float | None]


class AreraCompareAllResponse(BaseModel):
    n_pods:  int
    panels:  list[AreraDayPanel]
