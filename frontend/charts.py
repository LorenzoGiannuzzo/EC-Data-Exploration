"""Plotly chart helpers used by the tabs.

Style choices match the original dashboard:
    - dark navy backgrounds in the clustering tab
    - white backgrounds for GSE/ARERA so they print cleanly in reports
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

DARK_NAVY  = "#0d1f3c"
LIGHT_NAVY = "#1a3a6b"
ACCENT     = "#2e7d32"
LIGHT_TEXT = "#e8f4fd"
GRID_LIGHT = "#d0dff0"
PALETTE    = ["#4a9eff", "#2e7d32", "#e65100", "#9c27b0", "#00838f",
              "#f9a825", "#5d4037", "#c62828", "#1565c0", "#558b2f",
              "#6a1b9a", "#00695c", "#bf360c", "#283593", "#ad1457"]


# ── Cluster color palette (sampled from Pearson RdBu_r) ──────────────────────
def cluster_palette(k: int) -> list[str]:
    """Return k discrete colors sampled from the same RdBu_r diverging scale
    used by the Pearson heat-map, so the cluster charts share a visual
    language with the correlation panel.

    Sampling avoids the white midpoint (≈0.5) — which would be invisible on
    the dark navy background — by walking the scale from ``edge_pad`` to
    ``1 − edge_pad`` and skipping a window around 0.5.
    """
    import plotly.colors as pc
    if k <= 0:
        return []
    if k == 1:
        return ["#d6604d"]              # single warm-red default
    edge_pad   = 0.05
    inner_skip = 0.18                   # half-width of the dropped midband
    half = k // 2
    lefts  = [edge_pad + (0.5 - inner_skip - edge_pad) * i / max(half - 1, 1)
              for i in range(half)]
    rights_n = k - half
    rights = [0.5 + inner_skip
              + (1.0 - edge_pad - 0.5 - inner_skip) * i / max(rights_n - 1, 1)
              for i in range(rights_n)]
    positions = lefts + rights
    return pc.sample_colorscale("RdBu_r", positions, colortype="rgb")


# ── Number formatting (Lorenzo's preference: thousand separator = space) ─────
def fmt_int(n: int | float) -> str:
    """`12638` → `"12 638"`. Used everywhere POD counts are shown."""
    try:
        n = int(n)
    except Exception:
        return str(n)
    return f"{n:,}".replace(",", " ")


# ── Axis settings ─────────────────────────────────────────────────────────────
def _axis_dark(title: str = "") -> dict:
    return dict(
        title=title,
        title_font=dict(color=LIGHT_TEXT),
        tickfont=dict(color=LIGHT_TEXT),
        gridcolor=LIGHT_NAVY, showgrid=True, zeroline=False,
    )


def _axis_light(title: str = "") -> dict:
    return dict(
        title=title,
        title_font=dict(color=DARK_NAVY),
        tickfont=dict(color=LIGHT_NAVY),
        gridcolor=GRID_LIGHT, showgrid=True, zeroline=False,
    )


# ── Cluster centroids ─────────────────────────────────────────────────────────
def centroid_chart(
    centroids: dict[int, list[float]],
    n_pods_per_cluster: dict[int, int],
    title: str = "Cluster Centroids — Normalized Daily Profile",
    y_label: str = "Normalized Load [-]",
) -> go.Figure:
    x_labels = [f"{(i // 4):02d}:{(i % 4) * 15:02d}" for i in range(96)]
    fig = go.Figure()
    items = sorted(centroids.items())
    colors = cluster_palette(len(items))
    for i, (cl, profile) in enumerate(items):
        n = n_pods_per_cluster.get(cl, 0)
        fig.add_trace(go.Scatter(
            x=x_labels, y=profile, mode="lines",
            name=f"Cluster {cl} (n={fmt_int(n)})",
            line=dict(color=colors[i], width=2),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=LIGHT_TEXT, size=15), x=0.02),
        xaxis={**_axis_dark("Time of Day [hh:mm]"), "tickangle": -45, "dtick": 8},
        yaxis=_axis_dark(y_label),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT),
        legend=dict(font=dict(color=LIGHT_TEXT, size=10)),
        height=420, margin=dict(t=50, b=70, l=70, r=20),
    )
    return fig


# ── Cluster sizes ─────────────────────────────────────────────────────────────
def cluster_sizes_chart(sizes: dict[int, int]) -> go.Figure:
    items = sorted(sizes.items())
    colors = cluster_palette(len(items))
    fig = go.Figure(go.Bar(
        x=[f"Cluster {c}" for c, _ in items],
        y=[n for _, n in items],
        marker_color=colors,
        text=[fmt_int(n) for _, n in items], textposition="outside",
        textfont=dict(color=LIGHT_TEXT),
    ))
    fig.update_layout(
        title=dict(text="Cluster Sizes", font=dict(color=LIGHT_TEXT, size=15), x=0.02),
        xaxis=_axis_dark(), yaxis=_axis_dark("Number of PODs [-]"),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT),
        height=320, margin=dict(t=50, b=50, l=70, r=20),
        showlegend=False,
    )
    return fig


# ── Pearson correlation heat-map between centroids ───────────────────────────
def pearson_heatmap(
    matrix: list[list[float]], labels: list[int],
) -> go.Figure:
    arr = np.asarray(matrix, dtype=float)
    txt = [[f"{v:.2f}" for v in row] for row in arr]
    fig = go.Figure(go.Heatmap(
        z=arr,
        x=[f"Cluster {l}" for l in labels],
        y=[f"Cluster {l}" for l in labels],
        text=txt, texttemplate="%{text}",
        colorscale="RdBu", zmin=-1, zmax=1, reversescale=True,
        colorbar=dict(title="r", tickfont=dict(color=LIGHT_TEXT),
                      title_font=dict(color=LIGHT_TEXT)),
    ))
    fig.update_layout(
        title=dict(text="Pearson Correlation between Cluster Centroids [-]",
                   font=dict(color=LIGHT_TEXT, size=14), x=0.02),
        xaxis=dict(tickfont=dict(color=LIGHT_TEXT)),
        yaxis=dict(tickfont=dict(color=LIGHT_TEXT)),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(color=LIGHT_TEXT),
        height=420, margin=dict(t=50, b=50, l=70, r=20),
    )
    return fig


# ── GSE comparison: monthly grid of hourly profiles ──────────────────────────
def gse_monthly_chart(
    our_profile: dict[int, list[float]],
    ref_profile: dict[int, list[float]],
    profile_code: str,
) -> go.Figure:
    from plotly.subplots import make_subplots
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = make_subplots(
        rows=3, cols=4, shared_xaxes=True,
        subplot_titles=month_names,
        horizontal_spacing=0.04, vertical_spacing=0.16,
    )
    # Build a complete coverage view: which (month → which curves are present)
    # so we can annotate panels where the PoliTo curve is genuinely missing
    # (zero-row filter dropped every day → no real data).
    our_months = set(our_profile)
    ref_months = set(ref_profile)
    for m in range(1, 13):
        row = (m - 1) // 4 + 1
        col = (m - 1) % 4 + 1
        x = list(range(24))
        if m in our_months:
            fig.add_trace(go.Scatter(
                x=x, y=our_profile[m], mode="lines",
                name="PoliTo Dataset", legendgroup="ours",
                showlegend=(m == min(our_months)),
                line=dict(color="#4fa3ff", width=2.2),
            ), row=row, col=col)
        if m in ref_months:
            fig.add_trace(go.Scatter(
                x=x, y=ref_profile[m], mode="lines",
                name=f"GSE {profile_code}", legendgroup="ref",
                showlegend=(m == min(ref_months)),
                line=dict(color="#ff9f43", width=2.2, dash="dot"),
            ), row=row, col=col)
        # PoliTo curve absent for this month → tell the user why instead of
        # leaving the panel mysteriously half-empty.
        if m not in our_months and m in ref_months:
            fig.add_annotation(
                xref=f"x{m} domain" if m > 1 else "x domain",
                yref=f"y{m} domain" if m > 1 else "y domain",
                x=0.5, y=0.5, xanchor="center", yanchor="middle",
                text="PoliTo: no valid data",
                showarrow=False,
                font=dict(color="#4fa3ff", size=11, family="Arial"),
                bgcolor="rgba(13, 31, 60, 0.55)", borderpad=4,
                row=row, col=col,
            )

    fig.update_layout(
        title=dict(
            text=f"Monthly Hourly Profile — PoliTo vs GSE {profile_code} "
                 f"[% of Monthly Use]",
            font=dict(color=LIGHT_TEXT, size=14), x=0.02,
        ),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT, size=11),
        height=640, margin=dict(t=55, b=50, l=10, r=10),
        legend=dict(orientation="h", yanchor="top", y=-0.07,
                    x=0.5, xanchor="center",
                    font=dict(size=11, color=LIGHT_TEXT)),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color=LIGHT_TEXT, size=12)
        ann.yshift = 8
    # Soften gridlines (same recipe as Load Profiler) and only show axis
    # titles on the outer edges of the grid.
    fig.update_xaxes(
        gridcolor="rgba(208, 223, 240, 0.15)",
        color=LIGHT_TEXT, title_text="",
        showline=False, zeroline=False,
        tickmode="array", tickvals=[0, 6, 12, 18, 23],
    )
    fig.update_yaxes(
        gridcolor="rgba(208, 223, 240, 0.15)",
        color=LIGHT_TEXT, title_text="",
        showline=False, zeroline=False,
    )
    for c in range(1, 5):
        fig.update_xaxes(title_text="Hour of day [h]",
                         title_font=dict(size=10), row=3, col=c)
    for r in range(1, 4):
        fig.update_yaxes(title_text="Share [%]",
                         title_font=dict(size=10), row=r, col=1)
    return fig


# ── ARERA single-panel comparison: two lines ─────────────────────────────────
def arera_single_chart(
    our: list[float], ref: list[float], title_suffix: str = "",
    our_label: str = "PoliTo Dataset", ref_label: str = "ARERA Reference",
) -> go.Figure:
    """24-hour line chart with two lines (PoliTo + ARERA). Tick every 3 hours
    so the labels fit when the chart is narrow (3-column layout).

    The PoliTo line is suppressed (and replaced by a small "no valid data"
    annotation) when the supplied series is empty or effectively all-zero —
    the symptom of a day-type for which no POD in the selection has any
    real measurement in the chosen window.
    """
    x_labels = [f"{h:02d}:00" for h in range(24)]
    fig = go.Figure()

    our_arr = [v for v in (our or [])]
    has_our = bool(our_arr) and any(
        (v is not None and not (isinstance(v, float) and v != v) and v > 1e-9)
        for v in our_arr
    )

    if has_our:
        fig.add_trace(go.Scatter(
            x=x_labels, y=our_arr, mode="lines+markers", name=our_label,
            line=dict(color="#4fa3ff", width=2.5),
            marker=dict(color="#4fa3ff", size=6),
        ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=ref, mode="lines+markers", name=ref_label,
        line=dict(color="#ff9f43", width=2.5, dash="dash"),
        marker=dict(color="#ff9f43", size=6, symbol="diamond"),
    ))
    if not has_our:
        fig.add_annotation(
            xref="x domain", yref="y domain",
            x=0.5, y=0.55, xanchor="center", yanchor="middle",
            text="PoliTo: no valid data",
            showarrow=False,
            font=dict(color="#4fa3ff", size=12, family="Arial"),
            bgcolor="rgba(13, 31, 60, 0.55)", borderpad=4,
        )

    title_text = (f"Hourly Energy Profile{title_suffix}"
                  if title_suffix else "Hourly Energy Profile")
    fig.update_layout(
        title=dict(text=title_text,
                   font=dict(color=LIGHT_TEXT, size=13), x=0.02),
        xaxis=dict(title="Hour of day [h]",
                   title_font=dict(size=10, color=LIGHT_TEXT),
                   tickfont=dict(color=LIGHT_TEXT),
                   gridcolor="rgba(208, 223, 240, 0.15)",
                   showline=False, zeroline=False,
                   tickangle=-45,
                   tickmode="array",
                   tickvals=[f"{h:02d}:00" for h in range(0, 24, 3)]),
        yaxis=dict(title="Hourly consumption [kWh]",
                   title_font=dict(size=10, color=LIGHT_TEXT),
                   tickfont=dict(color=LIGHT_TEXT),
                   gridcolor="rgba(208, 223, 240, 0.15)",
                   showline=False, zeroline=False),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    x=0.5, xanchor="center",
                    font=dict(color=LIGHT_TEXT, size=11),
                    bgcolor="rgba(0,0,0,0)"),
        height=380, margin=dict(t=60, b=60, l=70, r=20),
    )
    return fig


# ── ARERA monthly grid: 12-panel breakdown for a single day-type ────────────
def arera_monthly_grid_chart(
    our_by_month:  dict[int, list[float]],
    ref_by_month:  dict[int, list[float]],
    day_type:      str,
) -> "go.Figure":
    """Render a 3×4 grid of mini PoliTo-vs-ARERA panels, one per calendar
    month, in the same dark-navy style as ``gse_monthly_chart``.

    `our_by_month` and `ref_by_month` are keyed by ``month_idx`` (1..12).
    A panel whose PoliTo series is empty or effectively all-zero gets a
    centered "PoliTo: no valid data" annotation instead of a flat zero line.
    """
    from plotly.subplots import make_subplots
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = make_subplots(
        rows=3, cols=4, shared_xaxes=True,
        subplot_titles=month_names,
        horizontal_spacing=0.04, vertical_spacing=0.16,
    )
    legend_shown_our, legend_shown_ref = False, False
    for m in range(1, 13):
        row = (m - 1) // 4 + 1
        col = (m - 1) % 4 + 1
        x = list(range(24))
        our_arr = our_by_month.get(m) or []
        ref_arr = ref_by_month.get(m) or []
        has_our = bool(our_arr) and any(
            (v is not None and not (isinstance(v, float) and v != v) and v > 1e-9)
            for v in our_arr
        )
        has_ref = bool(ref_arr) and any(
            (v is not None and not (isinstance(v, float) and v != v) and v > 1e-9)
            for v in ref_arr
        )
        if has_our:
            fig.add_trace(go.Scatter(
                x=x, y=our_arr, mode="lines",
                name="PoliTo Dataset", legendgroup="ours",
                showlegend=not legend_shown_our,
                line=dict(color="#4fa3ff", width=2.0),
            ), row=row, col=col)
            legend_shown_our = True
        if has_ref:
            fig.add_trace(go.Scatter(
                x=x, y=ref_arr, mode="lines",
                name="ARERA Reference", legendgroup="ref",
                showlegend=not legend_shown_ref,
                line=dict(color="#ff9f43", width=2.0, dash="dash"),
            ), row=row, col=col)
            legend_shown_ref = True
        if not has_our and has_ref:
            fig.add_annotation(
                xref=f"x{m} domain" if m > 1 else "x domain",
                yref=f"y{m} domain" if m > 1 else "y domain",
                x=0.5, y=0.5, xanchor="center", yanchor="middle",
                text="PoliTo: no valid data",
                showarrow=False,
                font=dict(color="#4fa3ff", size=10, family="Arial"),
                bgcolor="rgba(13, 31, 60, 0.55)", borderpad=3,
                row=row, col=col,
            )

    fig.update_layout(
        title=dict(
            text=f"Monthly Hourly Profile — PoliTo vs ARERA — {day_type} [kWh]",
            font=dict(color=LIGHT_TEXT, size=14), x=0.02,
        ),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT, size=11),
        height=640, margin=dict(t=55, b=50, l=10, r=10),
        legend=dict(orientation="h", yanchor="top", y=-0.07,
                    x=0.5, xanchor="center",
                    font=dict(size=11, color=LIGHT_TEXT)),
    )
    for ann in fig.layout.annotations:
        if ann.text in month_names:
            ann.font = dict(color=LIGHT_TEXT, size=12)
            ann.yshift = 8
    fig.update_xaxes(
        gridcolor="rgba(208, 223, 240, 0.15)",
        color=LIGHT_TEXT, title_text="",
        showline=False, zeroline=False,
        tickmode="array", tickvals=[0, 6, 12, 18, 23],
    )
    fig.update_yaxes(
        gridcolor="rgba(208, 223, 240, 0.15)",
        color=LIGHT_TEXT, title_text="",
        showline=False, zeroline=False,
    )
    for c in range(1, 5):
        fig.update_xaxes(title_text="Hour of day [h]",
                         title_font=dict(size=10), row=3, col=c)
    for r in range(1, 4):
        fig.update_yaxes(title_text="kWh",
                         title_font=dict(size=10), row=r, col=1)
    return fig
