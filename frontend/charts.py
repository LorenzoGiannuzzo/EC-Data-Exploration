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
        rows=3, cols=4,
        subplot_titles=month_names,
        horizontal_spacing=0.06, vertical_spacing=0.13,
    )
    months_in_data = sorted(set(our_profile) | set(ref_profile))
    for m in months_in_data:
        if not 1 <= m <= 12:
            continue
        row = (m - 1) // 4 + 1
        col = (m - 1) % 4 + 1
        x = list(range(24))
        if m in our_profile:
            fig.add_trace(go.Scatter(
                x=x, y=our_profile[m], mode="lines",
                name="PoliTo Dataset", legendgroup="ours",
                showlegend=(m == months_in_data[0]),
                line=dict(color="#1565c0", width=2.2),
            ), row=row, col=col)
        if m in ref_profile:
            fig.add_trace(go.Scatter(
                x=x, y=ref_profile[m], mode="lines",
                name=f"GSE {profile_code}", legendgroup="ref",
                showlegend=(m == months_in_data[0]),
                line=dict(color="#e65100", width=2.2, dash="dot"),
            ), row=row, col=col)

    fig.update_layout(
        title=dict(
            text=f"Monthly Hourly Profile — PoliTo vs GSE {profile_code} "
                 f"[% of Monthly Use]",
            font=dict(color=DARK_NAVY, size=14), x=0.02,
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", color=DARK_NAVY, size=11),
        height=900, margin=dict(t=80, b=60, l=60, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.06,
                    x=0.5, xanchor="center",
                    font=dict(size=12, color=DARK_NAVY),
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color=DARK_NAVY, size=12)
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID_LIGHT,
        tickfont=dict(size=9, color=LIGHT_NAVY), title="Hour of Day [h]",
        title_font=dict(size=9, color=LIGHT_NAVY),
        tickmode="array", tickvals=[0, 6, 12, 18, 23],
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID_LIGHT,
        tickfont=dict(size=9, color=LIGHT_NAVY),
        title="Share [%]", title_font=dict(size=9, color=LIGHT_NAVY),
    )
    return fig


# ── ARERA single-panel comparison: two lines ─────────────────────────────────
def arera_single_chart(
    our: list[float], ref: list[float], title_suffix: str = "",
    our_label: str = "PoliTo Dataset", ref_label: str = "ARERA Reference",
) -> go.Figure:
    """24-hour line chart with two lines (PoliTo + ARERA). Tick every 3 hours
    so the labels fit when the chart is narrow (3-column layout)."""
    x_labels = [f"{h:02d}:00" for h in range(24)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_labels, y=our, mode="lines+markers", name=our_label,
        line=dict(color="#1565c0", width=2.5),
        marker=dict(color="#1565c0", size=6),
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=ref, mode="lines+markers", name=ref_label,
        line=dict(color="#e65100", width=2.5, dash="dash"),
        marker=dict(color="#e65100", size=6, symbol="diamond"),
    ))
    # Title kept short — verbose context goes in the panel header above.
    title_text = (f"Hourly Energy Profile{title_suffix}"
                  if title_suffix else "Hourly Energy Profile")
    fig.update_layout(
        title=dict(text=title_text,
                   font=dict(color=DARK_NAVY, size=13), x=0.02),
        xaxis={**_axis_light("Hour of Day [h]"),
               "tickangle": -45,
               "tickmode": "array",
               "tickvals": [f"{h:02d}:00" for h in range(0, 24, 3)]},
        yaxis=_axis_light("Hourly Consumption [kWh]"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", color=DARK_NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    x=0.5, xanchor="center",
                    font=dict(color=DARK_NAVY, size=11),
                    bgcolor="rgba(255,255,255,0.8)"),
        height=420, margin=dict(t=70, b=70, l=70, r=20),
    )
    return fig
