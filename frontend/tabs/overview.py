"""Overview tab — population-level statistics about the POD dataset."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api
from charts import (DARK_NAVY, GRID_LIGHT, LIGHT_NAVY, LIGHT_TEXT,
                    PALETTE, cluster_palette, fmt_int)
from helpers import ateco_desc


# ─────────────────────────────────────────────────────────────────────────────
# Geographic bubble map
# ─────────────────────────────────────────────────────────────────────────────
def _geography_map(payload: dict) -> go.Figure | None:
    located = payload.get("located") or []
    if not located:
        return None

    lats   = [c["lat"]    for c in located]
    lons   = [c["lon"]    for c in located]
    counts = [c["n_pods"] for c in located]
    names  = [c["comune"] for c in located]

    max_n  = max(counts) if counts else 1
    sizes  = [12 + 50 * math.sqrt(n / max_n) for n in counts]

    view   = payload.get("map_view") or {}
    center = dict(lat=view.get("center_lat", 42.5),
                   lon=view.get("center_lon", 12.5))
    zoom   = float(view.get("zoom", 5.5))

    hover  = []
    for c in located:
        line2  = ", ".join(x for x in (c.get("provincia"), c.get("regione")) if x)
        suffix = f"<br>{line2}" if line2 else ""
        hover.append(f"<b>{c['comune']}</b>{suffix}<br>"
                      f"{fmt_int(c['n_pods'])} PODs")

    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons,
        mode="markers+text",
        marker=dict(size=sizes, color="#4a9eff",
                    sizemode="diameter", opacity=0.85),
        text=[fmt_int(n) for n in counts],
        textposition="top center",
        textfont=dict(color=LIGHT_TEXT, size=12),
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=center, zoom=zoom),
        paper_bgcolor=DARK_NAVY, plot_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT),
        height=480, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Power-class bar chart
# ─────────────────────────────────────────────────────────────────────────────
def _power_class_chart(by_class: list[dict]) -> go.Figure:
    labels = [c["label"]  for c in by_class]
    counts = [c["n_pods"] for c in by_class]
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color=cluster_palette(len(labels)),
        text=[fmt_int(n) for n in counts], textposition="outside",
        textfont=dict(color=LIGHT_TEXT, size=11),
    ))
    fig.update_layout(
        title=dict(text="POD Composition by Contractual Power Class [-]",
                   font=dict(color=LIGHT_TEXT, size=14), x=0.02),
        xaxis=dict(tickfont=dict(color=LIGHT_TEXT),
                   title="Power Class", title_font=dict(color=LIGHT_TEXT)),
        yaxis=dict(tickfont=dict(color=LIGHT_TEXT),
                   title="Number of PODs [-]",
                   title_font=dict(color=LIGHT_TEXT),
                   gridcolor=LIGHT_NAVY),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT),
        height=380, margin=dict(t=50, b=70, l=70, r=20),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Consumption histograms
# ─────────────────────────────────────────────────────────────────────────────
def _consumption_histogram(values: list[float], unit: str,
                            title: str) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=40,
        marker=dict(color="#4a9eff", line=dict(color=LIGHT_TEXT, width=0.4)),
    ))
    fig.update_layout(
        title=dict(text=f"{title} [{unit}]",
                   font=dict(color=LIGHT_TEXT, size=14), x=0.02),
        xaxis=dict(tickfont=dict(color=LIGHT_TEXT),
                   title=f"Consumption [{unit}]",
                   title_font=dict(color=LIGHT_TEXT)),
        yaxis=dict(tickfont=dict(color=LIGHT_TEXT),
                   title="Number of PODs [-]",
                   title_font=dict(color=LIGHT_TEXT),
                   gridcolor=LIGHT_NAVY),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT),
        height=340, margin=dict(t=50, b=70, l=70, r=20),
        bargap=0.05, showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ATECO catalogue-coverage donut
# ─────────────────────────────────────────────────────────────────────────────
def _ateco_coverage_chart(info: dict) -> go.Figure:
    name_plural    = info.get("name_plural", "Codes")
    present_in_cat = int(info.get("n_codes_present_in_catalogue", 0))
    catalogue_size = int(info.get("n_codes_in_catalogue", 0))
    missing        = max(catalogue_size - present_in_cat, 0)
    coverage_pct   = (present_in_cat / catalogue_size * 100.0
                       if catalogue_size > 0 else 0.0)

    fig = go.Figure(data=[go.Pie(
        labels=[f"Catalogue {name_plural} Present in Dataset "
                f"({fmt_int(present_in_cat)})",
                f"Catalogue {name_plural} Missing from Dataset "
                f"({fmt_int(missing)})"],
        values=[present_in_cat, missing],
        marker=dict(colors=["#2e7d32", "#c62828"]),
        hole=0.55, textinfo="percent",
        textfont=dict(color="white", size=13),
        sort=False,
    )])
    title = (f"ATECO {info.get('name', '?')} — Catalogue Coverage [-]"
              f"  (catalog level {info.get('catalogue_level', '?')})")
    fig.update_layout(
        title=dict(text=title,
                   font=dict(color=LIGHT_TEXT, size=13), x=0.5, xanchor="center"),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(color=LIGHT_TEXT),
        height=320, margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                    x=0.5, xanchor="center", font=dict(color=LIGHT_TEXT)),
        annotations=[dict(
            text=f"{coverage_pct:.1f}%",
            font=dict(size=18, color=LIGHT_TEXT), showarrow=False, x=0.5, y=0.5,
        )],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────
def render():
    st.subheader("Overview")
    st.caption(
        "Population-level overview of the POD dataset, summarising its "
        "geographic footprint, contractual-power composition, consumption "
        "characteristics, and ATECO classification coverage."
    )

    # ── Geographic distribution ───────────────────────────────────────────
    # Calling the endpoint also triggers a background geocoding pass on
    # the backend for any Comuni not yet in `comuni_centroids`. The user
    # only needs to refresh the page to see the map fill in.
    st.markdown("#### Geographic Distribution")
    try:
        # Cache disabled here so background geocoding progress is reflected
        # on each refresh; the call is cheap (a single JOIN query).
        geo = api.overview_geography()
    except api.BackendError as e:
        st.error(f"Geography fetch failed: {e}.")
    else:
        n_loc = int(geo.get("n_comuni", 0))
        n_un  = len(geo.get("unlocated") or [])
        fig_map = _geography_map(geo)

        if n_loc == 0 and n_un > 0:
            st.info(
                f"⏳ Geocoding {n_un} Comuni in the background — refresh "
                "the page in a few seconds to see the map populate."
            )
        elif fig_map is not None:
            c_map, c_side = st.columns([3, 1])
            with c_map:
                st.plotly_chart(fig_map, use_container_width=True)
                if n_un > 0:
                    st.caption(
                        f"⏳ {n_un} Comuni still being geocoded — they will "
                        "appear on the next refresh."
                    )
            with c_side:
                st.caption(f"**Scope:** {geo.get('scope_label', '—')}")
                st.metric("Comuni Mapped:",  fmt_int(n_loc))
                st.metric("PODs Mapped:",    fmt_int(geo.get("n_pods_geo", 0)))
                if geo.get("n_pods_no_geo"):
                    st.metric("PODs Pending Geocoding:",
                              fmt_int(geo["n_pods_no_geo"]))
                if geo.get("located"):
                    df_loc = pd.DataFrame([
                        {"Comune":    c["comune"],
                         "Provincia": c.get("provincia") or "—",
                         "PODs":      c["n_pods"]}
                        for c in geo["located"]
                    ])
                    st.dataframe(df_loc, hide_index=True,
                                  use_container_width=True, height=170)

    # ── Power class composition ───────────────────────────────────────────
    st.markdown("#### Composition by Contractual Power Class")
    try:
        pc = api.overview_power_class_distribution()
    except api.BackendError as e:
        st.error(f"Power-class fetch failed: {e}.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.plotly_chart(_power_class_chart(pc["by_class"]),
                            use_container_width=True)
        with c2:
            n_with    = int(pc.get("n_pods_with_meas",    pc.get("n_pods", 0)))
            n_without = int(pc.get("n_pods_without_meas", pc.get("n_unknown", 0)))
            st.metric("PODs with a Power Measurement:", fmt_int(n_with))
            st.caption("Most recent contractual-power measurement per POD.")
            st.metric("PODs Without a Power Measurement:", fmt_int(n_without))

    # ── Consumption histogram ─────────────────────────────────────────────
    st.markdown("#### Consumption Distribution")
    try:
        cd = api.overview_consumption_distribution(tipologia="AP")
    except api.BackendError as e:
        st.error(f"Consumption fetch failed: {e}.")
    else:
        if not cd.get("monthly_kwh"):
            st.info("No consumption data available for the selected tipologia.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    _consumption_histogram(cd["monthly_kwh"], "kWh",
                                            "Monthly Average per POD"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    _consumption_histogram(cd["annual_kwh"], "kWh",
                                            "Annual Average per POD"),
                    use_container_width=True,
                )
            stats = cd.get("stats", {})
            if stats:
                cM, cMed, cP10, cP90 = st.columns(4)
                cM.metric("Daily Mean [kWh]:",     f"{stats.get('daily_mean', 0):.2f}")
                cMed.metric("Daily Median [kWh]:", f"{stats.get('daily_median', 0):.2f}")
                cP10.metric("P10 [kWh]:",          f"{stats.get('daily_p10', 0):.2f}")
                cP90.metric("P90 [kWh]:",          f"{stats.get('daily_p90', 0):.2f}")

    # ── ATECO coverage per semantic level ─────────────────────────────────
    st.markdown("#### ATECO Catalogue Coverage")
    st.caption(
        "Coverage is reported at the three semantic levels stored in the "
        "POD metadata — **Division** (catalog level 2), **Class** "
        "(catalog level 4), and **Subcategory** (catalog level 6) — "
        "not at the literal catalog levels 1/2/3 (which would correspond "
        "to Section, Division, and Group)."
    )
    try:
        cov   = api.overview_ateco_coverage()
        descs = api.ateco_descriptions()
    except api.BackendError as e:
        st.error(f"ATECO coverage fetch failed: {e}.")
        return

    cols = st.columns(3)
    for col, lvl in zip(cols, [1, 2, 3]):
        info = cov.get(str(lvl), cov.get(lvl))
        if not info:
            continue
        name = info.get("name", f"Level {lvl}")
        with col:
            st.plotly_chart(_ateco_coverage_chart(info),
                            use_container_width=True)

            n_missing = int(info.get("n_missing", 0))
            with st.expander(f"Missing {name} Codes ({fmt_int(n_missing)})"):
                if info.get("missing_codes"):
                    rows = [{"Code": c, "Description": ateco_desc(c, descs)}
                            for c in info["missing_codes"]]
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                  use_container_width=True, height=240)
                else:
                    st.success(f"All catalog {name.lower()} codes are "
                                "present in the dataset.")

            n_out = int(info.get("n_out_of_catalogue", 0))
            if n_out:
                with st.expander(
                    f"Dataset {name} Codes Outside the Catalogue "
                    f"({fmt_int(n_out)})"
                ):
                    rows = [{"Code": c, "Description": ateco_desc(c, descs)}
                            for c in info.get("out_of_catalogue_codes", [])]
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                  use_container_width=True, height=240)
                    st.caption(
                        "These codes appear in the POD metadata but are not "
                        "part of the official ATECO catalog at this "
                        "semantic level (e.g. PoliTo tags such as DO, CO, "
                        "IL). They are excluded from the coverage "
                        "percentage above."
                    )
