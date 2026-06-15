"""Outliers Detection tab — replicates the legacy dashboard module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api
from charts import (DARK_NAVY, GRID_LIGHT, LIGHT_NAVY, LIGHT_TEXT, fmt_int)
from components import ateco_filter
from components.power_filter import power_filter


# ── Chart: cluster-size overview (red = outlier, green = normal, threshold) ──
def _cluster_size_chart(sizes: dict[int, int], outlier_set: set[int],
                         threshold: int) -> go.Figure:
    items = sorted(sizes.items(), key=lambda kv: int(kv[0]))
    xs    = [f"Cluster {c}" for c, _ in items]
    ys    = [int(n) for _, n in items]
    cols  = ["#e53935" if int(c) in outlier_set else "#2e7d32" for c, _ in items]
    fig = go.Figure(go.Bar(
        x=xs, y=ys, marker_color=cols,
        text=[fmt_int(n) for n in ys], textposition="outside",
        textfont=dict(color=LIGHT_TEXT, size=10),
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#f9a825",
        annotation_text=f"Threshold = {threshold}",
        annotation_font_color="#f9a825",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text="Number of PODs per Cluster (red = outlier, green = normal)",
                   font=dict(color=LIGHT_TEXT, size=14), x=0.02),
        xaxis=dict(tickfont=dict(color=LIGHT_TEXT)),
        yaxis=dict(title="Number of PODs [-]",
                   title_font=dict(color=LIGHT_TEXT),
                   tickfont=dict(color=LIGHT_TEXT),
                   gridcolor=LIGHT_NAVY),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT),
        height=360, margin=dict(t=50, b=50, l=70, r=20),
        showlegend=False,
    )
    return fig


# ── Chart: a single outlier cluster's mean ± std profile ────────────────────
def _outlier_profile_chart(cluster_id: int, n_pods: int,
                           mean_profile: list[float],
                           std_profile:  list[float]) -> go.Figure:
    mean = np.asarray(mean_profile, dtype=float)
    std  = np.asarray(std_profile,  dtype=float)
    x_labels = [f"{(i//4):02d}:{(i%4)*15:02d}" for i in range(len(mean))]
    fig = go.Figure()
    # Shaded ±1 std band
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=list(mean + std) + list((mean - std)[::-1]),
        fill="toself", fillcolor="rgba(229,57,53,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    # Mean line
    fig.add_trace(go.Scatter(
        x=x_labels, y=mean, mode="lines",
        name=f"Cluster {cluster_id} (n={fmt_int(n_pods)})",
        line=dict(color="#e53935", width=2.2),
    ))
    fig.update_layout(
        title=dict(text=f"Outlier — Cluster {cluster_id} (n={fmt_int(n_pods)})",
                   font=dict(size=12, color=LIGHT_TEXT), x=0.02),
        xaxis=dict(tickfont=dict(color=LIGHT_TEXT, size=9),
                   title="Time of Day [hh:mm]",
                   title_font=dict(color=LIGHT_TEXT, size=10),
                   tickmode="array",
                   tickvals=[x_labels[i] for i in range(0, len(x_labels), 12)],
                   gridcolor=LIGHT_NAVY),
        yaxis=dict(tickfont=dict(color=LIGHT_TEXT, size=9),
                   title="Normalized Load [-]",
                   title_font=dict(color=LIGHT_TEXT, size=10),
                   gridcolor=LIGHT_NAVY),
        plot_bgcolor=DARK_NAVY, paper_bgcolor=DARK_NAVY,
        font=dict(family="Arial, sans-serif", color=LIGHT_TEXT),
        height=300, margin=dict(t=40, b=50, l=60, r=20),
        showlegend=False,
    )
    return fig


def render():
    title_c, status_c = st.columns([4, 2])
    with title_c:
        st.subheader("Outliers Detection")
        st.caption(
            "Identifies outlier PODs via hierarchical clustering with "
            "single linkage. Clusters smaller than the threshold are flagged "
            "as outliers, capturing anomalous load profiles."
        )
    status_slot = status_c.empty()

    with st.expander("Methodology & Data-Gap Metrics"):
        st.markdown("""
**Methodology.**
- Profile aggregation: per-POD average daily 96-quarter profile [-].
- Clustering: single linkage with Euclidean distance.
- Outlier rule: clusters with fewer PODs than the threshold are flagged.

**Data gap metrics shown for each outlier POD:**
- *Days with data* — distinct days that carry at least one record.
- *Expected days* — calendar span [from first to last measurement, inclusive].
- *Missing days* — Expected − Days with data.
- *Missing %* — Missing days / Expected × 100.
""")

    # ── ATECO catalogue + tipologie ─────────────────────────────────────────
    try:
        codes_l1 = api.ateco_codes(level=1)
        codes_l2 = api.ateco_codes(level=2)
        codes_l3 = api.ateco_codes(level=3)
        descs    = api.ateco_descriptions()
        tipologie_options = api.tipologie() or ["AP"]
    except api.BackendError as e:
        st.error(f"ATECO catalog unavailable: {e}.")
        return

    st.markdown("#### ATECO Level Selection")
    selected_l1, selected_l2, selected_l3 = ateco_filter.render(
        key_prefix="od",
        codes_l1=codes_l1, codes_l2=codes_l2, codes_l3=codes_l3,
        descs=descs,
    )

    # ── Parameters ─────────────────────────────────────────────────────────
    st.markdown("#### Detection Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        tipologia = st.selectbox(
            "Measurement Type",
            options=tipologie_options,
            index=tipologie_options.index("AP") if "AP" in tipologie_options else 0,
            key="od_tipologia",
        )
    with c2:
        min_months = st.number_input(
            "Minimum Months of Data per POD", 1, 36, 12, key="od_mm",
        )
    with c3:
        normalise = st.selectbox(
            "Normalization", ["minmax", "max"],
            format_func=lambda k: "Min/Max" if k == "minmax" else "Max",
            index=0, key="od_norm",
        )

    s1, s2 = st.columns(2)
    with s1:
        n_clusters = st.slider(
            "Number of Clusters [k]", min_value=10, max_value=30,
            value=10, step=1, key="od_k",
            help="Single-linkage typically forms a few large clusters + many tiny ones."
        )
    with s2:
        threshold = st.slider(
            "Outlier Threshold (Max PODs per Cluster)",
            min_value=1, max_value=30, value=5, step=1, key="od_threshold",
            help="Clusters with fewer PODs than this value are flagged as outliers."
        )

    power_ranges, include_missing_power = power_filter(key_prefix="od")

    pod_filter = {
        "ateco_l1":   selected_l1 or None,
        "ateco_l2":   selected_l2 or None,
        "ateco_l3":   selected_l3 or None,
        "min_months": int(min_months),
        "tipologia":  tipologia,
        "power_ranges":          power_ranges,
        "include_missing_power": include_missing_power,
    }
    try:
        preview = api.pod_set_preview(pod_filter)
        st.info(f"**{fmt_int(preview['n_pods'])} PODs** available "
                f"(coverage filter alone: {fmt_int(preview['after_coverage'])}).")
    except api.BackendError as e:
        st.error(f"Preview failed: {e}.")
        return

    # ── Run ────────────────────────────────────────────────────────────────
    if st.button("▶ Perform Outlier Detection", type="primary", key="od_run"):
        if not (selected_l1 or selected_l2 or selected_l3):
            st.error("Select at least one ATECO code at any level.")
            return
        with status_slot.status("Aggregating profiles & running single-linkage.",
                                 expanded=False) as s:
            try:
                result = api.detect_outliers({
                    "filter":     pod_filter,
                    "n_clusters": int(n_clusters),
                    "threshold":  int(threshold),
                    "normalise":  normalise,
                })
                s.update(label="Outlier detection complete.", state="complete")
                st.session_state["outliers_last_result"] = {
                    "result":     result,
                    "tipologia":  tipologia,
                    "n_clusters": int(n_clusters),
                    "threshold":  int(threshold),
                    "normalise":  normalise,
                }
            except api.BackendError as e:
                s.update(label=f"Detection failed: {e}.", state="error")
                return

    # ── Render last stored result ─────────────────────────────────────────
    stored = st.session_state.get("outliers_last_result")
    if not stored:
        return

    result    = stored["result"]
    n_pods    = result["n_pods"]
    outlier_set = set(int(c) for c in result["outlier_clusters"])

    # ── Summary metrics ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PODs Analyzed:",         fmt_int(n_pods))
    c2.metric("Clusters Formed:",       fmt_int(stored["n_clusters"]))
    c3.metric("Outlier Clusters:",      fmt_int(len(outlier_set)))
    c4.metric("Outlier PODs:",          fmt_int(result["n_outliers"]))

    if not outlier_set:
        st.success(f"No outlier clusters found "
                   f"(every cluster has ≥ {stored['threshold']} PODs).")
        return

    st.markdown(
        f"**Outlier clusters** (< {stored['threshold']} PODs): "
        + ", ".join(f"Cluster {c} (n={result['cluster_sizes'].get(str(c), result['cluster_sizes'].get(c, 0))})"
                    for c in sorted(outlier_set))
    )

    # ── 1) Cluster-size overview chart ───────────────────────────────────
    st.markdown("#### Cluster Size Overview")
    sizes = {int(k): int(v) for k, v in result["cluster_sizes"].items()}
    st.plotly_chart(
        _cluster_size_chart(sizes, outlier_set, stored["threshold"]),
        use_container_width=True,
    )

    # ── 2) Outlier cluster profiles (mean ± std), 3 columns max ──────────
    st.markdown("#### Outlier Cluster Profiles")
    centroids = {int(k): v for k, v in result.get("outlier_centroids", {}).items()}
    stds      = {int(k): v for k, v in result.get("outlier_stds", {}).items()}
    out_sorted = sorted(centroids.keys())

    if out_sorted:
        # 3-column grid
        rows = [out_sorted[i:i + 3] for i in range(0, len(out_sorted), 3)]
        for row in rows:
            cols = st.columns(3)
            for i, cl in enumerate(row):
                with cols[i]:
                    st.plotly_chart(
                        _outlier_profile_chart(
                            cluster_id=cl,
                            n_pods=sizes.get(cl, 0),
                            mean_profile=centroids[cl],
                            std_profile=stds.get(cl, [0.0] * len(centroids[cl])),
                        ),
                        use_container_width=True,
                    )

    # ── 3) Cluster size table ────────────────────────────────────────────
    st.markdown("#### Cluster Size Distribution")
    size_rows = []
    for cl, n in sorted(sizes.items()):
        size_rows.append({
            "Cluster":  int(cl),
            "Size [-]": int(n),
            "Status":   "Outlier" if int(cl) in outlier_set else "Normal",
        })
    sizes_df = pd.DataFrame(size_rows)
    st.dataframe(
        sizes_df.style.format({"Size [-]": lambda v: fmt_int(v)}),
        hide_index=True, use_container_width=True,
    )

    # ── 4) Outlier POD list with data gaps ───────────────────────────────
    st.markdown("#### Outlier POD List with Data-Coverage Gaps")
    pods_list = result["outlier_pods"]
    if pods_list:
        df = pd.DataFrame(pods_list)
        df = df.rename(columns={
            "pod":            "POD",
            "cluster":        "Cluster",
            "ateco_l1":       "ATECO L1",
            "ateco_l2":       "ATECO L2",
            "days_with_data": "Days with Data [d]",
            "expected_days":  "Expected Days [d]",
            "missing_days":   "Missing Days [d]",
            "missing_pct":    "Missing Share [%]",
        })
        st.dataframe(
            df.style.format({
                "Days with Data [d]": lambda v: fmt_int(v),
                "Expected Days [d]":  lambda v: fmt_int(v),
                "Missing Days [d]":   lambda v: fmt_int(v),
                "Missing Share [%]":  "{:.2f}",
            }),
            hide_index=True, use_container_width=True,
        )

    import json
    st.download_button(
        "⬇ Download Outlier List (JSON)",
        data=json.dumps(result, indent=2).encode("utf-8"),
        file_name=f"outliers_k{stored['n_clusters']}_t{stored['threshold']}_"
                  f"{stored['tipologia']}.json",
        mime="application/json",
    )
