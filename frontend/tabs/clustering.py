"""Clustering Explorer tab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api
from charts import (DARK_NAVY, GRID_LIGHT, LIGHT_TEXT, centroid_chart,
                    cluster_sizes_chart, fmt_int, pearson_heatmap)
from components import ateco_filter
from components.power_filter import power_filter
from helpers import fmt_label

MONTH_NAMES = {
    0: "Annual Average", 1: "January", 2: "February", 3: "March",
    4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
COMPOSITION_COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1


TIPOLOGIA_LABELS = {
    "AP":  "Active Power Drawn (AP) [kWh]",
    "AN":  "Active Power Injected (AN) [kWh]",
    "RLP": "Reactive Power, Lagging — Drawn (RLP) [kVArh]",
    "RLN": "Reactive Power, Lagging — Injected (RLN) [kVArh]",
    "RCP": "Reactive Power, Capacitive — Drawn (RCP) [kVArh]",
    "RCN": "Reactive Power, Capacitive — Injected (RCN) [kVArh]",
}
NORM_LABELS    = {"minmax": "Min/Max", "max": "Max"}
LINKAGE_LABELS = {
    "ward": "Ward", "average": "Average", "complete": "Complete", "single": "Single",
}
LEVEL_NAMES = {1: "Level 1 — Section", 2: "Level 2 — Division", 3: "Level 3 — Class"}


# ── Result panel for a single clustering ────────────────────────────────────
def _render_panel(level_label: str, response: dict):
    m = response["metrics"]
    st.markdown(f"##### {level_label}")

    sizes = {int(k): v for k, v in m["sizes"].items()}

    # 1) Centroids chart
    st.plotly_chart(
        centroid_chart({int(k): v for k, v in response["centroids"].items()}, sizes,
                       title=f"Cluster Centroids — {level_label}"),
        use_container_width=True,
    )

    # 2) Cluster sizes chart
    st.plotly_chart(cluster_sizes_chart(sizes), use_container_width=True)

    # 3) Pearson correlation heatmap
    if response.get("pearson_matrix"):
        st.plotly_chart(
            pearson_heatmap(response["pearson_matrix"], response["pearson_labels"]),
            use_container_width=True,
        )

    # 4) Compact metrics table (replaces scattered st.metric widgets)
    st.markdown("###### Cluster Quality Metrics")
    metrics_rows = [
        ("PODs Clustered [-]",     fmt_int(m["n_pods"])),
        ("Number of Clusters [-]", fmt_int(m["n_clusters"])),
        ("Silhouette Score [-]",
         f"{m['silhouette']:.4f}" if m["silhouette"] is not None else "N/A"),
        ("Calinski–Harabasz [-]",
         f"{m['calinski_harabasz']:.1f}" if m["calinski_harabasz"] is not None else "N/A"),
        ("Davies–Bouldin [-]",
         f"{m['davies_bouldin']:.4f}" if m["davies_bouldin"] is not None else "N/A"),
    ]
    metrics_df = pd.DataFrame(metrics_rows, columns=["Metric", "Value"])
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # 5) Dominant ATECO codes per cluster (top 3, computed from the full
    #    breakdown so the same data also feeds the composition chart below)
    full_bdf = pd.DataFrame(response["ateco_breakdown"])
    st.markdown("###### Dominant ATECO Codes per Cluster")
    if not full_bdf.empty:
        dom = (
            full_bdf.sort_values(["cluster", "n_pods"],
                                 ascending=[True, False])
            .groupby("cluster", as_index=False, group_keys=False)
            .head(3)
            .rename(columns={
                "cluster":        "Cluster",
                "ateco":          "ATECO",
                "description":    "Description",
                "n_pods":         "Number of PODs [-]",
                "pct_of_cluster": "Share of Cluster [%]",
            })
        )
        st.dataframe(
            dom.style.format({"Share of Cluster [%]": "{:.1f}",
                              "Number of PODs [-]":   lambda v: fmt_int(v)}),
            hide_index=True, use_container_width=True,
        )

    # 6) Cluster composition — stacked bar per cluster (legacy
    #    `plot_cluster_composition`), built from the full breakdown
    if not full_bdf.empty:
        st.markdown("###### Cluster Composition by ATECO")
        fig_comp = go.Figure()
        ateco_codes = sorted(full_bdf["ateco"].fillna("N/A").unique())
        for i, code in enumerate(ateco_codes):
            sub = full_bdf[full_bdf["ateco"].fillna("N/A") == code]
            fig_comp.add_trace(go.Bar(
                x=[f"Cl. {int(c)}" for c in sub["cluster"]],
                y=sub["pct_of_cluster"],
                name=str(code),
                marker_color=COMPOSITION_COLORS[i % len(COMPOSITION_COLORS)],
                customdata=sub[["n_pods"]],
                hovertemplate=("%{x} — " + str(code) +
                               ": %{y:.1f}% (%{customdata[0]} PODs)"
                               "<extra></extra>"),
            ))
        fig_comp.update_layout(
            barmode="stack", height=320,
            paper_bgcolor=DARK_NAVY, plot_bgcolor=DARK_NAVY,
            font=dict(color=LIGHT_TEXT, size=11),
            yaxis=dict(title="Share of Cluster [%]", gridcolor=GRID_LIGHT,
                       range=[0, 100]),
            xaxis=dict(title=""),
            legend=dict(font=dict(size=9), orientation="h",
                        yanchor="bottom", y=1.02),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # 7) Full ATECO → Cluster pivot (legacy `build_ateco_cluster_breakdown`)
        with st.expander("ATECO → Cluster Breakdown (full pivot)"):
            piv_n = full_bdf.pivot_table(index="ateco", columns="cluster",
                                         values="n_pods", aggfunc="sum",
                                         fill_value=0)
            totals = piv_n.sum(axis=1)
            cells = piv_n.copy().astype(object)
            for col in piv_n.columns:
                cells[col] = [
                    (f"{fmt_int(n)} ({n / t * 100:.1f}%)" if n > 0 else "—")
                    for n, t in zip(piv_n[col], totals)
                ]
            cells.insert(0, "Total PODs [-]", [fmt_int(t) for t in totals])
            cells.columns = (["Total PODs [-]"]
                             + [f"Cluster {int(c)}" for c in piv_n.columns])
            cells.index.name = "ATECO"
            desc_map = (full_bdf.dropna(subset=["ateco"])
                        .drop_duplicates("ateco")
                        .set_index("ateco")["description"].to_dict())
            cells.insert(0, "Description",
                         [str(desc_map.get(a, "") or "")[:60]
                          for a in cells.index])
            st.dataframe(cells, use_container_width=True)
            st.caption("Each cell: PODs of that ATECO code assigned to the "
                       "cluster, with the share of the code's total.")


def render():
    # Title + status placeholder (status appears next to title when running)
    title_c, status_c = st.columns([4, 2])
    with title_c:
        st.subheader("Clustering Explorer")
        st.caption(
            "Hierarchical clustering on the per-POD daily profile. ."
            "When you select codes at multiple ATECO levels a separate "
            "clustering runs for each level."
        )
    status_slot = status_c.empty()

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
        key_prefix="cl",
        codes_l1=codes_l1, codes_l2=codes_l2, codes_l3=codes_l3,
        descs=descs,
    )

    # ── Clustering parameters ──────────────────────────────────────────────
    st.markdown("#### Clustering Parameters")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        tipologia = st.selectbox(
            "Measurement Type",
            options=tipologie_options,
            format_func=lambda t: TIPOLOGIA_LABELS.get(t, t),
            index=tipologie_options.index("AP") if "AP" in tipologie_options else 0,
            key="cl_tipologia",
        )
    with c2:
        cluster_mode = st.radio("Cluster Count Mode", ["Manual", "Auto"],
                                 horizontal=True, key="cl_mode",
                                 help="Auto: k chosen server-side by the legacy "
                                      "multi-metric vote (Silhouette, "
                                      "Calinski-Harabasz, Davies-Bouldin, Elbow).")
    auto_k = cluster_mode == "Auto"
    with c3:
        if not auto_k:
            n_clusters = st.slider("Number of Clusters", 2, 15, 5, key="cl_k")
        else:
            n_clusters = 5  # placeholder; server replaces it when auto_k=True
            st.caption("k selected automatically (scan k = 3 … min(10, √n) "
                       "with a 4-selector vote).")

    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        normalise = st.selectbox(
            "Normalization",
            options=list(NORM_LABELS.keys()),
            format_func=lambda k: NORM_LABELS[k],
            index=0, key="cl_norm",
        )
    with c5:
        method = st.selectbox(
            "Linkage Method",
            options=list(LINKAGE_LABELS.keys()),
            format_func=lambda k: LINKAGE_LABELS[k],
            index=1, key="cl_method",
            help="Original dashboard uses 'Average' linkage."
        )
    with c6:
        min_months = st.number_input(
            "Minimum Months of Data per POD", 1, 36, 12, key="cl_min_months",
        )

    profile_month = st.selectbox(
        "Profile Period",
        options=list(MONTH_NAMES.keys()),
        format_func=lambda m: MONTH_NAMES[m],
        index=0, key="cl_month",
        help="Annual Average: cluster on the all-days average profile. "
             "Single month: cluster only on that month's average profile.",
    )

    # ── Contractual power filter (legacy sidebar filter, per-range) ────────
    power_ranges, include_missing_power = power_filter(key_prefix="cl")

    # ── POD preview ────────────────────────────────────────────────────────
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
    if st.button("▶ Run Clustering", type="primary", key="cl_run"):

        if not (selected_l1 or selected_l2 or selected_l3):
            st.error("Select at least one ATECO code at any level.")
            return

        with status_slot.status("Aggregating profiles & clustering server-side.",
                                 expanded=False) as s:
            try:
                result = api.run_clustering_by_level({
                    "ateco_l1":   selected_l1 or None,
                    "ateco_l2":   selected_l2 or None,
                    "ateco_l3":   selected_l3 or None,
                    "min_months": int(min_months),
                    "tipologia":  tipologia,
                    "n_clusters": int(n_clusters),
                    "month":      int(profile_month),
                    "auto_k":     auto_k,
                    "method":     method,
                    "normalise":  normalise,
                    "top_ateco_per_cluster": 0,   # 0 = full breakdown
                    "power_ranges":          power_ranges,
                    "include_missing_power": include_missing_power,
                })
                s.update(label="Clustering complete.", state="complete")
                st.session_state["clustering_last_result"] = {
                    "result":     result,
                    "tipologia":  tipologia,
                    "n_clusters": int(n_clusters),
                    "auto_k":     auto_k,
                    "month":      int(profile_month),
                    "method":     method,
                    "normalise":  normalise,
                    "pod_filter": dict(pod_filter),
                }
            except api.BackendError as e:
                s.update(label=f"Clustering failed: {e}.", state="error")
                return

    # ── Render the last stored result (persists across tab switches) ──────
    stored = st.session_state.get("clustering_last_result")
    if not stored:
        return

    result     = stored["result"]
    tipologia  = stored["tipologia"]
    n_clusters = stored["n_clusters"]
    method     = stored["method"]
    normalise  = stored["normalise"]

    if not result["results"]:
        st.warning("No level produced a clustering — check your filters.")
        return

    valid = [r for r in result["results"] if r["response"]["n_pods"] > 0]
    if not valid:
        st.warning("No level produced a non-empty clustering.")
        return

    _period = MONTH_NAMES.get(stored.get("month", 0), "Annual Average")
    _hdr = f"Profile period: **{_period}**"
    if stored.get("auto_k"):
        _picked = sorted({
            r["response"]["metrics"]["n_clusters"] for r in valid
        })
        _hdr += (f" — k selected automatically: "
                 f"**{', '.join(str(k) for k in _picked)}**")
    st.caption(_hdr)

    cols = st.columns(len(valid))
    for col, lvl_result in zip(cols, valid):
        with col:
            level = lvl_result["level"]
            _render_panel(LEVEL_NAMES[level], lvl_result["response"])
            _akd = lvl_result["response"].get("auto_k_details")
            if _akd:
                with st.expander("Auto-k Selection Details"):
                    picks = _akd.get("method_picks", {})
                    votes = _akd.get("votes", {})
                    st.dataframe(
                        pd.DataFrame(
                            [(sel, f"k = {k}") for sel, k in picks.items()],
                            columns=["Selector", "Pick"],
                        ),
                        hide_index=True, use_container_width=True,
                    )
                    st.caption("Votes per k: "
                               + ", ".join(f"k={k}: {v}"
                                           for k, v in sorted(
                                               votes.items(),
                                               key=lambda x: int(x[0]))))

    # ── Monthly Breakdown (legacy section): one clustering per month ────────
    st.markdown("---")
    with st.expander("Monthly Breakdown — re-run the clustering month by month"):
        st.caption(
            "Runs the same clustering (same filters, k, method, "
            "normalization) twelve times, once per calendar month, on the "
            "highest selected ATECO level. Heavier than a single run — "
            "launch it explicitly."
        )
        _mbk_level_opts = [r["level"] for r in valid]
        _mbk_level = st.selectbox(
            "ATECO level for the breakdown",
            options=_mbk_level_opts,
            format_func=lambda l: LEVEL_NAMES[l],
            key="cl_mbk_level",
        )
        if st.button("Run Monthly Breakdown", key="cl_mbk_run"):
            _f = stored.get("pod_filter", {})
            _codes = next(r["ateco_filter"] for r in valid
                          if r["level"] == _mbk_level)
            _prog = st.progress(0.0, text="Starting monthly breakdown…")
            _monthly: dict[int, dict] = {}
            for _m in range(1, 13):
                _prog.progress(_m / 12,
                               text=f"Clustering {MONTH_NAMES[_m]} ({_m}/12)…")
                try:
                    _r = api.run_clustering_by_level({
                        "ateco_l1":   _codes if _mbk_level == 1 else None,
                        "ateco_l2":   _codes if _mbk_level == 2 else None,
                        "ateco_l3":   _codes if _mbk_level == 3 else None,
                        "min_months": _f.get("min_months", 12),
                        "tipologia":  stored["tipologia"],
                        "n_clusters": int(stored["n_clusters"]),
                        "month":      _m,
                        "auto_k":     False,
                        "method":     stored["method"],
                        "normalise":  stored["normalise"],
                        "top_ateco_per_cluster": 0,
                        "power_ranges":          _f.get("power_ranges"),
                        "include_missing_power": _f.get(
                            "include_missing_power", False),
                    })
                    _lvl = next((x for x in _r["results"]
                                 if x["level"] == _mbk_level), None)
                    if _lvl and _lvl["response"]["n_pods"] > 0:
                        _monthly[_m] = _lvl["response"]
                except api.BackendError:
                    continue
            _prog.empty()
            st.session_state["cl_mbk_result"] = {
                "level": _mbk_level, "monthly": _monthly,
            }

        _mbk = st.session_state.get("cl_mbk_result")
        if _mbk and _mbk.get("level") == _mbk_level:
            if not _mbk["monthly"]:
                st.warning("No month produced a non-empty clustering.")
            for _m, _resp in sorted(_mbk["monthly"].items()):
                _sizes = {int(k): v
                          for k, v in _resp["metrics"]["sizes"].items()}
                st.plotly_chart(
                    centroid_chart(
                        {int(k): v for k, v in _resp["centroids"].items()},
                        _sizes,
                        title=(f"Cluster Centroids — {MONTH_NAMES[_m]} "
                               f"({fmt_int(_resp['n_pods'])} PODs)"),
                    ),
                    use_container_width=True,
                    key=f"cl_mbk_chart_{_m}",
                )

    import json
    st.download_button(
        "⬇ Download Full Result (JSON)",
        data=json.dumps(result, indent=2).encode("utf-8"),
        file_name=(f"clustering_{tipologia}_k{n_clusters}_{method}_"
                   f"{normalise}_m{stored.get('month', 0)}.json"),
        mime="application/json",
    )
