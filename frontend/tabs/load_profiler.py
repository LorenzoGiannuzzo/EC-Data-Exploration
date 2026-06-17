"""Load Profiler tab — extract representative hourly profiles.

UI is fully in English. All selectors start empty: the user must explicitly
choose an end-use category, optionally a power range, and a market zone
before the Run button does anything. ATECO codes shown in the multiselects
are limited to those with at least one POD matching the coverage threshold,
so the L1-only path always returns a non-empty result.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api
from charts import DARK_NAVY, GRID_LIGHT, LIGHT_TEXT, fmt_int
from components.power_filter import power_filter

# ── Static catalogs ─────────────────────────────────────────────────────────
DAY_TYPES   = ["weekday", "saturday", "sunday"]
DAY_LABELS  = {"weekday": "Weekday", "saturday": "Saturday", "sunday": "Sunday"}
DAY_COLORS  = {"weekday": "#4fa3ff", "saturday": "#ff9f43", "sunday": "#e84545"}
PERCENTILES = ["p5", "p25", "p50", "p75", "p95"]

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

# Italian DSO measurement codes — common AP/AN/RP/RN/RC*/RI* family.
# Unknown codes fall back to "code" only.
TIPOLOGIA_LABELS: dict[str, str] = {
    "AP":  "Active energy drawn — load (Attiva Prelevata)",
    "AN":  "Active energy injected — generation (Attiva Negativa/Immessa)",
    "RP":  "Reactive energy drawn (Reattiva Prelevata)",
    "RN":  "Reactive energy injected (Reattiva Negativa)",
    "RCP": "Reactive capacitive drawn (Reattiva Capacitiva Prelevata)",
    "RCN": "Reactive capacitive injected (Reattiva Capacitiva Negativa)",
    "RIP": "Reactive inductive drawn (Reattiva Induttiva Prelevata)",
    "RIN": "Reactive inductive injected (Reattiva Induttiva Negativa)",
}

# L1 description overrides — used only at level 1 to give domestic codes a
# clearer human-readable name regardless of the DB description column.
L1_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "DO":    "Domestic",
    "DO.R":  "Domestic — Residente",
    "DO.NR": "Domestic — Non Residente",
}

GRANULARITY_HELP = {
    "daily":   "Full 8760-hour calendar year with probabilistic bands "
               "(P5/P25/P50/P75/P95) per (month, day-type, hour) bucket.",
    "monthly": "12 mean profiles per day-type (36 total), one per calendar "
               "month — useful for seasonality comparisons.",
    "annual":  "3 mean profiles (Weekday / Saturday / Sunday) pooled across "
               "the year, n-weighted to match a direct overall average.",
}


# ── UI helpers ──────────────────────────────────────────────────────────────
def _tipologia_label(code: str) -> str:
    desc = TIPOLOGIA_LABELS.get(code)
    return f"{code} — {desc}" if desc else code


def _ateco_label(
    code: str, n_covered: int, n_total: int, desc: str | None, *, level: int,
) -> str:
    """Render an ATECO multiselect option label.

    Format: ``CODE — Description  ✓ (n_covered / n_total PODs)``.
    The two numbers reveal how aggressively the coverage filter is cutting
    the catalogue: a low ratio (e.g. ``124 / 4 000``) almost always means
    the selected ``tipologia`` is sparse for that ATECO (typical case:
    using AN for domestic PODs, which rarely inject energy).
    """
    if level == 1 and code in L1_DESCRIPTION_OVERRIDES:
        d = L1_DESCRIPTION_OVERRIDES[code]
    else:
        d = (desc or "").strip()
    base = f"{code} — {d}" if d else code
    if n_total > n_covered:
        return f"{base}  ✓ ({fmt_int(n_covered)} / {fmt_int(n_total)} PODs)"
    return f"{base}  ✓ ({fmt_int(n_covered)} PODs)"


def _zone_label(z: dict) -> str:
    if z["available"]:
        return f"{z['label']}  ✓ ({fmt_int(z['n_pods'])} PODs)"
    return f"{z['label']} — Not available"


# ── Charts ──────────────────────────────────────────────────────────────────
def _bucket_chart(buckets: pd.DataFrame, granularity: str) -> go.Figure:
    """12-month small multiples grid (4 cols × 3 rows).

    The x-axis title appears only on the bottom row and the y-axis title only
    on the left column, with extra vertical spacing between rows so the
    subplot titles never overlap the axis labels from the row above.
    """
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=4, shared_yaxes=False, shared_xaxes=True,
        subplot_titles=[MONTH_NAMES[m] for m in range(1, 13)],
        vertical_spacing=0.16, horizontal_spacing=0.04,
    )
    legend_emitted: set[str] = set()
    for m in range(1, 13):
        r, c = (m - 1) // 4 + 1, (m - 1) % 4 + 1
        sub = (buckets[buckets["month"] == m]
               if "month" in buckets.columns else buckets)
        for dt in DAY_TYPES:
            d = sub[sub["day_type"] == dt].sort_values("hour")
            if d.empty:
                continue
            color = DAY_COLORS[dt]
            show_legend = dt not in legend_emitted
            legend_emitted.add(dt)
            if granularity == "daily" and "p25" in d.columns:
                # The P25-P75 envelope tells the user how spread the per-POD
                # distribution is at each hour; the SOLID line is the MEAN
                # (not the median): for residential load profiling the mean
                # is the more useful central tendency since it's additive
                # (total demand = N × mean) and it reflects the long tail
                # of high-consumption households that the median washes out.
                band_color = (f"rgba({int(color[1:3], 16)},"
                              f"{int(color[3:5], 16)},"
                              f"{int(color[5:7], 16)},0.12)")
                fig.add_trace(go.Scatter(
                    x=d["hour"], y=d["p75"], mode="lines",
                    line=dict(width=0), showlegend=False,
                    hoverinfo="skip"), row=r, col=c)
                fig.add_trace(go.Scatter(
                    x=d["hour"], y=d["p25"], mode="lines", fill="tonexty",
                    fillcolor=band_color, line=dict(width=0),
                    showlegend=False, hoverinfo="skip"), row=r, col=c)
                fig.add_trace(go.Scatter(
                    x=d["hour"], y=d["mean"], mode="lines",
                    line=dict(color=color, width=2.5),
                    name=DAY_LABELS[dt], legendgroup=dt,
                    showlegend=show_legend), row=r, col=c)
            else:
                fig.add_trace(go.Scatter(
                    x=d["hour"], y=d["mean"], mode="lines",
                    line=dict(color=color, width=2),
                    name=DAY_LABELS[dt], legendgroup=dt,
                    showlegend=show_legend), row=r, col=c)

    # Strip default axis titles, then re-add only on outer edges.
    fig.update_xaxes(gridcolor="rgba(208, 223, 240, 0.15)",
                     color=LIGHT_TEXT, dtick=6, range=[0, 23],
                     title_text="", showline=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(208, 223, 240, 0.15)",
                     color=LIGHT_TEXT, title_text="",
                     showline=False, zeroline=False)
    for c in range(1, 5):
        fig.update_xaxes(title_text="Hour of day [h]",
                         title_font=dict(size=10), row=3, col=c)
    for r in range(1, 4):
        fig.update_yaxes(title_text="kWh", title_font=dict(size=10),
                         row=r, col=1)

    fig.update_layout(
        height=620, margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor=DARK_NAVY, plot_bgcolor=DARK_NAVY,
        font=dict(color=LIGHT_TEXT, size=11),
        legend=dict(orientation="h", y=-0.10, x=0.5, xanchor="center",
                    font=dict(size=11)),
    )
    # Lift the per-subplot titles a touch so they no longer collide with the
    # axis labels of the row above.
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=LIGHT_TEXT)
        ann.yshift = 8
    return fig


def _annual_chart(annual: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for dt in DAY_TYPES:
        d = annual[annual["day_type"] == dt].sort_values("hour")
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["hour"], y=d["mean"], mode="lines+markers",
            line=dict(color=DAY_COLORS[dt], width=2.5),
            marker=dict(size=5),
            name=DAY_LABELS[dt],
        ))
    fig.update_layout(
        height=380, paper_bgcolor=DARK_NAVY, plot_bgcolor=DARK_NAVY,
        font=dict(color=LIGHT_TEXT),
        xaxis=dict(title="Hour of day [h]",
                   gridcolor="rgba(208, 223, 240, 0.15)",
                   showline=False, zeroline=False, dtick=2),
        yaxis=dict(title="kWh",
                   gridcolor="rgba(208, 223, 240, 0.15)",
                   showline=False, zeroline=False),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ── Export helpers ──────────────────────────────────────────────────────────
def _frame_for_export(
    granularity: str, orientation: str, response: dict,
) -> pd.DataFrame:
    buckets = pd.DataFrame(response["buckets"])

    if granularity == "annual":
        long = pd.DataFrame(response["annual"] or [])
        if orientation == "long":
            return long
        piv = long.pivot(index="day_type", columns="hour", values="mean")
        piv = piv.reindex(DAY_TYPES)
        piv.columns = [f"h{h}" for h in piv.columns]
        return piv.reset_index().rename(columns={"day_type": "Day Type"})

    if granularity == "monthly":
        if orientation == "long":
            return buckets
        piv = buckets.pivot_table(
            index=["month", "day_type"], columns="hour", values="mean",
        )
        piv.columns = [f"h{h}" for h in piv.columns]
        return piv.reset_index().rename(
            columns={"month": "Month", "day_type": "Day Type"}
        )

    # daily
    long = pd.DataFrame(response["daily_8760"] or [])
    if long.empty:
        return long
    long["timestamp"] = pd.to_datetime(long["timestamp"])
    if orientation == "long":
        value_cols = [c for c in ["mean", "std", *PERCENTILES]
                      if c in long.columns]
        return long.melt(
            id_vars=["timestamp", "month", "day_type", "hour", "n"],
            value_vars=value_cols,
            var_name="statistic", value_name="value",
        )
    cols = ["timestamp", "mean", "std", "n"]
    cols += [p for p in PERCENTILES if p in long.columns]
    return long[cols].copy()


def _serialize(df: pd.DataFrame, fmt: str) -> tuple[bytes, str]:
    if fmt == "CSV":
        return (df.to_csv(index=False).encode("utf-8"), "text/csv")
    if fmt == "Parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        return (buf.getvalue(), "application/octet-stream")
    if fmt == "XLSX":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="profiles", index=False)
        return (buf.getvalue(),
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet")
    if fmt == "JSON":
        return (df.to_json(orient="records", date_format="iso",
                           indent=2).encode("utf-8"),
                "application/json")
    if fmt == "Feather":
        buf = io.BytesIO()
        df.to_feather(buf)
        return (buf.getvalue(), "application/octet-stream")
    raise ValueError(f"Unknown format: {fmt}")


# ── Main render ─────────────────────────────────────────────────────────────
def render() -> None:
    st.markdown("### Load Profiler")
    st.caption(
        "Extract representative hourly load profiles for a specific end-use "
        "category × contractual power × market zone slice. The output is "
        "always grounded in the real measurements — no synthetic curves, "
        "no clustering."
    )

    # ── 0) Tipologia + minimum months — declared first so we can use them
    #      to scope the ATECO-availability counts that follow.
    c_tip, c_mm = st.columns([2, 1])
    with c_tip:
        _tipologie = api.tipologie()
        _default_idx = _tipologie.index("AP") if "AP" in _tipologie else 0
        tipologia = st.selectbox(
            "Measurement type",
            options=_tipologie,
            format_func=_tipologia_label,
            index=_default_idx, key="lp_tip",
            help="AP (active drawn) is by far the densest series — pick AN "
                 "only if you actually want injected energy (PV / CHP), and "
                 "expect the POD counts below to drop sharply since few "
                 "non-residential PODs inject anything.",
        )
    with c_mm:
        min_months = st.number_input(
            "Min months of data per POD", 0, 36, 0, key="lp_mm",
            help="0 (default) ⇒ no coverage filter: every POD that matches "
                 "the ATECO / power / zone selection contributes, regardless "
                 "of how many months of data it has. Raise this only if you "
                 "want to drop poorly-covered PODs from the average.",
        )

    # ── 1) End-use category (ATECO with descriptions, searchable) ───────────
    st.markdown("#### End-use category (ATECO)")
    st.caption(
        "Type any keyword (description or code) to filter the list. "
        "Selections across levels are combined with OR. Each label shows "
        "**n_covered / n_total** PODs — i.e. PODs satisfying the current "
        "Measurement Type + Min Months filters, versus the full pod_metadata "
        "count. When the ratio is very small (e.g. 124 / 4 000), the chosen "
        "tipologia is the likely culprit — try AP for the bulk of the load "
        "data."
    )
    try:
        l1_avail = api.load_profiler_ateco_availability(
            level=1, tipologia=tipologia, min_months=int(min_months))["codes"]
    except api.BackendError as e:
        st.error(f"Could not load ATECO availability: {e}")
        return

    l1_codes = [r["code"] for r in l1_avail]
    l1_lbl = {r["code"]: _ateco_label(r["code"], r["n_pods"], r.get("n_pods_total", r["n_pods"]),
                                       r["description"], level=1)
              for r in l1_avail}

    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        selected_l1 = st.multiselect(
            "Level 1 — Section",
            options=l1_codes, format_func=lambda c: l1_lbl.get(c, c),
            default=[], placeholder="Type to search…", key="lp_l1",
        )

    # L2 cascades from L1: when an L1 is picked, only its children appear.
    try:
        if selected_l1:
            l2_avail = api.load_profiler_ateco_subcodes(
                target_level=2, parent_l1=",".join(selected_l1),
                tipologia=tipologia, min_months=int(min_months))["codes"]
        else:
            l2_avail = api.load_profiler_ateco_availability(
                level=2, tipologia=tipologia,
                min_months=int(min_months))["codes"]
    except api.BackendError as e:
        st.error(f"Could not load L2 subcodes: {e}")
        return
    l2_codes = [r["code"] for r in l2_avail]
    l2_lbl = {r["code"]: _ateco_label(r["code"], r["n_pods"], r.get("n_pods_total", r["n_pods"]),
                                       r["description"], level=2)
              for r in l2_avail}
    with col_l2:
        # Drop stale L2 selections that no longer fall under the new L1 set.
        if "lp_l2" in st.session_state:
            st.session_state["lp_l2"] = [
                c for c in st.session_state["lp_l2"] if c in l2_codes
            ]
        selected_l2 = st.multiselect(
            "Level 2 — Division",
            options=l2_codes, format_func=lambda c: l2_lbl.get(c, c),
            default=[], placeholder="Type to search…", key="lp_l2",
        )

    # L3 cascades from L1+L2 with the same logic.
    try:
        if selected_l1 or selected_l2:
            l3_avail = api.load_profiler_ateco_subcodes(
                target_level=3,
                parent_l1=",".join(selected_l1),
                parent_l2=",".join(selected_l2),
                tipologia=tipologia, min_months=int(min_months))["codes"]
        else:
            l3_avail = api.load_profiler_ateco_availability(
                level=3, tipologia=tipologia,
                min_months=int(min_months))["codes"]
    except api.BackendError as e:
        st.error(f"Could not load L3 subcodes: {e}")
        return
    l3_codes = [r["code"] for r in l3_avail]
    l3_lbl = {r["code"]: _ateco_label(r["code"], r["n_pods"], r.get("n_pods_total", r["n_pods"]),
                                       r["description"], level=3)
              for r in l3_avail}
    with col_l3:
        if "lp_l3" in st.session_state:
            st.session_state["lp_l3"] = [
                c for c in st.session_state["lp_l3"] if c in l3_codes
            ]
        selected_l3 = st.multiselect(
            "Level 3 — Class",
            options=l3_codes, format_func=lambda c: l3_lbl.get(c, c),
            default=[], placeholder="Type to search…", key="lp_l3",
        )
    if not (selected_l1 or selected_l2 or selected_l3):
        st.caption("⚠ Pick at least one ATECO code (any level). With no "
                   "ATECO filter the request would aggregate every POD "
                   "with the minimum coverage.")

    # ── 2) Contractual power (starts empty) ─────────────────────────────────
    st.markdown("#### Contractual power")
    power_ranges, include_missing_power = power_filter(
        key_prefix="lp", default_all=False,
    )

    # ── 3) Market zone (starts empty, with availability ticks) ──────────────
    st.markdown("#### Market zone")
    try:
        zones_info = api.load_profiler_zones()
    except api.BackendError as e:
        st.error(f"Could not load market zones: {e}")
        return
    if zones_info["n_geocoded"] == 0:
        st.warning(
            "No PODs have a resolved market zone yet — the Overview tab "
            "geocoding might still be running. You can launch the Load "
            "Profiler without a zone filter and still get a valid result."
        )
    available_codes = [z["code"] for z in zones_info["zones"] if z["available"]]
    label_map = {z["code"]: _zone_label(z) for z in zones_info["zones"]}
    selected_zones = st.multiselect(
        "Select one or more zones — ✓ = data available, "
        "“Not available” options are ignored",
        options=[z["code"] for z in zones_info["zones"]],
        default=[],
        format_func=lambda c: label_map[c],
        key="lp_zones",
    )
    selected_zones = [z for z in selected_zones if z in available_codes]

    # ── 4) Granularity ──────────────────────────────────────────────────────
    st.markdown("#### Output granularity")
    granularity = st.radio(
        "Granularity",
        options=["daily", "monthly", "annual"],
        format_func=lambda g: {
            "daily":   "Daily (probabilistic, 8760 hours)",
            "monthly": "Monthly (12 × day-type means)",
            "annual":  "Annual (3 day-type means)",
        }[g],
        index=2, horizontal=True, key="lp_gran",
        help="Daily is the most informative but the heaviest query — it "
             "computes percentiles per bucket via PERCENTILE_CONT in SQL.",
    )
    st.caption(GRANULARITY_HELP[granularity])

    payload = {
        "ateco_l1":              selected_l1 or None,
        "ateco_l2":              selected_l2 or None,
        "ateco_l3":              selected_l3 or None,
        "power_ranges":          power_ranges,
        "include_missing_power": include_missing_power,
        "zones":                 selected_zones or None,
        "min_months":            int(min_months),
        "tipologia":             tipologia,
        "granularity":           granularity,
    }

    # ── 5) Pre-Run note ─────────────────────────────────────────────────────
    if not (selected_l1 or selected_l2 or selected_l3 or selected_zones
            or power_ranges):
        st.caption(
            "No filters selected — the run will aggregate every POD with "
            "the chosen Measurement Type and Min Months coverage."
        )

    # ── 6) Run ──────────────────────────────────────────────────────────────
    if st.button("▶ Compute Load Profiles", type="primary",
                 use_container_width=True, key="lp_run"):
        with st.status("Computing profiles…", expanded=False) as s:
            try:
                response = api.run_load_profiler(payload)
                s.update(label="Profiles ready.", state="complete")
                st.session_state["lp_last"] = {
                    "response": response, "payload": payload,
                }
            except api.BackendError as e:
                s.update(label="Failed.", state="error")
                st.error(str(e))
                return

    last = st.session_state.get("lp_last")
    if not last:
        return

    response = last["response"]
    granularity = response["granularity"]
    sel = response["selection"]

    # ── 7) Result ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Result — {granularity.title()} Granularity")

    # Single, headline number — no intermediate filter counts. The user asked
    # for a specific selection; showing how many PODs each filter shed in
    # isolation just adds noise to the result.
    n_final = sel["n_pods"]
    st.metric("PODs in the selection", fmt_int(n_final))

    # When the result is based on very few PODs, the chart below will have
    # gaps (months / day-types without measurements). Tell the user plainly
    # so they don't mistake the gaps for a rendering bug — no numbers, just
    # a clear-language note.
    if n_final < 10:
        st.warning(
            "Few PODs survived the selection. Empty months or missing "
            "day-types in the chart below are calendar gaps in those "
            "POD(s) own data — they are not a bug. If you want denser "
            "coverage, broaden the ATECO selection or lower **Min months "
            "of data per POD** at the top of this tab."
        )

    buckets = pd.DataFrame(response["buckets"])
    if granularity == "annual":
        annual = pd.DataFrame(response["annual"])
        st.plotly_chart(_annual_chart(annual), use_container_width=True)
        with st.expander("Show 12-month panel for reference"):
            st.plotly_chart(_bucket_chart(buckets, "monthly"),
                            use_container_width=True)
    else:
        st.plotly_chart(_bucket_chart(buckets, granularity),
                        use_container_width=True)
        if granularity == "daily":
            st.caption(
                "Shaded band: P25–P75 spread of per-POD hourly consumption. "
                "Solid line: **mean** across PODs (the additive central "
                "tendency used for load profiling). The 8760-hour file you "
                "download below contains all of P5/P25/P50/P75/P95 plus "
                "mean (kWh), std, and the per-bucket sample count."
            )

    # ── 8) Export ───────────────────────────────────────────────────────────
    st.markdown("#### Download")
    c_fmt, c_or = st.columns([1, 1])
    with c_fmt:
        fmt = st.selectbox(
            "File format",
            options=["CSV", "Parquet", "XLSX", "JSON", "Feather"],
            index=0, key="lp_fmt",
            help="CSV / XLSX for spreadsheets; Parquet / Feather for "
                 "scientific pipelines; JSON for cross-language use.",
        )
    with c_or:
        orientation = st.radio(
            "Layout",
            options=["wide", "long"], horizontal=True, key="lp_or",
            format_func=lambda o: o.capitalize(),
            help="Wide = one row per (period, day-type) with hour columns. "
                 "Long = one row per (period, day-type, hour) — easier for "
                 "Plotly / ggplot / SQL ingestion.",
        )
    try:
        df = _frame_for_export(granularity, orientation, response)
        data, mime = _serialize(df, fmt)
        st.success(f"Ready — {fmt_int(len(df))} rows × {df.shape[1]} columns.")
        ext = {"CSV": "csv", "Parquet": "parquet", "XLSX": "xlsx",
               "JSON": "json", "Feather": "feather"}[fmt]
        fname = (f"load_profile_{granularity}_{orientation}_"
                 f"{datetime.now():%Y%m%d_%H%M}.{ext}")
        st.download_button(
            f"⬇ Download {fmt} ({orientation})",
            data=data, file_name=fname, mime=mime,
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Could not prepare the file: {e}")
