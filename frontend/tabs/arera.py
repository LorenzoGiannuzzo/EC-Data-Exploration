"""ARERA Profile Comparison tab."""

from __future__ import annotations

import numpy as np
import streamlit as st

import api
from charts import arera_single_chart, fmt_int
from helpers import fmt_label


MONTH_LABELS = {
    0: "Annual Average",
    **{m: f"Month {m:02d}" for m in range(1, 13)},
}
ALL_MARKETS_LABEL = "All Available Markets"


def _safe_24h(values) -> np.ndarray:
    arr = np.zeros(24, dtype=float)
    if values is None:
        return arr
    for i, v in enumerate(values[:24]):
        try:
            f = float(v)
            if not np.isfinite(f):
                f = 0.0
            arr[i] = f
        except (TypeError, ValueError):
            arr[i] = 0.0
    return arr


def _average_panels(panels_per_market: list[list[dict]]) -> list[dict]:
    """Element-wise average across markets, day by day."""
    if not panels_per_market:
        return []
    by_day: dict[str, list[dict]] = {}
    for market_panels in panels_per_market:
        if not market_panels:
            continue
        for panel in market_panels:
            dt = panel.get("day_type")
            if not dt:
                continue
            by_day.setdefault(dt, []).append(panel)

    out: list[dict] = []
    for dt in ("Weekday", "Saturday", "Sunday"):
        bucket = by_day.get(dt, [])
        if not bucket:
            continue
        our_stack = np.vstack([_safe_24h(p.get("our_profile"))      for p in bucket])
        ref_stack = np.vstack([_safe_24h(p.get("reference_profile")) for p in bucket])
        our_avg = our_stack.mean(axis=0)
        ref_avg = ref_stack.mean(axis=0)
        err = our_avg - ref_avg
        ref_mean = float(ref_avg.mean())
        metrics = {
            "rmse":        float(np.sqrt(np.mean(err**2))),
            "mae":         float(np.mean(np.abs(err))),
            "max_abs_err": float(np.max(np.abs(err))),
            "bias":        float(np.mean(err)),
        }
        if ref_mean > 0:
            metrics["cv_rmse_pct"]  = metrics["rmse"] / ref_mean * 100
            metrics["nmae_pct"]     = metrics["mae"]  / ref_mean * 100
            metrics["rel_bias_pct"] = metrics["bias"] / ref_mean * 100
        else:
            metrics["cv_rmse_pct"] = metrics["nmae_pct"] = metrics["rel_bias_pct"] = None
        out.append({
            "day_type":          dt,
            "our_profile":       our_avg.tolist(),
            "reference_profile": ref_avg.tolist(),
            "metrics":           metrics,
        })
    return out


def _render_panel(panel: dict, header_caption: str):
    """One day-type column: chart on top, a clean metrics table below."""
    dt = panel["day_type"]
    st.markdown(f"##### {dt}")
    if header_caption:
        st.caption(header_caption)

    # Chart title kept to just the day-type → never gets truncated
    st.plotly_chart(
        arera_single_chart(
            panel["our_profile"], panel["reference_profile"],
            title_suffix=f" — {dt}",
        ),
        use_container_width=True,
    )

    m = panel["metrics"]

    def _fmt(v, sign=False, decimals=4):
        if v is None:
            return "N/A"
        return (f"{v:+.{decimals}f}" if sign else f"{v:.{decimals}f}")

    rows = [
        ("RMSE [kWh]",         _fmt(m.get("rmse"))),
        ("MAE [kWh]",          _fmt(m.get("mae"))),
        ("Max Abs Err [kWh]",  _fmt(m.get("max_abs_err"))),
        ("Bias [kWh]",         _fmt(m.get("bias"), sign=True)),
        ("CV-RMSE [%]",        _fmt(m.get("cv_rmse_pct"),  decimals=2)),
        ("Normalized MAE [%]", _fmt(m.get("nmae_pct"),     decimals=2)),
        ("Relative Bias [%]",  _fmt(m.get("rel_bias_pct"), sign=True, decimals=2)),
    ]
    import pandas as pd
    st.dataframe(
        pd.DataFrame(rows, columns=["Metric", "Value"]),
        hide_index=True, use_container_width=True,
    )


def render():
    title_c, status_c = st.columns([4, 2])
    with title_c:
        st.subheader("ARERA Profile Comparison")
        st.caption(
            "Compare the aggregated PoliTo hourly load profile against the "
            "official ARERA reference, **residenza-by-residenza** (legacy "
            "dashboard parity). Pick *Residente* or *Non Residente* and the "
            "POD set is restricted to the matching `ateco_l2` automatically "
            "(`DO.02` = Residente, `DO.01` = Non Residente). The three "
            "day-type profiles (Weekday, Saturday, Sunday) are shown together."
        )
    status_slot = status_c.empty()

    try:
        keys = api.arera_keys()
        codes_l2 = api.ateco_codes(level=2)
        descs    = api.ateco_descriptions()
    except api.BackendError as e:
        st.error(f"ARERA discovery failed: {e}.")
        return

    power_classes = sorted({k["power_class"] for k in keys})
    if not power_classes:
        st.warning("No ARERA reference data loaded.")
        return

    col1, col2 = st.columns(2)
    with col1:
        power_class = st.selectbox("Power Class", power_classes, key="arera_pc")
    pc_keys = [k for k in keys if k["power_class"] == power_class]

    with col2:
        markets_available = sorted({k["market"] for k in pc_keys})
        markets_options   = [ALL_MARKETS_LABEL] + markets_available
        market = st.selectbox("Market Type", markets_options, key="arera_mk")
    mk_keys = pc_keys if market == ALL_MARKETS_LABEL \
              else [k for k in pc_keys if k["market"] == market]

    col3, col4 = st.columns(2)
    with col3:
        # Legacy parity: ARERA comparisons are *always* residenza-consistent.
        # The original dashboard splits the POD population by FDESC-derived
        # ATECO_L2 (DO.R / DO.NR) and compares each group against the matching
        # ARERA reference. Mixing both groups against the "Tutti" reference
        # produces apples-to-oranges aggregates, so we hide that option here.
        candidate_res = sorted({k["residenza"] for k in mk_keys})
        residenze = [r for r in candidate_res if r != "Tutti"]
        if not residenze:
            st.warning("No residenza-specific reference profiles are loaded "
                       "for this selection; the comparison is paused.")
            return
        residenza = st.selectbox(
            "Residence Type", residenze, key="arera_res",
            help="ARERA distinguishes Residente from Non Residente domestics. "
                 "We mirror the legacy dashboard and split your PODs by the "
                 "matching ATECO L2 code (Residente ⇒ `DO.02`, "
                 "Non Residente ⇒ `DO.01`).",
        )
    with col4:
        provinces = sorted({k["province"] for k in mk_keys
                            if k["residenza"] == residenza})
        province = st.selectbox("Province", provinces, key="arera_prov")

    col5, col6 = st.columns(2)
    with col5:
        month_pick = st.selectbox(
            "Month", options=list(range(0, 13)),
            format_func=lambda m: MONTH_LABELS[m],
            key="arera_month",
        )
    with col6:
        min_months = st.number_input("Minimum Months of Data per POD",
                                      0, 36, 0, key="arera_mm",
                                      help="0 ⇒ no coverage filter (every "
                                           "POD matching the ATECO selection "
                                           "contributes). Raise this only to "
                                           "drop poorly-covered PODs.")

    # ── Derive the ATECO L2 filter from the residenza choice ────────────────
    # Lorenzo's Postgres ingestion preserved the pre-FDESC ATECO L2 codes
    # (DO.01 / DO.02) rather than the FDESC-derived DO.R / DO.NR labels the
    # legacy dashboard works with. The two map directly though:
    #     DO.02 ↔ Residente
    #     DO.01 ↔ Non Residente
    # So the residenza dropdown drives the L2 filter directly — no separate
    # multiselect is needed (and exposing one would only let the user
    # accidentally rebuild a mixed-residenza set).
    RESIDENZA_TO_L2 = {"Residente": "DO.02", "Non Residente": "DO.01"}
    target_l2 = RESIDENZA_TO_L2.get(residenza)
    if target_l2 is None:
        st.error(f"Unknown residenza '{residenza}' — no L2 mapping defined.")
        return

    pod_filter = {
        "ateco_l2":   [target_l2],
        "min_months": int(min_months),
        "tipologia":  "AP",
    }
    try:
        preview = api.pod_set_preview(pod_filter)
        n_pods_ateco = int(preview["n_pods"])
        st.info(f"**{fmt_int(n_pods_ateco)} PODs** match `ateco_l2 = "
                f"{target_l2}` ({residenza}), before the ARERA power-class "
                f"bucket below.")
    except api.BackendError as e:
        st.error(f"Preview failed: {e}.")
        return

    if st.button("▶ Run Comparison", type="primary", key="arera_run"):
        market_list = markets_available if market == ALL_MARKETS_LABEL else [market]
        panels_per_market: list[list[dict]] = []
        monthly_per_market: list[list[dict]] = []
        n_pods = 0
        with status_slot.status("Aggregating profiles in PostgreSQL & comparing.",
                                 expanded=False) as s:
            for mk in market_list:
                try:
                    res = api.compare_arera_all({
                        "filter":      pod_filter,
                        "power_class": power_class,
                        "market":      mk,
                        "residenza":   residenza,
                        "month":       int(month_pick),
                        "province":    province,
                    })
                    panels_per_market.append(res["panels"])
                    monthly_per_market.append(res.get("monthly") or [])
                    n_pods = max(n_pods, res["n_pods"])
                except api.BackendError as e:
                    st.warning(f"Market '{mk}': {e}.")
            s.update(label="Comparison complete.", state="complete")

        if not panels_per_market:
            st.error("No comparison could be produced for the current selection.")
            return

        panels = (_average_panels(panels_per_market)
                  if len(panels_per_market) > 1 else panels_per_market[0])
        # ── Average monthly breakdowns across markets too (same recipe) ────
        monthly: list[dict] = []
        if any(monthly_per_market):
            by_month: dict[int, list[list[dict]]] = {}
            for mk_bundles in monthly_per_market:
                for bundle in (mk_bundles or []):
                    m = int(bundle["month_idx"])
                    by_month.setdefault(m, []).append(bundle["panels"])
            for m in sorted(by_month):
                merged = (_average_panels(by_month[m])
                          if len(by_month[m]) > 1 else by_month[m][0])
                monthly.append({"month_idx": m, "panels": merged})

        st.session_state["arera_last_result"] = {
            "panels":        panels,
            "monthly":       monthly,
            "n_pods":        n_pods,
            "n_pods_ateco":  n_pods_ateco,
            "market_list":   market_list,
            "power_class":   power_class,
            "market":        market,
            "residenza":     residenza,
            "month_pick":    int(month_pick),
        }

    # ── Render the last stored result ─────────────────────────────────────
    stored = st.session_state.get("arera_last_result")
    if not stored:
        return

    panels         = stored["panels"]
    n_pods         = stored["n_pods"]
    n_pods_ateco   = stored.get("n_pods_ateco")
    market_list    = stored["market_list"]
    power_class    = stored["power_class"]
    market         = stored["market"]
    residenza      = stored["residenza"]
    month_pick     = stored["month_pick"]

    st.metric("PODs in the Aggregate:", fmt_int(n_pods))
    # Surface the power-class filter effect explicitly. The buckets here use
    # the legacy PoliTo dashboard convention (lo ≤ x < hi), so a 3.0 kW
    # residential POD belongs to "3-4.5 kW" rather than to "1.5-3 kW".
    if n_pods_ateco and n_pods_ateco > n_pods:
        kept_pct = n_pods / n_pods_ateco * 100
        st.caption(
            f"Of the **{fmt_int(n_pods_ateco)}** ATECO-matching PODs, "
            f"**{fmt_int(n_pods)}** ({kept_pct:.1f} %) have a contractual "
            f"power in the **{power_class}** bucket. The bucket boundaries "
            f"are left-closed (`lo ≤ x < hi`), so e.g. an exactly 3.0 kW "
            f"contract sits in *3–4.5 kW*, not in *1.5–3 kW*."
        )
    if len(market_list) > 1:
        st.caption(f"Averaged across {len(market_list)} markets: "
                   f"{', '.join(market_list)}.")

    header_caption = (f"{power_class} | {market} | {residenza} | "
                      f"{MONTH_LABELS[month_pick]}.")

    if not panels:
        st.warning("No day-type panels available for the selection.")
        return

    panel_cols = st.columns(len(panels))
    for col, panel in zip(panel_cols, panels):
        with col:
            _render_panel(panel, header_caption)

    import json
    st.download_button(
        "⬇ Download Full Result (JSON)",
        data=json.dumps({"panels": panels, "n_pods": n_pods}, indent=2).encode("utf-8"),
        file_name=f"arera_compare_{power_class}_{market}_{residenza}_"
                  f"m{month_pick}.json".replace(" ", "_"),
        mime="application/json",
    )

    # ── Full Monthly Overview (only when the request was for the annual) ──
    monthly_bundles = stored.get("monthly") or []
    if month_pick == 0 and monthly_bundles:
        from charts import arera_monthly_grid_chart
        st.markdown("#### Full Monthly Overview")
        st.caption(
            "Same comparison, broken down by calendar month. Open a day-type "
            "expander to see the 12-panel grid."
        )
        # Restructure: {day_type: {month_idx: profile}} for both ours/ref
        for day_type in ("Weekday", "Saturday", "Sunday"):
            our_by_month: dict[int, list[float]] = {}
            ref_by_month: dict[int, list[float]] = {}
            for bundle in monthly_bundles:
                m = int(bundle["month_idx"])
                # Each bundle.panels has 3 entries, one per day_type
                for p in bundle["panels"]:
                    if p.get("day_type") != day_type:
                        continue
                    our_by_month[m] = p.get("our_profile") or []
                    ref_by_month[m] = p.get("reference_profile") or []
            with st.expander(f"{day_type} — monthly breakdown", expanded=False):
                st.plotly_chart(
                    arera_monthly_grid_chart(
                        our_by_month, ref_by_month, day_type=day_type,
                    ),
                    use_container_width=True,
                    key=f"arera_monthly_{day_type}_{power_class}_"
                        f"{market}_{residenza}".replace(" ", "_"),
                )
