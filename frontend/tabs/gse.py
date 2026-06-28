"""GSE Profile Comparison tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api
from charts import fmt_int, gse_monthly_chart
from components import ateco_filter
from helpers import fmt_label


def _is_domestic_code(code: str) -> bool:
    if not code or len(code) < 2:
        return False
    return code[1] == "D"


def render():
    title_c, status_c = st.columns([4, 2])
    with title_c:
        st.subheader("GSE Profile Comparison")
        st.caption(
            "Compare the aggregated PoliTo profile against an official GSE 2025 ."
            "reference column. Profiles ending in **M** (monorario) are normalized "
            "as % of monthly consumption; profiles ending in **F** (in fasce) are "
            "normalized as % of monthly band consumption."
        )
    status_slot = status_c.empty()

    with st.expander("GSE Reference Profiles Legend"):
        st.markdown("""
| Code     | Description |
|----------|-------------|
| **PDMM** | Domestic Monorario (single tariff) — % of Monthly Consumption [%] |
| **PDMF** | Domestic Time-of-Use (fasce) — % of Monthly Band Consumption [%] |
| **PAUM** | Other Uses Monorario [%] |
| **PAUF** | Other Uses Time-of-Use [%] |
| **PIRM** | Public Lighting Monorario [%] |
| **PIRF** | Public Lighting Time-of-Use [%] |
| **PACM** | Air Conditioning Monorario [%] |
| **PACF** | Air Conditioning Time-of-Use [%] |
| **MDMM** | Domestic Mode-Medium Monorario [%] |
| **MDMF** | Domestic Mode-Medium Time-of-Use [%] |
| **MAUM** | Other Uses Mode-Medium Monorario [%] |
| **MAUF** | Other Uses Mode-Medium Time-of-Use [%] |
""")

    try:
        profile_codes = api.gse_profile_codes()
        codes_l1      = api.ateco_codes(level=1)
        codes_l2      = api.ateco_codes(level=2)
        codes_l3      = api.ateco_codes(level=3)
        descs         = api.ateco_descriptions()
    except api.BackendError as e:
        st.error(f"Backend unavailable: {e}.")
        return

    col_pc, _ = st.columns([1, 2])
    with col_pc:
        profile_code = st.selectbox(
            "GSE Profile Code", options=profile_codes, index=0, key="gse_pc",
        )
    is_dom = _is_domestic_code(profile_code)
    if is_dom:
        st.info("Selected profile is **Domestic** → ATECO L1 restricted to `DO`. "
                "All household codes are selectable below.")
    else:
        st.info("Selected profile is **Non-Domestic** → ATECO L1 restricted to "
                "codes other than `DO`.")

    # ── ATECO cascading filter ────────────────────────────────────────────
    st.markdown("#### ATECO Level Selection")
    selected_l1, selected_l2, selected_l3 = ateco_filter.render(
        key_prefix="gse",
        codes_l1=codes_l1, codes_l2=codes_l2, codes_l3=codes_l3,
        descs=descs,
        restrict_l1=["DO"] if is_dom else None,
        exclude_l1=None     if is_dom else ["DO"],
    )

    pod_filter = {
        "ateco_l1":   selected_l1 or (["DO"] if is_dom else None),
        "ateco_l2":   selected_l2 or None,
        "ateco_l3":   selected_l3 or None,
        "min_months": None,
        "tipologia":  "AP",
    }

    col_a, col_b = st.columns(2)
    with col_a:
        min_months = st.number_input("Min Months", 0, 36, 0, key="gse_mm",
            help="0 ⇒ no coverage filter (every POD matching the ATECO "
                 "selection contributes). Raise this only if you want to "
                 "exclude poorly-covered PODs from the aggregate.")
    with col_b:
        # Legacy parity: `compute_our_normalized_profiles` / `compute_our_
        # fascia_profiles` in the original dashboard never filter by
        # day-of-week — the monorario normalization aggregates all days of
        # the month, and the in-fascia normalization encodes the
        # weekday/weekend distinction inside the F1/F2/F3 band definitions
        # rather than as a row-level filter. We hardcode `all` here to match.
        dayset = "all"
        st.text_input("Day Set", value="All Days (legacy parity)",
                      disabled=True, key="gse_ds_display",
                      help="GSE monorario and fascia profiles are computed "
                           "over the full month in the legacy dashboard. "
                           "We mirror that behaviour exactly: every day of "
                           "the month contributes to both numerator and "
                           "denominator.")

    pod_filter["min_months"] = int(min_months)

    try:
        preview = api.pod_set_preview(pod_filter)
        st.info(f"**{fmt_int(preview['n_pods'])} PODs** match the current filter.")
    except api.BackendError as e:
        st.error(f"Preview failed: {e}.")
        return

    if st.button("▶ Run Comparison", type="primary", key="gse_run"):
        with status_slot.status("Aggregating profile in PostgreSQL & comparing.",
                                 expanded=False) as s:
            try:
                result = api.compare_gse({
                    "filter":       pod_filter,
                    "profile_code": profile_code,
                    "dayset":       dayset,
                })
                s.update(label="Comparison complete.", state="complete")
                st.session_state["gse_last_result"] = {
                    "result":       result,
                    "profile_code": profile_code,
                    "dayset":       dayset,
                }
            except api.BackendError as e:
                s.update(label=f"GSE comparison failed: {e}.", state="error")
                return

    # ── Render last stored result ─────────────────────────────────────────
    stored = st.session_state.get("gse_last_result")
    if not stored:
        return

    result       = stored["result"]
    profile_code = stored["profile_code"]
    dayset       = stored["dayset"]

    st.metric("PODs in the Aggregate:", fmt_int(result["n_pods"]))

    st.plotly_chart(
        gse_monthly_chart(
            {int(k): v for k, v in result["our_profile"].items()},
            {int(k): v for k, v in result["reference_profile"].items()},
            profile_code,
        ),
        use_container_width=True,
    )

    st.markdown("#### Comparison Metrics")
    metrics = pd.DataFrame(result["metrics"])
    if not metrics.empty:
        metrics = metrics.rename(columns={
            "month":       "Month",
            "rmse":        "RMSE [%]",
            "mae":         "MAE [%]",
            "bias":        "Bias [%]",
            "max_abs_err": "Max Abs Err [%]",
        })
        st.dataframe(
            metrics.style.format({
                "RMSE [%]":        "{:.3f}",
                "MAE [%]":         "{:.3f}",
                "Bias [%]":        "{:+.3f}",
                "Max Abs Err [%]": "{:.3f}",
            }),
            hide_index=True, use_container_width=True,
        )

    import json
    st.download_button(
        "⬇ Download Full Result (JSON)",
        data=json.dumps(result, indent=2).encode("utf-8"),
        file_name=f"gse_compare_{profile_code}_{dayset}.json",
        mime="application/json",
    )
