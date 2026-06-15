"""Reusable contractual-power range filter.

Mirrors the legacy dashboard's sidebar "Contractual Power" filter: the same
11 bins (pd.cut, right-closed) presented as a multiselect list of ranges,
plus an "include PODs with missing power" checkbox.

Usage:
    from components.power_filter import power_filter
    power_ranges, include_missing = power_filter(key_prefix="cl")
    pod_filter["power_ranges"]          = power_ranges        # None = off
    pod_filter["include_missing_power"] = include_missing
"""

from __future__ import annotations

import streamlit as st

# Legacy POTCONTR_BINS / POTCONTR_BIN_LABELS — (label, min_kw, max_kw).
# A POD matches a range when min_kw < d_pot1 <= max_kw (max None = open).
POWER_RANGES: list[tuple[str, float, float | None]] = [
    ("≤1 kW",        0.0,   1.0),
    ("1–1.5 kW",     1.0,   1.5),
    ("1.5–2 kW",     1.5,   2.0),
    ("2–3 kW",       2.0,   3.0),
    ("3–6 kW",       3.0,   6.0),
    ("6–10 kW",      6.0,  10.0),
    ("10–16.5 kW",  10.0,  16.5),
    ("16.5–33 kW",  16.5,  33.0),
    ("33–55 kW",    33.0,  55.0),
    ("55–110 kW",   55.0, 110.0),
    (">110 kW",    110.0,  None),
]
_LABEL_TO_RANGE = {lbl: (lo, hi) for lbl, lo, hi in POWER_RANGES}


def power_filter(key_prefix: str) -> tuple[list[list[float | None]] | None, bool]:
    """Render the filter inside an expander; return ``(ranges, include_missing)``.

    ``ranges`` is None when the filter is disabled or no range is selected
    (= no power filtering), otherwise a list of ``[min_kw, max_kw]`` pairs
    ready for the ``PodFilter.power_ranges`` API field.
    """
    with st.expander("Contractual Power Filter", expanded=False):
        enabled = st.checkbox(
            "Enable power filter", value=False, key=f"{key_prefix}_pw_en",
            help="Restrict the POD set to the selected contractual-power "
                 "ranges (pod_metadata.d_pot1, kW).",
        )
        if not enabled:
            return None, False

        _key = f"{key_prefix}_pw_sel"
        all_labels = [lbl for lbl, _, _ in POWER_RANGES]
        # Canonical pattern: initialise the key once, let the buttons mutate
        # it BEFORE the widget renders, and never pass `default=` alongside a
        # session-state-managed key (raises on some Streamlit versions).
        if _key not in st.session_state:
            st.session_state[_key] = list(all_labels)

        c_a, c_b = st.columns([1, 1])
        with c_a:
            if st.button("Select All", key=f"{key_prefix}_pw_all",
                         use_container_width=True):
                st.session_state[_key] = list(all_labels)
        with c_b:
            if st.button("Deselect All", key=f"{key_prefix}_pw_none",
                         use_container_width=True):
                st.session_state[_key] = []

        selected = st.multiselect(
            "Power ranges [kW]",
            options=all_labels,
            key=_key,
        )
        include_missing = st.checkbox(
            "Include PODs with missing contractual power",
            value=False, key=f"{key_prefix}_pw_miss",
        )

        if not selected:
            st.warning("No range selected — the power filter would exclude "
                       "every POD. Select at least one range or disable it.")
            return None, False

        ranges = [list(_LABEL_TO_RANGE[lbl]) for lbl in selected]
        return ranges, include_missing
