"""Reusable ATECO L1 → L2 → L3 cascading checkbox filter.

Same UI in every tab that needs to subset PODs by ATECO codes:
    * three side-by-side columns (Section / Division / Class)
    * Select All / Deselect All per column
    * children appear only after a parent is ticked
"""

from __future__ import annotations

import streamlit as st

from helpers import fmt_label


def _select_all_buttons(key_prefix: str, level: str, options: list[str]):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Select All", key=f"{key_prefix}_sa_{level}",
                      use_container_width=True):
            for o in options:
                st.session_state[f"{key_prefix}_cb_{level}_{o}"] = True
            st.rerun()
    with c2:
        if st.button("Deselect All", key=f"{key_prefix}_da_{level}",
                      use_container_width=True):
            for o in options:
                st.session_state[f"{key_prefix}_cb_{level}_{o}"] = False
            st.rerun()


def _checkbox_list(
    key_prefix: str, level: str, codes: list[str], descs: dict[str, str]
) -> list[str]:
    chosen: list[str] = []
    with st.container(height=320):
        for code in codes:
            label = fmt_label(code, descs, max_len=50)
            key   = f"{key_prefix}_cb_{level}_{code}"
            st.session_state.setdefault(key, False)
            if st.checkbox(label, key=key):
                chosen.append(code)
    return chosen


def render(
    key_prefix:    str,
    codes_l1:      list[str],
    codes_l2:      list[str],
    codes_l3:      list[str],
    descs:         dict[str, str],
    restrict_l1:   list[str] | None = None,
    exclude_l1:    list[str] | None = None,
    header_l1:     str = "Level 1 — Section",
    header_l2:     str = "Level 2 — Division",
    header_l3:     str = "Level 3 — Class",
) -> tuple[list[str], list[str], list[str]]:
    """Render the 3-column cascading filter and return the selected codes.

    Args:
        key_prefix: unique prefix to keep session-state keys disjoint across tabs.
        codes_lN:   available codes at each level (from the API).
        descs:      ATECO code → description dict.
        restrict_l1: if set, only these L1 codes appear (e.g. ['DO']).
        exclude_l1: if set, these L1 codes are filtered out (e.g. ['DO']).
    """
    # Apply restrictions on L1
    l1_options = sorted(codes_l1)
    if restrict_l1 is not None:
        l1_options = [c for c in l1_options if c in set(restrict_l1)]
    if exclude_l1 is not None:
        l1_options = [c for c in l1_options if c not in set(exclude_l1)]

    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown(f"**{header_l1}**")
        _select_all_buttons(key_prefix, "l1", l1_options)
        selected_l1 = _checkbox_list(key_prefix, "l1", l1_options, descs)

    with col_l2:
        st.markdown(f"**{header_l2}**")
        if not selected_l1:
            st.caption("← Select at least one Level 1 code to see Level 2 options.")
            selected_l2: list[str] = []
        else:
            l2_options = sorted(
                c for c in codes_l2
                if any(c.startswith(l1 + ".") or c == l1 for l1 in selected_l1)
            )
            _select_all_buttons(key_prefix, "l2", l2_options)
            selected_l2 = _checkbox_list(key_prefix, "l2", l2_options, descs)

    with col_l3:
        st.markdown(f"**{header_l3}**")
        if not selected_l2:
            st.caption("← Select at least one Level 2 code to see Level 3 options.")
            selected_l3: list[str] = []
        else:
            l3_options = sorted(
                c for c in codes_l3
                if any(c.startswith(l2 + ".") for l2 in selected_l2)
            )
            _select_all_buttons(key_prefix, "l3", l3_options)
            selected_l3 = _checkbox_list(key_prefix, "l3", l3_options, descs)

    return selected_l1, selected_l2, selected_l3
