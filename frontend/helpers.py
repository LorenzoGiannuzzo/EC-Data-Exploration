"""Helpers shared by tabs."""

from __future__ import annotations


def ateco_desc(code: str, descs: dict[str, str]) -> str:
    """Hierarchical ATECO description lookup.

    Many L3 codes don't have a description in the official ATECO lookup. Walk
    up the hierarchy ('47.11.10' → '47.11' → '47') until we find a non-empty
    description, so the user never sees an unlabelled checkbox.
    """
    if not code:
        return ""
    c = code
    while c:
        v = descs.get(c)
        if v:
            return v
        if "." in c:
            c = c.rsplit(".", 1)[0]
        else:
            break
    return ""


def fmt_label(code: str, descs: dict[str, str], max_len: int = 60) -> str:
    """Compose a 'CODE — description' label, truncated."""
    d = ateco_desc(code, descs)
    return f"{code} — {d[:max_len]}" if d else code
