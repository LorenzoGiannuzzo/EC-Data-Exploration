"""Thin httpx wrapper over the FastAPI backend.

All HTTP calls in the frontend go through this module. Errors are surfaced
as exceptions with the API's `detail` message so each tab can ``st.error()``
them cleanly.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT     = httpx.Timeout(120.0, connect=5.0)


class BackendError(Exception):
    """Raised on any non-2xx response. The message is the API ``detail`` field."""


def _handle(resp: httpx.Response) -> Any:
    if resp.is_success:
        return resp.json()
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    raise BackendError(f"[{resp.status_code}] {detail}")


# ── Generic helpers ──────────────────────────────────────────────────────────
def get(path: str, **params) -> Any:
    with httpx.Client(timeout=TIMEOUT) as c:
        return _handle(c.get(f"{BACKEND_URL}{path}", params=params))


def post(path: str, payload: dict) -> Any:
    with httpx.Client(timeout=TIMEOUT) as c:
        return _handle(c.post(f"{BACKEND_URL}{path}", json=payload))


# ── Cached read-only endpoints ───────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def health() -> dict:
    return get("/health")


@st.cache_data(ttl=300, show_spinner=False)
def info() -> dict:
    return get("/info")


@st.cache_data(ttl=600, show_spinner=False)
def ateco_codes(level: int = 1) -> list[str]:
    return get("/metadata/ateco", level=level)["codes"]


@st.cache_data(ttl=3600, show_spinner=False)
def ateco_descriptions() -> dict[str, str]:
    return get("/metadata/ateco/descriptions")["descriptions"]


@st.cache_data(ttl=3600, show_spinner=False)
def tipologie() -> list[str]:
    return get("/metadata/tipologie")["tipologie"]


@st.cache_data(ttl=600, show_spinner=False)
def gse_profile_codes() -> list[str]:
    return get("/gse/profiles")["profile_codes"]


@st.cache_data(ttl=600, show_spinner=False)
def arera_power_classes() -> list[str]:
    return get("/arera/power-classes")["power_classes"]


@st.cache_data(ttl=600, show_spinner=False)
def arera_keys() -> list[dict]:
    return get("/arera/keys")["keys"]


# ── POD-set preview (used everywhere as the "n PODs selected" badge) ─────────
@st.cache_data(ttl=120, show_spinner=False)
def pod_set_preview(pod_filter: dict) -> dict:
    return post("/metadata/pod-set", pod_filter)


# ── Heavy analytical endpoints — not cached (params often unique) ────────────
def run_clustering(payload: dict) -> dict:
    return post("/clustering/run", payload)


def run_clustering_by_level(payload: dict) -> dict:
    return post("/clustering/run-by-level", payload)


def detect_outliers(payload: dict) -> dict:
    return post("/clustering/outliers", payload)


# ── Overview ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def overview_ateco_coverage() -> dict:
    return get("/overview/ateco-coverage")


@st.cache_data(ttl=600, show_spinner=False)
def overview_power_class_distribution() -> dict:
    return get("/overview/power-class-distribution")


@st.cache_data(ttl=600, show_spinner=False)
def overview_consumption_distribution(tipologia: str = "AP") -> dict:
    return get("/overview/consumption-distribution", tipologia=tipologia)

def overview_geography() -> dict:
    # NOT cached: calling this endpoint triggers the background geocoding
    # task on the backend, and the map must reflect progress on each refresh.
    return get("/overview/geography")

def compare_gse(payload: dict) -> dict:
    return post("/gse/compare", payload)


def compare_arera(payload: dict) -> dict:
    return post("/arera/compare", payload)


def compare_arera_all(payload: dict) -> dict:
    return post("/arera/compare-all-day-types", payload)
