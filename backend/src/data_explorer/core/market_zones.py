"""Italian electricity market zones (zone di mercato Terna, post-2021).

Source: Terna "Codice di Rete" Annex A.24 — geographical bidding zones used by
GME PUN and Terna's MSD service. Calabria became its own zone on 2021-01-01
(EU CACM separation), so we use the 7-zone scheme.

Region names come from Nominatim's `address.state` field and may include
the bilingual form for Trentino-Alto Adige and Valle d'Aosta. We normalise
on lookup so all the common spellings collapse to the same key.
"""

from __future__ import annotations

# ── Canonical zone → list of regions ─────────────────────────────────────────
ZONE_REGIONS: dict[str, list[str]] = {
    "NORD": [
        "Lombardia", "Piemonte", "Liguria", "Valle d'Aosta",
        "Trentino-Alto Adige", "Veneto", "Friuli-Venezia Giulia",
        "Emilia-Romagna",
    ],
    "CNOR": ["Toscana", "Umbria", "Marche"],
    "CSUD": ["Lazio", "Abruzzo", "Molise", "Campania"],
    "SUD":  ["Puglia", "Basilicata"],
    "CALA": ["Calabria"],
    "SICI": ["Sicilia"],
    "SARD": ["Sardegna"],
}

ZONE_LABELS: dict[str, str] = {
    "NORD": "Nord",
    "CNOR": "Centro-Nord",
    "CSUD": "Centro-Sud",
    "SUD":  "Sud",
    "CALA": "Calabria",
    "SICI": "Sicilia",
    "SARD": "Sardegna",
}

# Canonical order for UI display (top-to-bottom geographic).
ZONE_ORDER: list[str] = ["NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD"]


def _normalise_region(raw: str | None) -> str | None:
    """Return the canonical region name for a Nominatim-style string, or None."""
    if not raw:
        return None
    r = raw.strip()
    # Nominatim returns the bilingual forms; collapse to the Italian name.
    r = r.replace("/Südtirol", "").replace(" – ", "-").replace("–", "-")
    r = r.replace("Friuli Venezia Giulia", "Friuli-Venezia Giulia")
    r = r.replace("Trentino Alto Adige", "Trentino-Alto Adige")
    r = r.replace("Valle d’Aosta", "Valle d'Aosta")
    r = r.replace("Valle d Aosta", "Valle d'Aosta")
    r = r.replace("Aosta Valley", "Valle d'Aosta")
    if r in _REGION_TO_ZONE:
        return r
    # Case-insensitive fallback.
    for canon in _REGION_TO_ZONE:
        if canon.lower() == r.lower():
            return canon
    return None


# Reverse index — built once, used by SQL filter and tick-availability logic.
_REGION_TO_ZONE: dict[str, str] = {
    region: zone
    for zone, regions in ZONE_REGIONS.items()
    for region in regions
}


def region_to_zone(region: str | None) -> str | None:
    """Map a free-form region name to its zone code, or None when unknown."""
    canon = _normalise_region(region)
    return _REGION_TO_ZONE.get(canon) if canon else None


def regions_for_zone(zone: str) -> list[str]:
    """List of canonical region names belonging to the given zone."""
    return list(ZONE_REGIONS.get(zone, []))
