"""Reading the monthly folders.

The source layout is one folder per month, named <3-letter Italian month><2-digit
year> in mixed case (ago24, Ago25, Apr24, ...), each holding:

    Metadati POD <mesYY>.xlsx     POD, CCATETE, D_49DES, FDESC, TATE3DES
    misure_<anything>.csv         POD, DataMisura, Q1..Q96, PotenzaContrattuale

Nothing about the encoding, the field separator or the decimal mark is assumed:
the files come from a distributor's export and all three have been seen to vary.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

Q_COLS = [f"Q{i}" for i in range(1, 97)]

MONTHS_IT = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
}
DIR_RE = re.compile(r"^([a-zA-Z]{3})(\d{2})$")


# ── folder scan ──────────────────────────────────────────────────────────────
def scan_data_dir(root: Path) -> pd.DataFrame:
    """Return one row per monthly folder found, with its parsed month and year."""
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = DIR_RE.match(d.name)
        if not m:
            continue                              # _cache/ and anything else
        mon = m.group(1).lower()
        if mon not in MONTHS_IT:
            continue
        meta = _find_one(d, ["Metadati*.xlsx", "metadati*.xlsx", "*.xlsx"])
        meas = _find_one(d, ["misure*.csv", "Misure*.csv", "*.csv"])
        rows.append({
            "folder": d.name,
            "path": d,
            "month": MONTHS_IT[mon],
            "year": 2000 + int(m.group(2)),
            "meta_file": meta,
            "meas_file": meas,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No monthly folder matching <mesYY> under {root}")
    return df.sort_values(["year", "month"]).reset_index(drop=True)


def _find_one(folder: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


# ── robust readers ───────────────────────────────────────────────────────────
def _sniff_csv(path: Path) -> tuple[str, str, str]:
    """Guess (encoding, separator, decimal) from the first two lines."""
    head = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, encoding=enc) as f:
                head = [f.readline() for _ in range(2)]
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if head is None:
        raise UnicodeDecodeError("unknown", b"", 0, 1, f"cannot decode {path}")

    header = head[0]
    sep = max([";", ",", "\t", "|"], key=header.count)
    body = head[1] if len(head) > 1 and head[1] else ""
    # If the separator is ';' a comma can only be the decimal mark.
    decimal = "," if (sep == ";" and "," in body) else "."
    return encoding, sep, decimal


def read_measurements(path: Path, date_col: str = "DataMisura",
                      date_format: str | None = "%d/%m/%Y") -> pd.DataFrame:
    """Read one monthly measurement CSV. Q columns come back as float32."""
    encoding, sep, decimal = _sniff_csv(path)
    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding,
                     low_memory=False)
    df = _normalise_columns(df)

    qs = _q_columns(df)
    if len(qs) != 96:
        raise ValueError(f"{path.name}: found {len(qs)} quarter-hour columns, expected 96")

    df[qs] = df[qs].apply(pd.to_numeric, errors="coerce").astype("float32")

    dcol = _match(df, date_col)
    if dcol is None:
        raise ValueError(f"{path.name}: no date column matching '{date_col}'")
    df[dcol] = pd.to_datetime(df[dcol], format=date_format, errors="coerce")
    if df[dcol].isna().all():                     # the declared format did not apply
        df[dcol] = pd.to_datetime(df[dcol], dayfirst=True, errors="coerce")
    df = df.rename(columns={dcol: "date"})

    pcol = _match(df, "POD")
    if pcol is None:
        raise ValueError(f"{path.name}: no POD column")
    df = df.rename(columns={pcol: "pod"})
    df["pod"] = df["pod"].astype(str).str.strip()

    return df


def read_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = _normalise_columns(df)
    pcol = _match(df, "POD")
    if pcol is None:
        raise ValueError(f"{path.name}: no POD column")
    df = df.rename(columns={pcol: "pod"})
    df["pod"] = df["pod"].astype(str).str.strip()
    return df


# ── column helpers ───────────────────────────────────────────────────────────
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _match(df: pd.DataFrame, name: str) -> str | None:
    """Case-insensitive column lookup."""
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower())


def _q_columns(df: pd.DataFrame) -> list[str]:
    """The 96 quarter-hour columns, in numeric order, however they are spelled."""
    found = {}
    for c in df.columns:
        m = re.fullmatch(r"[Qq]\s*(\d{1,2})", str(c).strip())
        if m:
            found[int(m.group(1))] = c
    return [found[i] for i in sorted(found)]


# ── ATECO hierarchy ──────────────────────────────────────────────────────────
def split_ateco(code: str | float) -> tuple[str | None, str | None, str | None]:
    """Split CCATETE into its three levels.

    Beware of the naming: these follow the source column, not NACE.
        l1 = 2 chars  -> NACE Division   ('47', 'DO')
        l2 = 4 chars  -> NACE Class      ('47.11', 'DO.01')
        l3 = 6 chars  -> NACE Subcategory('47.11.10')
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None, None, None
    parts = str(code).strip().split(".")
    l1 = parts[0] if parts and parts[0] else None
    l2 = ".".join(parts[:2]) if len(parts) >= 2 else None
    l3 = ".".join(parts[:3]) if len(parts) >= 3 else None
    return l1, l2, l3


# ── year assembly ────────────────────────────────────────────────────────────
def load_year(root: Path, year: int, meas_date_col: str = "DataMisura",
              date_format: str | None = "%d/%m/%Y",
              verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate every monthly folder of `year` into one measurement frame
    plus one metadata frame (last observation per POD wins)."""
    idx = scan_data_dir(root)
    idx = idx[idx["year"] == year]
    if idx.empty:
        available = sorted(scan_data_dir(root)["year"].unique())
        raise ValueError(f"No folder for year {year}. Available: {available}")

    meas_parts, meta_parts = [], []
    for _, r in idx.iterrows():
        if r["meas_file"] is not None:
            m = read_measurements(r["meas_file"], meas_date_col, date_format)
            meas_parts.append(m)
            if verbose:
                print(f"    {r['folder']:8s} {len(m):>8,} rows   {m['pod'].nunique():>6,} PODs")
        if r["meta_file"] is not None:
            meta_parts.append(read_metadata(r["meta_file"]))

    meas = pd.concat(meas_parts, ignore_index=True)
    meta = pd.concat(meta_parts, ignore_index=True).drop_duplicates("pod", keep="last")

    meas = meas[meas["date"].dt.year == year].reset_index(drop=True)
    return meas, meta
