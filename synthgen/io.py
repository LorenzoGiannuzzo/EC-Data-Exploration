"""Reading the source archive.

One folder per month, each holding a measurement CSV and a metadata workbook. The
measurement file is semicolon separated, decimal comma, latin dates, and a byte
order mark on the first header. A POD-day appears on several rows, one per
quantity recorded, and only the active withdrawn row carries the consumption
curve.

Three things this reader does that are easy to miss.

First, the meter constant. The `K` column is 1 for most points and 20, 25 or 40
for those metered through a current transformer, and the readings of the latter
are the raw meter counts, not the energy. Ignoring it understates the largest
users of the archive by that same factor, which is why it is applied here, per
row rather than per point, since a POD can change it inside a month.

Second, the padding. Q97 to Q100 exist so that every row has the same width; they
are not the extra quarter-hours of the October daylight saving day, which the
archive does not carry. Only Q1 to Q96 are read.

Third, duplicates. A handful of POD-day pairs appear twice, and the resolution has
to be deterministic or two runs of the same code disagree: the row with the most
valued quarter-hours wins, ties broken on the record identifier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def to_number(s: pd.Series) -> pd.Series:
    """Latin decimals to float. '3,000' is three, not three thousand.

    The result is always numpy backed float64 with NaN where the value is
    missing. Recent pandas returns a nullable array from `to_numeric`, and a
    nullable array refuses to become a plain float64 matrix while it still holds
    nulls, which is precisely what the caller asks of it next.
    """
    if s.dtype.kind in "fiu":
        return pd.Series(np.asarray(s, dtype="float64"), index=s.index)
    out = pd.to_numeric(
        s.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce")
    return pd.Series(out.to_numpy(dtype="float64", na_value=np.nan), index=s.index)


def q_columns(n: int = 96) -> list[str]:
    return [f"Q{i}" for i in range(1, n + 1)]


def month_folders(root: Path, meas_glob: str, meta_glob: str) -> list[dict]:
    """Every folder that holds both a measurement file and a metadata workbook."""
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        meas = sorted(d.glob(meas_glob))
        meta = sorted(d.glob(meta_glob))
        if not meas or not meta:
            continue
        out.append({"folder": d.name, "meas": meas[0], "meta": meta[0]})
    return out


def read_metadata(path: Path, cols: dict) -> pd.DataFrame:
    """The POD register of one month, reduced to the columns that are used."""
    raw = pd.read_excel(path, dtype=str)
    low = {c.lower(): c for c in raw.columns}

    def col(key: str) -> str | None:
        name = cols.get(key)
        return low.get(str(name).lower()) if name else None

    out = pd.DataFrame({"pod": raw[col("pod")].astype("string").str.strip()})
    ateco = col("ateco")
    out["ateco_raw"] = raw[ateco].astype("string").str.strip() if ateco else pd.NA
    power = col("power")
    out["power_meta_kW"] = to_number(raw[power]) if power else np.nan
    tariff = col("tariff")
    out["tariff"] = raw[tariff].astype("string").str.strip() if tariff else pd.NA
    desc = col("tariff_desc")
    out["tariff_desc"] = raw[desc].astype("string").str.strip() if desc else pd.NA
    off = col("deactivated")
    out["deactivated"] = (raw[off].astype("string").str.strip().fillna("") != "") \
        if off else False
    return out.drop_duplicates("pod")


def read_measures(path: Path, cfg_data: dict) -> tuple[pd.DataFrame, np.ndarray, set]:
    """One monthly measurement file.

    Returns the day frame, the (n_days, 96) matrix of quarter-hourly energy in
    kWh, and the set of PODs that also inject.
    """
    mc = cfg_data["meas_cols"]
    nq = int(cfg_data.get("n_quarters", 96))
    qs = q_columns(nq)
    keep = str(cfg_data.get("keep_kind", "AP"))
    inject = str(cfg_data.get("inject_kind", "AN"))
    kcol = str(cfg_data.get("k_col", "K"))

    usecols = ["Id", mc["pod"], mc["date"], mc["kind"], mc["power"], kcol] + qs
    df = pd.read_csv(path, sep=str(cfg_data.get("sep", ";")),
                     encoding=str(cfg_data.get("encoding", "utf-8-sig")),
                     usecols=lambda c: c in usecols, dtype=str,
                     low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    prosumers = set(df.loc[df[mc["kind"]] == inject, mc["pod"]].dropna().unique())
    ap = df[df[mc["kind"]] == keep].copy()
    if not len(ap):
        raise ValueError(f"{path.name}: no '{keep}' row found")

    ap["pod"] = ap[mc["pod"]].astype("string").str.strip()
    ap["date"] = pd.to_datetime(ap[mc["date"]],
                                format=str(cfg_data.get("date_format", "%d/%m/%Y")),
                                errors="coerce")
    ap["power_kW"] = to_number(ap[mc["power"]])
    ap["k"] = to_number(ap[kcol]) if kcol in ap.columns else 1.0
    ap = ap.dropna(subset=["pod", "date"])

    values = ap[qs].apply(to_number).to_numpy(dtype="float64", na_value=np.nan)

    #Lorenzo Giannuzzo: deterministic resolution of duplicate POD-days
    ap["_valued"] = np.isfinite(values).sum(axis=1)
    ap["_id"] = pd.to_numeric(ap.get("Id"), errors="coerce").fillna(-1)
    order = np.lexsort((ap["_id"].to_numpy(), ap["_valued"].to_numpy()))[::-1]
    ap = ap.iloc[order]
    values = values[order]
    first = ~ap.duplicated(["pod", "date"])
    n_dup = int((~first).sum())
    ap, values = ap[first.values], values[first.to_numpy()]

    #Lorenzo Giannuzzo: the meter constant, then the unit
    if bool(cfg_data.get("apply_k", True)):
        k = ap["k"].to_numpy(dtype="float64")
        k = np.where(np.isfinite(k) & (k > 0), k, 1.0)
        values *= k[:, None]
    if str(cfg_data.get("reading_unit", "Wh")).lower() == "wh":
        values /= 1000.0

    days = ap[["pod", "date", "power_kW", "k"]].reset_index(drop=True)
    days = days.sort_values(["pod", "date"], kind="stable")
    values = values[days.index.to_numpy()]
    days = days.reset_index(drop=True)
    days.attrs["n_duplicates_resolved"] = n_dup
    return days, values.astype("float32"), prosumers


def load_archive(root: Path, cfg_data: dict,
                 folders: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Every monthly folder, concatenated."""
    found = month_folders(root, str(cfg_data.get("meas_glob", "misure*.csv")),
                          str(cfg_data.get("meta_glob", "Metadati*.xls*")))
    if folders:
        wanted = {f.lower() for f in folders}
        found = [f for f in found if f["folder"].lower() in wanted]
    if not found:
        raise SystemExit(f"\n  no monthly folder found under {root}\n")

    day_parts, val_parts, meta_parts = [], [], []
    prosumers: set = set()
    n_dup = 0
    for f in found:
        d, v, pro = read_measures(f["meas"], cfg_data)
        m = read_metadata(f["meta"], cfg_data["meta_cols"])
        n_dup += int(d.attrs.get("n_duplicates_resolved", 0))
        prosumers |= pro
        day_parts.append(d)
        val_parts.append(v)
        meta_parts.append(m)
        print(f"    {f['folder']:<8s} {len(d):>7,} POD-days   "
              f"{d['pod'].nunique():>6,} PODs   "
              f"{int((d['k'] > 1).sum()):>4,} rows with K > 1")

    days = pd.concat(day_parts, ignore_index=True)
    #Lorenzo Giannuzzo: np.vstack holds the parts and the result alive at once, which over the whole
    # archive means twice about 1.8 GB. The result is allocated once, each month
    # is copied into it and released, so the peak is the result plus the largest
    # single month.
    n_rows = sum(len(v) for v in val_parts)
    values = np.empty((n_rows, val_parts[0].shape[1]), dtype="float32")
    at = 0
    while val_parts:
        part = val_parts.pop(0)
        values[at:at + len(part)] = part
        at += len(part)
        del part
    meta = (pd.concat(meta_parts, ignore_index=True)
              .sort_values("deactivated")
              .drop_duplicates("pod", keep="first")
              .reset_index(drop=True))
    days["prosumer"] = days["pod"].isin(prosumers)
    days.attrs["n_duplicates_resolved"] = n_dup
    days.attrs["n_folders"] = len(found)
    return days, values, meta


def read_pod_csv(path, **kwargs) -> "pd.DataFrame":
    """Read a CSV that carries a POD column, keeping the code as text.

    A POD such as 99999E00000053 is valid scientific notation, so any reader left
    to infer types turns it into 9.9999e+57, and 99999E00033204 into infinity.
    The file on disk is correct; it is the reading that destroys it. Excel does
    the same on a double click, so the column has to be imported as text there
    too.
    """
    dtype = {"pod": str}
    dtype.update(kwargs.pop("dtype", {}) or {})
    return pd.read_csv(path, dtype=dtype, **kwargs)