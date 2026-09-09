"""Loading of the national standard profiles (Section 2.5, input I2).

The ARERA workbooks are identified by the `Classe_potenza` column *inside* the file and
never by the file name. One of the files supplied for this study carried a name announcing
one power class and contained another, and the duplicate was only visible by reading the
column, so content-based identification is a correctness requirement here rather than a
stylistic preference. Duplicated classes are reported and dropped.

Market selection follows a single rule: where a workbook resolves the market, the free
market is taken; where it carries only the aggregate, the aggregate is taken. The choice
is recorded per class so that Section 2.5 can state it.
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

VALUE_COL = "Prelievo medio Orario Provinciale (kWh)"
FREE_MARKET = "Mercato Libero"
AGGREGATE = "Tutti"

_CLASS_LABEL = {
    "0<potenza_impegnata<=1.5": "0-1.5",
    "1.5<potenza_impegnata<=3": "1.5-3",
    "3<potenza_impegnata<=4.5": "3-4.5",
    "4.5<potenza_impegnata<=6": "4.5-6",
    "potenza_impegnata>6": ">6",
}
_DAYTYPE = {"Giorno feriale": "weekday", "Sabato": "saturday", "Domenica": "sunday"}


# ------------------------------------------------------------------------------- ARERA
def _read_arera_workbook(path: Path, province: str) -> pd.DataFrame:
    df = pd.read_excel(path, usecols=[
        "Anno Mese", "Provincia", "Tipo mercato", "Classe_potenza",
        "Residenza ", "Working Day", "Orario", VALUE_COL])
    df = df[df["Provincia"] == province].copy()
    # The annual summary rows carry a bare year in `Anno Mese`, which pandas would parse
    # as a nanosecond offset from the epoch and silently relabel as January, duplicating
    # that month. Only genuine timestamps are kept.
    is_month = df["Anno Mese"].map(lambda x: isinstance(x, (pd.Timestamp, datetime.datetime)))
    df = df[is_month].copy()
    df["month"] = pd.to_datetime(df["Anno Mese"]).dt.month.astype(int)
    df["hour"] = df["Orario"].str.replace("Ora", "", regex=False).astype(int) - 1
    df["daytype"] = df["Working Day"].map(_DAYTYPE)
    df["residency"] = df["Residenza "].str.strip()
    df["power_class"] = df["Classe_potenza"].map(_CLASS_LABEL)
    if df["power_class"].isna().any():
        unknown = sorted(df.loc[df["power_class"].isna(), "Classe_potenza"].unique())
        raise ValueError(f"{path.name}: unrecognised power class {unknown}")
    return df.rename(columns={"Tipo mercato": "market", VALUE_COL: "kWh"})


def load_arera(data_dir: Path, province: str, cache_dir: Path | None = None,
               verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (table, provenance).

    table      month x daytype x hour x power_class x residency -> mean hourly kWh per point
    provenance one row per power class: file used, market taken, sha1 of the values
    """
    data_dir = Path(data_dir)
    cache = (Path(cache_dir) / f"arera_{province.lower()}.parquet") if cache_dir else None
    prov_cache = cache.with_name(cache.stem + "_provenance.parquet") if cache else None
    if cache and cache.exists() and prov_cache.exists():
        return pd.read_parquet(cache), pd.read_parquet(prov_cache)

    frames, provenance, seen = [], [], {}
    for path in sorted(data_dir.glob("*.xlsx")):
        try:
            raw = _read_arera_workbook(path, province)
        except Exception:
            continue                        # not an ARERA workbook
        if raw.empty:
            continue
        for cls, block in raw.groupby("power_class", observed=True):
            markets = set(block["market"])
            if FREE_MARKET in markets:
                chosen, sel = FREE_MARKET, block[block["market"] == FREE_MARKET]
            elif AGGREGATE in markets:
                chosen, sel = AGGREGATE, block[block["market"] == AGGREGATE]
            else:
                continue
            sel = sel[["month", "daytype", "hour", "residency", "kWh"]].copy()
            digest = hashlib.sha1(
                np.ascontiguousarray(
                    sel.sort_values(["month", "daytype", "hour", "residency"])["kWh"]
                    .to_numpy(dtype="float64"))).hexdigest()[:12]
            if cls in seen:
                if verbose:
                    print(f"  ! {path.name}: class {cls} already loaded from "
                          f"{seen[cls]['file']}, dropped "
                          f"({'identical values' if digest == seen[cls]['sha1'] else 'DIFFERENT values'})")
                continue
            sel["power_class"] = cls
            frames.append(sel)
            rec = {"power_class": cls, "file": path.name, "market": chosen,
                   "sha1": digest, "rows": len(sel)}
            seen[cls] = rec
            provenance.append(rec)

    if not frames:
        raise FileNotFoundError(f"no usable ARERA workbook for province {province!r} in {data_dir}")
    table = pd.concat(frames, ignore_index=True)
    prov = pd.DataFrame(provenance).sort_values("power_class").reset_index(drop=True)

    dup = prov[prov.duplicated("sha1", keep=False)]
    if len(dup) and verbose:
        print("  ! classes sharing identical values, inspect before use:\n",
              dup[["power_class", "file"]].to_string(index=False))

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache, index=False)
        prov.to_parquet(prov_cache, index=False)
    return table, prov


# --------------------------------------------------------------------------------- GSE
def load_gse(data_dir: Path, cache_dir: Path | None = None) -> pd.DataFrame:
    """Hourly GSE profiles, one column per published profile.

    Returns a frame indexed by (month, day, hour) with the year the file refers to kept in
    an attribute. The normalisation is left untouched: the monorario columns sum to one
    over each month, the banded columns sum to one over each band within each month.
    """
    data_dir = Path(data_dir)
    cache = (Path(cache_dir) / "gse.parquet") if cache_dir else None
    if cache and cache.exists():
        return pd.read_parquet(cache)
    candidates = [p for p in sorted(data_dir.glob("*.xlsx")) if "gse" in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"no GSE workbook in {data_dir}")
    df = pd.read_excel(candidates[0])
    df = df.rename(columns={"Anno": "year", "Mese": "month", "Giorno": "day", "Ora": "hour"})
    keep = ["year", "month", "day", "hour"] + [c for c in df.columns if len(str(c)) == 4
                                               and str(c).isupper()]
    df = df[keep]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def gse_normalisation_report(gse: pd.DataFrame) -> pd.DataFrame:
    """Evidence for the normalisation claims made in Section 2.5.

    For each profile column: the annual total, the monthly total, and the ratio between
    the mean daily energy of a Sunday and of a working day. The last one is the quantity
    that shows whether the profile differentiates day types in level at all.
    """
    cols = [c for c in gse.columns if c not in ("year", "month", "day", "hour")]
    d = gse.copy()
    d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day))
    d["dow"] = d["date"].dt.dayofweek
    rows = []
    for c in cols:
        monthly = d.groupby("month")[c].sum()
        daily = d.groupby(["date", "dow"])[c].sum().reset_index()
        wk = daily.loc[daily.dow < 5, c].mean()
        su = daily.loc[daily.dow == 6, c].mean()
        rows.append({"profile": c,
                     "annual_total": d[c].sum(),
                     "monthly_total_min": monthly.min(),
                     "monthly_total_max": monthly.max(),
                     "sunday_over_weekday_energy": su / wk if wk else np.nan})
    return pd.DataFrame(rows)
