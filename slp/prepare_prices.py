"""Convert a published price file into the format read by Eq. 14.

The comparison stage reads a CSV with a timestamp column and a price column in EUR/MWh
(comparison.price_file and comparison.imbalance_price_file in config.yaml). The files
published by GME (day-ahead zonal prices) and by Terna (imbalance prices) come as Excel
or CSV workbooks with a date column, an hour or period column and one column per zone.
This helper turns either into the expected format.

Examples
    # GME annual workbook, hourly rows, date as YYYYMMDD, hour from 1 to 24, North zone
    python prepare_prices.py ../data/prices/Anno2025.xlsx ../data/prices/pz_nord_2025.csv
        --sheet Prezzi-Prices --date-col Data --hour-col Ora --price-col NORD
        --date-format %Y%m%d --hour-base 1

    # a file that already carries a timestamp
    python prepare_prices.py terna_2025.csv ../data/prices/imbalance_nord_2025.csv
        --timestamp-col "Data Ora" --price-col "Prezzo" --decimal ,

    # quarter-hourly rows: give the period column and how many periods an hour holds
    python prepare_prices.py mgp_15min.xlsx out.csv --date-col Data --hour-col Periodo
        --periods-per-hour 4 --hour-base 1 --price-col NORD

-------------------------------------------------------------------------------
Author:        Lorenzo Giannuzzo
Affiliation:   Politecnico di Torino, Department of Energy (DENERG)
               Energy Center Lab
Contact:       lorenzo.giannuzzo@polito.it
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_any(path: Path, sheet: str | None, decimal: str, sep: str | None) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path, sep=sep, decimal=decimal, engine="python")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--timestamp-col", default=None)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--date-format", default=None)
    ap.add_argument("--hour-col", default=None)
    ap.add_argument("--hour-base", type=int, default=1, help="1 when the first hour is 1, 0 when it is 0")
    ap.add_argument("--periods-per-hour", type=int, default=1)
    ap.add_argument("--price-col", required=True)
    ap.add_argument("--decimal", default=".")
    ap.add_argument("--sep", default=None)
    a = ap.parse_args()

    df = read_any(Path(a.source), a.sheet, a.decimal, a.sep)
    #Lorenzo Giannuzzo: column names are matched ignoring case and surrounding spaces, since the
    # published workbooks are not consistent across years
    cols = {str(c).strip().lower(): c for c in df.columns}

    def col(name: str) -> str:
        key = name.strip().lower()
        if key not in cols:
            raise SystemExit(f"column {name!r} not found; available: {list(df.columns)}")
        return cols[key]

    price = pd.to_numeric(df[col(a.price_col)].astype(str).str.replace(",", ".", regex=False),
                          errors="coerce")
    if a.timestamp_col:
        ts = pd.to_datetime(df[col(a.timestamp_col)], dayfirst=True, errors="coerce")
    else:
        if not (a.date_col and a.hour_col):
            raise SystemExit("give --timestamp-col, or --date-col together with --hour-col")
        date = pd.to_datetime(df[col(a.date_col)].astype(str), format=a.date_format,
                              dayfirst=True, errors="coerce")
        period = pd.to_numeric(df[col(a.hour_col)], errors="coerce") - a.hour_base
        minutes = period * (60 // a.periods_per_hour)
        ts = date + pd.to_timedelta(minutes, unit="min")
    out = pd.DataFrame({"timestamp": ts, "price_EUR_MWh": price}).dropna()
    out = out.sort_values("timestamp")
    Path(a.target).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.target, index=False)
    print(f"{len(out)} rows written to {a.target}, from {out['timestamp'].min()} "
          f"to {out['timestamp'].max()}, mean price {out['price_EUR_MWh'].mean():.2f} EUR/MWh")


if __name__ == "__main__":
    main()
