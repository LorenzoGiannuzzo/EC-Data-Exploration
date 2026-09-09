"""What is in the Q columns beyond Q96?

    python inspect_columns.py

Reads one monthly file and reports how many Q columns there are, which rows use
the ones past Q96, and on what dates. If the surplus columns only carry values
on the day DST ends, the reading in io.py is right and nothing else is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.io import _q_columns, _sniff_csv, scan_data_dir

cfg = load_config()
idx = scan_data_dir(cfg.data_dir)
print(f"\nfolders found: {len(idx)}   years: {sorted(idx['year'].unique())}\n")

# a file from October, where DST ends; fall back to the first available
oct_rows = idx[(idx["month"] == 10) & (idx["year"] == cfg.year)]
row = oct_rows.iloc[0] if len(oct_rows) else idx.iloc[0]
path = row["meas_file"]
print(f"inspecting: {row['folder']}/{path.name}")

enc, sep, dec = _sniff_csv(path)
print(f"  encoding={enc}  separator='{sep}'  decimal='{dec}'\n")

df = pd.read_csv(path, sep=sep, decimal=dec, encoding=enc, low_memory=False)
df.columns = [str(c).strip() for c in df.columns]

qs = _q_columns(df)
non_q = [c for c in df.columns if c not in qs]
print(f"  Q columns : {len(qs)}   from {qs[0]} to {qs[-1]}")
print(f"  other cols: {non_q}\n")

if len(qs) > 96:
    extra = qs[96:]
    df[extra] = df[extra].apply(pd.to_numeric, errors="coerce")
    used = df[extra].notna().any(axis=1)
    print(f"  columns past Q96: {extra}")
    print(f"  rows using them : {int(used.sum()):,} of {len(df):,}\n")

    dcol = next((c for c in df.columns if c.lower() == "datamisura"), None)
    if dcol and used.any():
        d = pd.to_datetime(df.loc[used, dcol], dayfirst=True, errors="coerce")
        print("  dates on which they are used:")
        for dt, n in d.value_counts().sort_index().items():
            print(f"      {dt.date()}  ({dt.strftime('%A')})  {n:,} rows")
    elif not used.any():
        print("  never used in this file: they are padding on every day.")

# how many readings per row, to see the 92 / 96 / 100 pattern
print("\n  readings per row (non-null Q values):")
df[qs] = df[qs].apply(pd.to_numeric, errors="coerce")
counts = df[qs].notna().sum(axis=1)
for n, c in counts.value_counts().sort_index().items():
    print(f"      {n:>3} readings : {c:>7,} rows")

dcol = next((c for c in df.columns if c.lower() == "datamisura"), None)
if dcol:
    odd = counts != 96
    if odd.any():
        d = pd.to_datetime(df.loc[odd, dcol], dayfirst=True, errors="coerce")
        print("\n  dates whose rows do not have exactly 96 readings:")
        for dt, n in d.value_counts().sort_index().head(12).items():
            print(f"      {dt.date()}  ({dt.strftime('%A')})  {n:,} rows")
print()
