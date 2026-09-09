"""What is actually in a row?

    python inspect_rows.py            inspects February, where there is no DST
    python inspect_rows.py Ott24      inspects a given folder

Two things do not add up in the funnel and this prints enough to settle both:
there are ~2 rows per POD-day, and in February every row uses columns past Q96
although daylight saving does not exist in February.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.io import _q_columns, _sniff_csv, scan_data_dir

cfg = load_config()
idx = scan_data_dir(cfg.data_dir)

want = sys.argv[1] if len(sys.argv) > 1 else None
if want:
    row = idx[idx["folder"].str.lower() == want.lower()].iloc[0]
else:
    feb = idx[idx["month"] == 2]
    row = feb.iloc[0] if len(feb) else idx.iloc[0]

path = row["meas_file"]
print(f"\ninspecting {row['folder']}/{path.name}\n")

enc, sep, dec = _sniff_csv(path)
df = pd.read_csv(path, sep=sep, decimal=dec, encoding=enc, low_memory=False)
df.columns = [str(c).strip() for c in df.columns]

qs = _q_columns(df)
non_q = [c for c in df.columns if c not in qs]

print("=" * 72)
print("EVERY COLUMN THAT IS NOT A Q")
print("=" * 72)
for c in non_q:
    v = df[c]
    nun = v.nunique(dropna=True)
    print(f"\n  {c}   ({nun} distinct)")
    if nun <= 12:
        for val, n in v.value_counts(dropna=False).items():
            print(f"      {str(val)[:44]:46s} {n:>9,} rows")
    else:
        print(f"      e.g. {[str(x)[:22] for x in v.dropna().unique()[:4]]}")

# ── what makes a row unique? ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("ROWS PER POD-DAY")
print("=" * 72)
pod_c = next((c for c in df.columns if c.lower() == "pod"), None)
date_c = next((c for c in df.columns if "data" in c.lower()), None)
if pod_c and date_c:
    per = df.groupby([pod_c, date_c]).size()
    print(f"\n  POD-days       : {len(per):,}")
    print(f"  rows           : {len(df):,}")
    print(f"  rows per POD-day:")
    for n, c in per.value_counts().sort_index().items():
        print(f"      {n} row(s) : {c:>9,} POD-days")

    # which column separates the duplicates?
    dup_keys = per[per > 1].index[:400]
    if len(dup_keys):
        sample = df.set_index([pod_c, date_c]).loc[dup_keys]
        print("\n  columns that differ within the same POD-day:")
        for c in non_q:
            if c in (pod_c, date_c):
                continue
            varies = sample.groupby(level=[0, 1])[c].nunique(dropna=False)
            frac = float((varies > 1).mean())
            flag = "   <-- THIS SPLITS THE ROWS" if frac > 0.8 else ""
            print(f"      {c:28s} differs in {frac*100:5.1f}% of POD-days{flag}")

# ── the columns past Q96 ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("COLUMNS PAST Q96")
print("=" * 72)
if len(qs) <= 96:
    print(f"\n  none: this file has {len(qs)} Q columns")
else:
    extra = qs[96:]
    df[qs] = df[qs].apply(pd.to_numeric, errors="coerce")
    print(f"\n  they are: {extra}")
    for c in extra:
        v = df[c]
        print(f"\n  {c}: {v.notna().sum():,} non-null of {len(v):,}   "
              f"zeros={int((v == 0).sum()):,}   "
              f"min={v.min()}  median={v.median()}  max={v.max()}")
    q96 = df[qs[95]]
    print(f"\n  for comparison, Q96: {q96.notna().sum():,} non-null   "
          f"zeros={int((q96 == 0).sum()):,}   median={q96.median()}")
    print("\n  If the surplus columns are all zero rather than empty, they are")
    print("  padding and the DST reading is wrong: nothing is lost by dropping")
    print("  them, but the log line is misleading.")

# ── readings per row ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("NON-NULL READINGS PER ROW")
print("=" * 72)
counts = df[qs].notna().sum(axis=1)
for n, c in counts.value_counts().sort_index().items():
    print(f"      {n:>4} readings : {c:>9,} rows")

nz = (df[qs].fillna(0) != 0).sum(axis=1)
print("\n  non-ZERO readings per row:")
for n, c in nz.value_counts().sort_index().head(8).items():
    print(f"      {n:>4} non-zero : {c:>9,} rows")
print(f"      ...")
print(f"\n  rows entirely at zero: {int((nz == 0).sum()):,} of {len(df):,} "
      f"({100*(nz == 0).mean():.1f}%)")
print()
