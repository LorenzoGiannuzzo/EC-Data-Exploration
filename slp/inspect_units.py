"""What unit are the readings in?

    python inspect_units.py

The outlier filter of Section 2.2 compares a quarter-hourly reading against the
contractual power. That comparison only holds if the units are what the code
assumes: kWh for the readings, kW for the power. This prints enough to tell.
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
idx_all = scan_data_dir(cfg.data_dir)

MON = {1: "gen", 2: "feb", 3: "mar", 4: "apr", 5: "mag", 6: "giu",
       7: "lug", 8: "ago", 9: "set", 10: "ott", 11: "nov", 12: "dic"}

print("\n" + "=" * 70)
print("COVERAGE BY YEAR")
print("=" * 70)
for y, g in idx_all.groupby("year"):
    present = sorted(g["month"].tolist())
    missing = sorted(set(range(1, 13)) - set(present))
    days_ok = sum(pd.Period(f"{y}-{m:02d}").days_in_month for m in present)
    mark = "  <-- complete" if not missing else ""
    print(f"\n  {y}: {len(present)}/12 months, {days_ok} calendar days{mark}")
    print(f"      present: {' '.join(MON[m] for m in present)}")
    if missing:
        print(f"      MISSING: {' '.join(MON[m] for m in missing)}")
        print(f"      -> at most {days_ok} days per POD. min_valid_days must be below that.")

idx = idx_all[idx_all["year"] == cfg.year]
if idx.empty:
    print(f"\nNo folder for the configured year {cfg.year}. Nothing to inspect.")
    sys.exit(0)

print(f"\n\nconfigured year: {cfg.year}")

row = idx.iloc[len(idx) // 2]
path = row["meas_file"]
print(f"\ninspecting: {row['folder']}/{path.name}")

enc, sep, dec = _sniff_csv(path)
df = pd.read_csv(path, sep=sep, decimal=dec, encoding=enc, low_memory=False)
df.columns = [str(c).strip() for c in df.columns]
qs = _q_columns(df)[:96]
df[qs] = df[qs].apply(pd.to_numeric, errors="coerce")

q = df[qs].to_numpy(dtype="float64")
finite = q[np.isfinite(q)]

print("\n" + "=" * 70)
print("QUARTER-HOURLY READINGS")
print("=" * 70)
for label, v in [("min", np.min(finite)), ("p25", np.percentile(finite, 25)),
                 ("median", np.median(finite)), ("p75", np.percentile(finite, 75)),
                 ("p99", np.percentile(finite, 99)), ("max", np.max(finite))]:
    print(f"    {label:8s} {v:>14,.4f}")
print(f"    zeros    {np.mean(finite == 0) * 100:>13.1f} %")
print(f"    negative {np.mean(finite < 0) * 100:>13.1f} %")

daily = np.nansum(q, axis=1)
daily = daily[daily > 0]
print("\n    daily total per row (sum of the 96 readings):")
for label, v in [("median", np.median(daily)), ("p99", np.percentile(daily, 99))]:
    print(f"        {label:8s} {v:>14,.2f}")

print("\n    A domestic POD uses roughly 8-12 kWh a day, a small business 50-300.")
print("    If the median daily total is in the thousands, the readings are in Wh.")

# ── contractual power ────────────────────────────────────────────────────────
pcol = None
for c in df.columns:
    if "potenza" in c.lower() or "power" in c.lower():
        pcol = c
        break

print("\n" + "=" * 70)
print(f"CONTRACTUAL POWER  (column '{pcol}')")
print("=" * 70)
if pcol is None:
    print("    not found")
else:
    p = pd.to_numeric(df[pcol], errors="coerce")
    print(f"    distinct values : {p.nunique()}")
    print(f"    min / median / max : {p.min():,.2f} / {p.median():,.2f} / {p.max():,.2f}")
    print("\n    most common values:")
    for v, n in p.value_counts().head(10).items():
        print(f"        {v:>12,.2f}   {n:>8,} rows")
    print("\n    A value like 3.0 / 4.5 / 6.0 is kW. A value like 3000 is W.")
    print("    A small integer such as 1, 2, 3 with few distinct values is a")
    print("    power CLASS id, not a power, and cannot be used as a threshold.")

    # implied ratio: what does the peak reading imply about the power?
    peak = np.nanmax(q, axis=1)
    ok = np.isfinite(peak) & p.notna().to_numpy() & (p.to_numpy() > 0)
    ratio = (peak[ok] * 4) / p.to_numpy()[ok]
    print("\n" + "=" * 70)
    print("IMPLIED RATIO  (peak reading x 4) / contractual power")
    print("=" * 70)
    for label, v in [("median", np.median(ratio)), ("p95", np.percentile(ratio, 95)),
                     ("p99", np.percentile(ratio, 99)), ("max", np.max(ratio))]:
        print(f"    {label:8s} {v:>14,.3f}")
    print("\n    If the readings are kWh and the power kW, this ratio is the")
    print("    fraction of the contractual power actually drawn: median well")
    print("    below 1, p99 near or slightly above 1.")
    print("    A median in the hundreds means the readings are in Wh (factor 1000).")
    print(f"\n    fraction above the current margin ({cfg.get('preprocessing.power_margin')}): "
          f"{np.mean(ratio > cfg.get('preprocessing.power_margin')) * 100:.1f} %")
print()
