"""Which window? More months or more PODs?

    python inspect_windows.py

The meters roll out over time, so a long window keeps only the PODs already
installed at its start. Reads dates and PODs alone, no Q columns.

Everything runs on a POD-by-month table (~200k rows) rather than on the POD-day
rows (~4M), and the windows are selected with numpy rather than with a Python
loop over every row, so this takes seconds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.io import _sniff_csv, read_metadata, scan_data_dir, split_ateco

MON = {1: "gen", 2: "feb", 3: "mar", 4: "apr", 5: "mag", 6: "giu",
       7: "lug", 8: "ago", 9: "set", 10: "ott", 11: "nov", 12: "dic"}


def tag(ym: int) -> str:
    return f"{MON[ym % 100]}{str(ym // 100)[2:]}"


def step(ym: int, k: int) -> int:
    y, m = ym // 100, ym % 100 + k
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return y * 100 + m


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    idx = scan_data_dir(cfg.data_dir)
    seasons = cfg.get("preprocessing.seasons")
    season_of = {m: lab for lab, ms in seasons.items() for m in ms}
    n_seasons = len(seasons)
    keep_kind = str(cfg.get("data.keep_kind", "AP")).upper()
    kind_col = str(cfg.get("data.meas_cols.kind", "Tipologia"))

    print("\nreading dates and PODs (no Q columns)...")
    parts = []
    for _, r in idx.iterrows():
        if r["meas_file"] is None:
            continue
        enc, sep, _ = _sniff_csv(r["meas_file"])
        head = pd.read_csv(r["meas_file"], sep=sep, encoding=enc, nrows=0)
        head.columns = [str(c).strip() for c in head.columns]
        cols = [c for c in head.columns
                if c.lower() in ("pod", "datamisura", kind_col.lower())]
        df = pd.read_csv(r["meas_file"], sep=sep, encoding=enc, usecols=cols,
                         low_memory=False)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if kind_col.lower() in df.columns:
            df = df[df[kind_col.lower()].astype(str).str.strip().str.upper() == keep_kind]
        dt = pd.to_datetime(df["datamisura"], dayfirst=True, errors="coerce")
        g = (pd.DataFrame({"pod": df["pod"].astype(str).str.strip(),
                           "ym": dt.dt.year * 100 + dt.dt.month,
                           "day": dt.dt.day})
             .dropna(subset=["ym"])
             .drop_duplicates(["pod", "ym", "day"])
             .groupby(["pod", "ym"], as_index=False)
             .size())
        parts.append(g)
        print(f"    {r['folder']:8s} {int(g['size'].sum()):>9,} POD-days   "
              f"{g['pod'].nunique():>6,} PODs")

    pm = (pd.concat(parts, ignore_index=True)
            .groupby(["pod", "ym"], as_index=False)["size"].sum())
    pm["ym"] = pm["ym"].astype(int)
    pm["season"] = (pm["ym"] % 100).map(season_of)
    print(f"\n  POD-by-month table: {len(pm):,} rows, {pm['pod'].nunique():,} PODs "
          f"({time.time()-t0:.0f}s)")

    avail = sorted(int(x) for x in pm["ym"].unique())
    avail_set = set(avail)
    days_in = {ym: pd.Period(f"{ym//100}-{ym%100:02d}").days_in_month for ym in avail}

    meta = pd.concat([read_metadata(r["meta_file"]) for _, r in idx.iterrows()
                      if r["meta_file"] is not None], ignore_index=True)
    meta = meta.drop_duplicates("pod", keep="last")
    acol = next((c for c in meta.columns if c.lower() == "ccatete"), None)
    if acol:
        meta["ateco_l1"] = [split_ateco(v)[0] for v in meta[acol]]
        meta["is_dom"] = meta["ateco_l1"].astype(str).str.startswith(("DO", "CO", "IL"))

    pods_all = pm["pod"].to_numpy()
    ym_all = pm["ym"].to_numpy()
    size_all = pm["size"].to_numpy()
    season_all = pm["season"].to_numpy()

    seen: set[tuple] = set()
    rows = []
    for start in avail:
        for length in (12, 11, 10, 9, 8, 7, 6):
            span = [step(start, k) for k in range(length)]
            present = tuple(x for x in span if x in avail_set)
            if len(present) < 6 or present in seen:
                continue
            seen.add(present)

            sel = np.isin(ym_all, present)
            if not sel.any():
                continue
            cal_days = sum(days_in[x] for x in present)
            sub = pd.DataFrame({"pod": pods_all[sel], "n": size_all[sel],
                                "s": season_all[sel]})
            per = sub.groupby("pod").agg(n=("n", "sum"), s=("s", "nunique"))

            for frac in (0.82, 0.70):
                thr = int(cal_days * frac)
                ok = per[(per["n"] >= thr) & (per["s"] == n_seasons)]
                pods = set(ok.index)
                n_cls, n_nondom = -1, -1
                if acol and pods:
                    mm = meta[meta["pod"].isin(pods)]
                    n_cls = int((mm.groupby("ateco_l1").size() >= 30).sum())
                    n_nondom = int((~mm["is_dom"]).sum())
                rows.append({
                    "window": f"{tag(present[0])}-{tag(present[-1])}",
                    "months": len(present),
                    "cal_days": cal_days,
                    "thr%": int(frac * 100),
                    "min_days": thr,
                    "PODs": len(pods),
                    "non_domestic": n_nondom,
                    "classes>=30": n_cls,
                })

    res = pd.DataFrame(rows)
    res = res[res["PODs"] > 0].sort_values(
        ["classes>=30", "non_domestic", "months"], ascending=False)

    print("\n" + "=" * 92)
    print("CANDIDATE WINDOWS   (sorted by usable classes, then non-domestic PODs)")
    print("=" * 92)
    print(res.head(20).to_string(index=False))

    out = cfg.results_dir("preprocessing") / "window_choice.csv"
    res.to_csv(out, index=False)
    print(f"\n  full table -> {out}")
    print("\n  'classes>=30' is what the metrics of Section 2.6 can use: a class")
    print("  below the floor does not enter them. A window is worth its months")
    print("  only if the classes survive it.")
    print(f"\n  done in {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
