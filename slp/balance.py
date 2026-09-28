"""Stage 1b — Balanced population (branch balanced-downsampling).

    python main.py --config config_balanced.yaml --stage balance
    python main.py --config config_balanced.yaml --from balance

Residential points are 87% of the retained population, so the dictionary and the
user partition are shaped almost entirely by them. This stage draws a random,
stratified subset of the residential points so that the population the clustering
starts from is balanced between residential and non-residential users, and hands
it to the unchanged methodology.

The stage reads the cache written by the pre-processing of the base configuration
(balance.source_cache) and writes a reduced copy into the cache of this
configuration. The raw data are not read again, the source cache is never
modified, and shapes.npy is rewritten with shape_idx renumbered, so every later
stage sees an ordinary cache that happens to hold fewer points.

Outputs
    <cache_dir>/shapes.npy, days.parquet, users.parquet     the reduced population
    <results_dir>/1_preprocessing/                          copied from the source run
    <results_dir>/1_preprocessing/balancing/
        composition_before_after.csv   points, energy and days per class
        strata.csv                     residential points per stratum, drawn and kept
        selected_pods.csv              every retained point, with its class
        balance_summary.txt

-------------------------------------------------------------------------------
Author:        Lorenzo Giannuzzo
Affiliation:   Politecnico di Torino, Department of Energy (DENERG)
               Energy Center Lab
Contact:       lorenzo.giannuzzo@polito.it

Developed in collaboration with ENEA within the Italian Research on the Electric
System programme (Ricerca di Sistema Elettrico).
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import ROOT, STAGE_FOLDERS, load_config  # noqa: E402

CACHE_FILES = ("shapes.npy", "days.parquet", "users.parquet")
CHUNK = 200_000


def residential_mask(users: pd.DataFrame, prefixes: list[str]) -> np.ndarray:
    """True where the activity label starts with one of the residential codes.

    Points without a label are not residential and are kept whole, since nothing
    says which side of the balance they belong to; the summary counts them.
    """
    if "ateco_l1" not in users:
        raise KeyError("users.parquet has no ateco_l1 column: re-run the pre-processing "
                       "of the base configuration")
    l1 = users["ateco_l1"].astype("string").str.strip().str.upper()
    return l1.isin([p.upper() for p in prefixes]).fillna(False).to_numpy(dtype=bool)


def build_strata(res: pd.DataFrame, columns: list[str], n_energy_bins: int) -> pd.Series:
    """One label per residential point, joining the configured stratification keys.

    'E_quantile' is not a column of users.parquet: it is the quantile bin of the
    annual energy, computed on the residential points alone, so that the draw keeps
    the distribution of consumption and not only its composition by label.
    """
    parts = []
    for c in columns:
        if c == "E_quantile":
            b = pd.qcut(res["E"].rank(method="first"), q=n_energy_bins,
                        labels=False, duplicates="drop")
            parts.append("Eq" + b.fillna(-1).astype(int).astype(str))
        elif c in res:
            parts.append(res[c].astype("string").fillna("NA").astype(str))
        else:
            raise KeyError(f"balance.strata: column {c!r} is not in users.parquet "
                           f"(available: {', '.join(res.columns)})")
    if not parts:
        return pd.Series("all", index=res.index)
    lab = parts[0]
    for p in parts[1:]:
        lab = lab + "|" + p
    return lab


def allocate(sizes: pd.Series, target: int) -> pd.Series:
    """Proportional allocation of `target` draws over the strata, largest remainder.

    The quotas sum to the target exactly, which a plain rounding does not guarantee,
    and no stratum is asked for more points than it holds.
    """
    total = int(sizes.sum())
    if target >= total:
        return sizes.astype(int)
    exact = sizes * (target / total)
    quota = np.floor(exact).astype(int)
    rest = target - int(quota.sum())
    order = (exact - quota).sort_values(ascending=False).index
    for s in order[:rest]:
        quota[s] += 1
    return quota.clip(upper=sizes).astype(int)


def copy_rows(src: np.ndarray, idx: np.ndarray, dest: Path) -> None:
    """Write src[idx] to dest in chunks, so the full shapes.npy is never in memory."""
    out = np.lib.format.open_memmap(dest, mode="w+", dtype=src.dtype,
                                    shape=(len(idx), src.shape[1]))
    for i in range(0, len(idx), CHUNK):
        j = idx[i:i + CHUNK]
        out[i:i + len(j)] = src[j]
    out.flush()
    del out


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    bal = cfg.get("balance") or {}
    print(f"\n{'='*78}\nSTAGE 1b — BALANCED POPULATION\n{'='*78}\n")
    if not bal.get("enabled", False):
        print("  balance.enabled is false or absent in this configuration: nothing to do.\n")
        return

    src_cache = (ROOT / str(bal.get("source_cache", "cache"))).resolve()
    dst_cache = cfg.cache_dir.resolve()
    #Lorenzo Giannuzzo: the stage rewrites its own cache from the source one, so the two must
    # never coincide: pointed at the same folder it would overwrite the base population.
    if src_cache == dst_cache:
        raise ValueError(f"balance.source_cache and output.cache_dir are the same folder "
                         f"({dst_cache}). The balanced run needs a cache of its own.")
    missing = [f for f in CACHE_FILES if not (src_cache / f).exists()]
    if missing:
        raise FileNotFoundError(f"{src_cache} lacks {', '.join(missing)}: run the "
                                f"pre-processing of the base configuration first.")

    prefixes = list(bal.get("residential_prefixes", ["DO"]))
    basis = str(bal.get("basis", "count")).lower()
    ratio = float(bal.get("ratio", 1.0))
    seed = int(bal.get("seed", 42))
    strata_cols = list(bal.get("strata", ["ateco_l2", "prosumer", "E_quantile"]))
    n_bins = int(bal.get("energy_bins", 5))
    if basis not in ("count", "energy"):
        raise ValueError(f"balance.basis must be count | energy, got {basis!r}")

    users = pd.read_parquet(src_cache / "users.parquet")
    days = pd.read_parquet(src_cache / "days.parquet")
    shapes = np.load(src_cache / "shapes.npy", mmap_mode="r")
    print(f"  source cache: {src_cache}")
    print(f"  {len(users):,} PODs, {len(days):,} days, {len(shapes):,} shapes\n")

    is_res = residential_mask(users, prefixes)
    res, oth = users[is_res], users[~is_res]
    n_unlabeled = int(users["ateco_l1"].isna().sum())

    #Lorenzo Giannuzzo: the target is set on the non-residential side, which is kept whole. On a
    # count basis the residential points drawn equal ratio x the non-residential points; on
    # an energy basis their annual energy does, which on this population keeps more
    # residential points, since a non-residential point withdraws several times as much.
    if basis == "count":
        frac = ratio * len(oth) / max(len(res), 1)
    else:
        frac = ratio * oth["E"].sum() / max(res["E"].sum(), 1e-9)
    frac = min(frac, 1.0)
    target = int(round(frac * len(res)))
    if frac >= 1.0:
        print(f"  NOTE: the target reaches every residential point (fraction {frac:.3f}); "
              f"nothing is drawn away.")

    strata = build_strata(res, strata_cols, n_bins)
    sizes = strata.value_counts().sort_index()
    quota = allocate(sizes, target)

    rng = np.random.default_rng(seed)
    picked = []
    for s in sizes.index:
        pool = res.index[strata.to_numpy() == s].to_numpy()
        k = int(quota[s])
        if k > 0:
            picked.append(rng.choice(pool, size=k, replace=False))
    picked = np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)

    keep_users = pd.concat([users.loc[picked], oth]).sort_index()
    keep_pods = set(keep_users["pod"])
    print(f"  residential codes {prefixes}, basis {basis}, ratio {ratio}, seed {seed}")
    print(f"  strata {strata_cols}: {len(sizes)} strata")
    print(f"  residential     {len(res):>7,} -> {len(picked):>7,}  "
          f"(fraction kept {len(picked)/max(len(res),1):.3f})")
    print(f"  non-residential {len(oth):>7,} -> {len(oth):>7,}  (kept whole"
          f"{f', {n_unlabeled} without a label' if n_unlabeled else ''})")

    # ── reduce the cache ─────────────────────────────────────────────────────
    #Lorenzo Giannuzzo: the order of days.parquet is preserved, and shape_idx is renumbered over
    # the retained days that carry a shape, which is exactly how the pre-processing
    # numbers it: every later stage indexes shapes.npy through it and needs nothing else.
    keep_day = days["pod"].isin(keep_pods).to_numpy()
    d_new = days.loc[keep_day].copy()
    has = d_new["shape_idx"].to_numpy() >= 0
    old_idx = d_new.loc[has, "shape_idx"].to_numpy(dtype="int64")
    d_new["shape_idx"] = -1
    d_new.loc[has, "shape_idx"] = np.arange(int(has.sum()))
    d_new = d_new.reset_index(drop=True)

    #Lorenzo Giannuzzo: whatever a previous run left in this cache (dictionary, groups, profiles,
    # manifest) belongs to another population and is removed, so that a stage failing
    # halfway cannot hand stale artefacts to the next one.
    for p in dst_cache.iterdir():
        if p.is_file():
            p.unlink()
    copy_rows(shapes, old_idx, dst_cache / "shapes.npy")
    d_new.to_parquet(dst_cache / "days.parquet", index=False)
    keep_users.reset_index(drop=True).to_parquet(dst_cache / "users.parquet", index=False)

    # ── results ──────────────────────────────────────────────────────────────
    #Lorenzo Giannuzzo: the later stages (numbers above all) read the tables of the
    # pre-processing, which this branch does not re-run. They are copied as they are and
    # describe the full population; what the balancing changed is written beside them.
    src_res = (ROOT / str(bal.get("source_results", "paper_results"))
               / STAGE_FOLDERS["preprocessing"]).resolve()
    out_pre = cfg.results_dir("preprocessing")
    if src_res.exists() and src_res != out_pre.resolve():
        shutil.copytree(src_res, out_pre, dirs_exist_ok=True)
    out = out_pre / "balancing"
    out.mkdir(parents=True, exist_ok=True)

    cls_all = pd.Series(np.where(is_res, "residential", "non-residential"),
                        index=users.index, name="class")
    n_days = days.groupby("pod").size()
    n_shp = days.loc[days["shape_idx"] >= 0].groupby("pod").size()

    def composition(u: pd.DataFrame, label: str) -> pd.DataFrame:
        c = cls_all.loc[u.index]
        g = pd.DataFrame({"class": c, "E": u["E"],
                          "days": u["pod"].map(n_days).fillna(0),
                          "shapes": u["pod"].map(n_shp).fillna(0)})
        t = g.groupby("class").agg(n_pods=("E", "size"), annual_energy_MWh=("E", "sum"),
                                   n_days=("days", "sum"), n_shapes=("shapes", "sum"))
        t["annual_energy_MWh"] /= 1000.0
        for col, share in (("n_pods", "share_of_pods"),
                           ("annual_energy_MWh", "share_of_energy"),
                           ("n_shapes", "share_of_shapes")):
            t[share] = t[col] / t[col].sum()
        t.insert(0, "population", label)
        return t.reset_index()

    comp = pd.concat([composition(users, "full"), composition(keep_users, "balanced")],
                     ignore_index=True)
    comp = comp[["population", "class"] + [c for c in comp.columns
                                           if c not in ("population", "class")]]
    comp.round(4).to_csv(out / "composition_before_after.csv", index=False)

    st = pd.DataFrame({"n_residential": sizes, "n_drawn": quota})
    st.index.name = "stratum"
    st.to_csv(out / "strata.csv")

    sel = keep_users[["pod", "E"] + [c for c in ("ateco_l1", "ateco_l2", "prosumer")
                                     if c in keep_users]].copy()
    sel.insert(1, "class", cls_all.loc[keep_users.index].to_numpy())
    sel.to_csv(out / "selected_pods.csv", index=False)

    b = comp[comp["population"] == "balanced"].set_index("class")
    f = comp[comp["population"] == "full"].set_index("class")
    lines = [
        f"Source cache                 {src_cache}",
        f"Residential codes            {prefixes}",
        f"Basis / ratio / seed         {basis} / {ratio} / {seed}",
        f"Strata                       {strata_cols} ({len(sizes)} strata, {n_bins} energy bins)",
        f"Residential PODs             {len(res):,} -> {len(picked):,}",
        f"Non-residential PODs         {len(oth):,} (kept whole, {n_unlabeled} without a label)",
        f"PODs retained                {len(users):,} -> {len(keep_users):,}",
        f"Days retained                {len(days):,} -> {len(d_new):,}",
        f"Shapes retained              {len(shapes):,} -> {int(has.sum()):,}",
        "",
        "Shares, full -> balanced        PODs            energy          shapes",
    ]
    for c in ("residential", "non-residential"):
        if c in b.index and c in f.index:
            lines.append(f"  {c:28s}"
                         f"{f.at[c,'share_of_pods']:.3f} -> {b.at[c,'share_of_pods']:.3f}   "
                         f"{f.at[c,'share_of_energy']:.3f} -> {b.at[c,'share_of_energy']:.3f}   "
                         f"{f.at[c,'share_of_shapes']:.3f} -> {b.at[c,'share_of_shapes']:.3f}")
    (out / "balance_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n  " + "\n  ".join(lines[-3:]))
    #Lorenzo Giannuzzo: the second stage weighs forms by energy, so a population balanced in
    # points is not balanced in what the frequency vectors carry. Said at run time, since
    # it decides how the new dictionary is to be read.
    if "non-residential" in b.index and b.at["non-residential", "share_of_energy"] > 0.65:
        print(f"\n  NOTE: balanced in points, the population is "
              f"{b.at['non-residential','share_of_energy']:.0%} non-residential in energy.")
        print(f"  basis: energy balances what the energy-weighted frequencies carry.")
    print(f"\n  cache/   {dst_cache.name}/shapes.npy, days.parquet, users.parquet")
    print(f"  results/ {out}")
    print(f"  done in {time.time()-t0:.0f}s\n")

if __name__ == "__main__":
    main()