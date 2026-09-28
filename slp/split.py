"""Stage 1c — One class of users on the shared dictionary (branch split-domestic).

    python main.py --config config_split_domestic.yaml --from split --to generation
    python main.py --config config_split_nondomestic.yaml --from split --to generation

The domestic and the non-domestic points are clustered apart, each class in a run
of its own, while their days keep the vocabulary of the whole population: the
dictionary of the base run is copied next to the reduced cache and the clustering
stage reads it (clustering.fixed_dictionary) instead of building a new one. The two
catalogs are therefore written in the same forms and can be compared form by form,
which two dictionaries built apart would not allow.

The stage reads the cache of the base run (split.source_cache), keeps the points
of one class, renumbers shape_idx as the pre-processing does, and never modifies
the source.

Outputs
    <cache_dir>/shapes.npy, days.parquet, users.parquet     the points of the class
    <cache_dir>/shared_dictionary.npy                        the dictionary of the base run
    <results_dir>/1_preprocessing/                           copied from the source run
    <results_dir>/1_preprocessing/split/split_summary.txt

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
from balance import CACHE_FILES, copy_rows, residential_mask  # noqa: E402
from common.cache import read_manifest  # noqa: E402
from common.config import ROOT, STAGE_FOLDERS, load_config  # noqa: E402

CLASSES = ("domestic", "non_domestic")


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    sp = cfg.get("split") or {}
    print(f"\n{'='*78}\nSTAGE 1c — ONE CLASS ON THE SHARED DICTIONARY\n{'='*78}\n")
    if not sp.get("enabled", False):
        print("  split.enabled is false or absent in this configuration: nothing to do.\n")
        return

    keep = str(sp.get("keep", "")).lower()
    if keep not in CLASSES:
        raise ValueError(f"split.keep must be one of {CLASSES}, got {keep!r}")
    src_cache = (ROOT / str(sp.get("source_cache", "cache"))).resolve()
    dst_cache = cfg.cache_dir.resolve()
    if src_cache == dst_cache:
        raise ValueError(f"split.source_cache and output.cache_dir are the same folder "
                         f"({dst_cache}). The split run needs a cache of its own.")
    missing = [f for f in CACHE_FILES + ("dictionary.npy",) if not (src_cache / f).exists()]
    if missing:
        raise FileNotFoundError(f"{src_cache} lacks {', '.join(missing)}: run the base "
                                f"configuration through the clustering stage first.")
    if not cfg.get("clustering.fixed_dictionary"):
        raise ValueError("clustering.fixed_dictionary is not set: without it the clustering "
                         "stage would build a new dictionary on this class alone.")

    #Lorenzo Giannuzzo: the dictionary is only meaningful on shapes normalised and cut the way
    # it was built on, so the configuration that wrote it is checked before it is reused
    man = read_manifest(src_cache) or {}
    for key in ("shape_unit", "dictionary_resolution", "dictionary_scope"):
        here = cfg.get(f"clustering.{key}")
        if key in man and str(man[key]) != str(here):
            raise ValueError(f"the source dictionary was built with {key} = {man[key]!r}, "
                             f"this configuration declares {here!r}")
    cent = np.load(src_cache / "dictionary.npy")

    prefixes = list(sp.get("residential_prefixes", ["DO"]))
    users = pd.read_parquet(src_cache / "users.parquet")
    days = pd.read_parquet(src_cache / "days.parquet")
    shapes = np.load(src_cache / "shapes.npy", mmap_mode="r")
    print(f"  source cache: {src_cache}")
    print(f"  {len(users):,} PODs, {len(days):,} days, {len(shapes):,} shapes")
    print(f"  shared dictionary: D = {len(cent)} codewords "
          f"(manifest D = {man.get('n_codewords', 'n/a')})\n")

    is_dom = residential_mask(users, prefixes)
    sel = is_dom if keep == "domestic" else ~is_dom
    keep_users = users[sel]
    keep_pods = set(keep_users["pod"])

    keep_day = days["pod"].isin(keep_pods).to_numpy()
    d_new = days.loc[keep_day].copy()
    has = d_new["shape_idx"].to_numpy() >= 0
    old_idx = d_new.loc[has, "shape_idx"].to_numpy(dtype="int64")
    d_new["shape_idx"] = -1
    d_new.loc[has, "shape_idx"] = np.arange(int(has.sum()))
    d_new = d_new.reset_index(drop=True)

    #Lorenzo Giannuzzo: whatever a previous run left in this cache belongs to another population
    for p in dst_cache.iterdir():
        if p.is_file():
            p.unlink()
    copy_rows(shapes, old_idx, dst_cache / "shapes.npy")
    d_new.to_parquet(dst_cache / "days.parquet", index=False)
    keep_users.reset_index(drop=True).to_parquet(dst_cache / "users.parquet", index=False)
    target = Path(str(cfg.get("clustering.fixed_dictionary")))
    target = target if target.is_absolute() else dst_cache / target
    np.save(target, cent)

    src_res = (ROOT / str(sp.get("source_results", "paper_results"))
               / STAGE_FOLDERS["preprocessing"]).resolve()
    out_pre = cfg.results_dir("preprocessing")
    if src_res.exists() and src_res != out_pre.resolve():
        shutil.copytree(src_res, out_pre, dirs_exist_ok=True)
    out = out_pre / "split"
    out.mkdir(parents=True, exist_ok=True)

    e_all = users["E"].sum()
    lines = [
        f"Source cache                 {src_cache}",
        f"Class kept                   {keep} (residential codes {prefixes})",
        f"Shared dictionary            D = {len(cent)}, from {src_cache.name}/dictionary.npy",
        f"PODs retained                {len(users):,} -> {len(keep_users):,}",
        f"Share of the PODs            {len(keep_users)/max(len(users),1):.3f}",
        f"Share of the annual energy   {keep_users['E'].sum()/max(e_all,1e-9):.3f}",
        f"Days retained                {len(days):,} -> {len(d_new):,}",
        f"Shapes retained              {len(shapes):,} -> {int(has.sum()):,}",
        f"PODs without any shape       {int((d_new.groupby('pod')['has_shape'].sum() == 0).sum())}",
    ]
    if keep == "non_domestic":
        lines.append(f"Without an activity label   {int(keep_users['ateco_l1'].isna().sum())}")
    (out / "split_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  " + "\n  ".join(lines))
    print(f"\n  cache/   {dst_cache.name}/shapes.npy, days.parquet, users.parquet, {target.name}")
    print(f"  results/ {out}")
    print(f"  done in {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
