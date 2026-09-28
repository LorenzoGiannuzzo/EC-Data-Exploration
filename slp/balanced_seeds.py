"""How much of the balanced result is the draw?

    python balanced_seeds.py
    python balanced_seeds.py --config config_balanced.yaml

A single draw of the residential points is one sample among many. This repeats the
balance and clustering stages for every seed of balance.seeds, keeps what each run
selected, and measures how far the partitions agree on the points they share: the
non-residential points are in every run, the residential ones only in some.

The configured seed (balance.seed) is run last, so the cache is left on it and the
later stages (generation onwards) stay consistent with it.

Outputs, under <results_dir>/2_clustering/seeds/
    seed_<s>/                 the clustering tables of each run, and groups.parquet
    balanced_seeds.csv        D, K, silhouettes and shares, one row per seed
    balanced_seeds_ari.csv    ARI between every pair of seeds, on the shared points

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

import argparse
import itertools
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run(stage: str, config: Path) -> None:
    """Run one stage through main.py, echoing its output line by line."""
    print(f"\n  --- {stage} ---", flush=True)
    proc = subprocess.Popen([sys.executable, "-u", "main.py", "--config", str(config),
                             "--stage", stage],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print("  " + line.rstrip(), flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"{stage} failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_balanced.yaml")
    args = ap.parse_args()
    base_cfg = (ROOT / args.config).resolve()
    raw = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    bal = raw.get("balance") or {}
    if not bal.get("enabled", False):
        raise SystemExit(f"{args.config}: balance.enabled is not true")

    main_seed = int(bal.get("seed", 42))
    seeds = [int(s) for s in bal.get("seeds", []) if int(s) != main_seed] + [main_seed]

    #Lorenzo Giannuzzo: every run reads the same configuration with only the seed changed. The
    # temporary file sits next to the original so that the relative paths it holds
    # (data root, source cache) resolve exactly as they do there.
    tmp = base_cfg.with_name(base_cfg.stem + "_seedrun.yaml")
    os.environ["SLP_CONFIG"] = str(tmp)
    from common.config import load_config  # noqa: E402

    rows, parts = [], {}
    t_all = time.time()
    try:
        for s in seeds:
            print(f"\n{'#'*70}\n#  seed {s}{'  (configured seed, run last)' if s == main_seed else ''}"
                  f"\n{'#'*70}")
            t0 = time.time()
            raw["balance"]["seed"] = s
            tmp.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
                           encoding="utf-8")
            run("balance", tmp)
            run("clustering", tmp)

            cfg = load_config(tmp)
            res = cfg.results_dir("clustering")
            dest = res / "seeds" / f"seed_{s}"
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            for p in res.iterdir():
                if p.is_file():
                    shutil.copy2(p, dest / p.name)
            g = pd.read_parquet(cfg.cache_dir / "groups.parquet")
            g.to_parquet(dest / "groups.parquet", index=False)
            parts[s] = g.set_index("pod")["group"]

            users = pd.read_parquet(cfg.cache_dir / "users.parquet")
            pref = [p.upper() for p in bal.get("residential_prefixes", ["DO"])]
            is_res = users["ateco_l1"].astype("string").str.upper().isin(pref).fillna(False)
            res_pods = set(users.loc[is_res.to_numpy(dtype=bool), "pod"])
            g["res"] = g["pod"].isin(res_pods)
            by_group = g.groupby("group")["res"].mean()

            abl = pd.read_csv(res / "ablation.csv").iloc[0]
            dic = pd.read_csv(res / "dictionary.csv")
            rows.append({
                "seed": s,
                "n_pods": len(g),
                "D": len(dic),
                "K": int(g["group"].nunique()),
                "silhouette_two_stage": round(float(abl["silhouette_two_stage"]), 4),
                "silhouette_mean_curves": round(float(abl["silhouette_mean_curves"]), 4),
                "ARI_vs_mean_curves": round(float(abl["ARI"]), 4),
                "largest_form_share": round(float(dic["share_of_days"].max()), 4),
                "largest_group_share": round(float(g["group"].value_counts(normalize=True).max()), 4),
                #Lorenzo Giannuzzo: how cleanly the partition separates the two classes: groups
                # at least 80% residential or at most 20%, the threshold of the comparison
                "groups_mostly_residential": int((by_group >= 0.8).sum()),
                "groups_mostly_nonresidential": int((by_group <= 0.2).sum()),
                "groups_mixed": int(((by_group > 0.2) & (by_group < 0.8)).sum()),
                "seconds": round(time.time() - t0),
            })
    finally:
        if tmp.exists():
            tmp.unlink()

    out = load_config(base_cfg).results_dir("clustering") / "seeds"
    df = pd.DataFrame(rows)
    df.to_csv(out / "balanced_seeds.csv", index=False)

    #Lorenzo Giannuzzo: ARI on the points two runs share. The non-residential points are always
    # shared, so the ARI restricted to them is reported too: it says whether the
    # non-residential groups survive a change of the residential draw.
    users = pd.read_parquet(load_config(base_cfg).cache_dir / "users.parquet")
    pref = [p.upper() for p in bal.get("residential_prefixes", ["DO"])]
    nonres = set(users.loc[~users["ateco_l1"].astype("string").str.upper()
                           .isin(pref).fillna(False).to_numpy(dtype=bool), "pod"])
    pairs = []
    for a, b in itertools.combinations(seeds, 2):
        common = parts[a].index.intersection(parts[b].index)
        nr = common.intersection(pd.Index(sorted(nonres)))
        pairs.append({
            "seed_a": a, "seed_b": b,
            "shared_pods": len(common),
            "ARI_shared": round(adjusted_rand_score(parts[a][common], parts[b][common]), 4),
            "shared_nonresidential": len(nr),
            "ARI_nonresidential": (round(adjusted_rand_score(parts[a][nr], parts[b][nr]), 4)
                                   if len(nr) else float("nan")),
        })
    ari = pd.DataFrame(pairs)
    ari.to_csv(out / "balanced_seeds_ari.csv", index=False)

    print(f"\n{'='*70}\nSEEDS\n{'='*70}")
    print(df.to_string(index=False))
    num = df.drop(columns=["seed", "seconds"])
    print("\n  mean / std over the seeds")
    print(pd.DataFrame({"mean": num.mean().round(4), "std": num.std().round(4)}).to_string())
    if len(ari):
        print(f"\n  ARI between seeds, shared points:      mean {ari['ARI_shared'].mean():.3f}"
              f"  min {ari['ARI_shared'].min():.3f}")
        print(f"  ARI between seeds, non-residential:    mean {ari['ARI_nonresidential'].mean():.3f}"
              f"  min {ari['ARI_nonresidential'].min():.3f}")
    print(f"\n  tables -> {out}")
    print(f"  cache left on the configured seed {main_seed}")
    print(f"  total {time.time()-t_all:.0f}s\n")


if __name__ == "__main__":
    main()
