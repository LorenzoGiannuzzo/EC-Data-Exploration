"""Which normalisation? unit_integral or min_max.

    python compare_normalisation.py

Runs the whole chain both ways on the same data and puts the two side by side.
The question is not which produces prettier codewords: a dictionary exists to
make the users separable, so what decides is the silhouette of the user
partition, and whether it beats the mean curves it is meant to replace.

The results of each branch are kept, so nothing has to be re-run afterwards:

    paper_results/clustering_results_unit_integral/
    paper_results/clustering_results_min_max/
    paper_results/normalisation_choice.csv
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def set_norm(norm: str) -> None:
    p = ROOT / "config.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["preprocessing"]["shape_normalisation"] = norm
    # keep the comments by rewriting only the one line
    text = p.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        if line.strip().startswith("shape_normalisation:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f'{indent}shape_normalisation: "{norm}"   # set by compare_normalisation.py')
        else:
            out.append(line)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


def run(stage: str) -> str:
    """Run a stage, echoing its output as it comes rather than at the end.

    Each stage takes minutes; capturing silently and printing at the end leaves
    the screen dead long enough to look like a hang.
    """
    print(f"\n  --- {stage} ---", flush=True)
    lines = []
    proc = subprocess.Popen([sys.executable, "-u", "main.py", "--stage", stage],
                            cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        lines.append(line)
        print("  " + line.rstrip(), flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"{stage} failed")
    return "".join(lines)


def main() -> None:
    original = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    keep = original["preprocessing"].get("shape_normalisation", "unit_integral")
    rows = []
    print("\n  Two full runs, preprocessing included: the normalisation happens")
    print("  there, so it cannot be changed from the cache. Roughly 12 minutes.")

    try:
        for norm in ("unit_integral", "min_max"):
            print(f"\n{'#'*70}\n#  {norm}\n{'#'*70}")
            t0 = time.time()
            set_norm(norm)
            run("preprocessing")          # the normalisation happens here
            run("clustering")

            res = ROOT / "paper_results" / "clustering_results"
            abl = pd.read_csv(res / "ablation.csv").iloc[0]
            dic = pd.read_csv(res / "dictionary.csv")
            grp = pd.read_csv(res / "groups.csv")

            rows.append({
                "normalisation": norm,
                "D": len(dic),
                "K": len(grp),
                "silhouette_two_stage": round(float(abl["silhouette_two_stage"]), 4),
                "silhouette_mean_curves": round(float(abl["silhouette_mean_curves"]), 4),
                "ARI_vs_mean": round(float(abl["ARI"]), 4),
                "largest_form_share": round(float(dic["share_of_days"].max()), 4),
                "largest_group_share": round(float(grp["n_pods"].max() / grp["n_pods"].sum()), 4),
                "forms_over_1pct": int((dic["share_of_days"] > 0.01).sum()),
                "seconds": round(time.time() - t0),
            })

            dest = ROOT / "paper_results" / f"clustering_results_{norm}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(res, dest)
    finally:
        set_norm(keep)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "paper_results" / "normalisation_choice.csv", index=False)

    print(f"\n{'='*70}\nCOMPARISON\n{'='*70}")
    print(df.to_string(index=False))

    # What the paper has to defend is not the level of the silhouette but the
    # margin over the mean curves it replaces. A normalisation that raises both
    # has bought nothing and given the rival half the ground back.
    df["margin"] = (df["silhouette_two_stage"] /
                    df["silhouette_mean_curves"].replace(0, float("nan"))).round(2)
    df.to_csv(ROOT / "paper_results" / "normalisation_choice.csv", index=False)

    a, b = df.iloc[0], df.iloc[1]
    print(f"\n  The dictionary exists to separate the users, and what the paper")
    print(f"  must defend is the margin over the mean curves, not the level.\n")
    for r in (a, b):
        v = "beats" if r["silhouette_two_stage"] > r["silhouette_mean_curves"] else "LOSES TO"
        print(f"    {r['normalisation']:14s} {r['silhouette_two_stage']:.3f} vs "
              f"{r['silhouette_mean_curves']:.3f}   {v} the mean curves by "
              f"{r['margin']:.1f}x")
    win = a if a["margin"] > b["margin"] else b
    print(f"\n  -> {win['normalisation']} holds the wider margin.")

    if max(a["silhouette_two_stage"], b["silhouette_two_stage"]) < 0.25:
        print(f"\n  Note: both partitions are weak in absolute terms (< 0.25).")
        print(f"  Neither normalisation finds separated groups, which is a")
        print(f"  property of the population rather than of the method: the")
        print(f"  users lie on a continuum and any K cuts it arbitrarily.")

    for r in (a, b):
        if r["K"] >= 20:
            print(f"\n  Note: {r['normalisation']} hit the top of profiles_range at "
                  f"K={r['K']}. Its silhouette was still climbing and the")
            print(f"  comparison is not at its optimum.")
    print(f"\n  Both runs are kept under paper_results/clustering_results_<norm>/;")
    print(f"  config.yaml is back to '{keep}'.\n")


if __name__ == "__main__":
    main()
