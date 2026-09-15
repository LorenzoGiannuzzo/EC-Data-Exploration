"""Selection of K on representativeness, read back from the clustering stage.

The sweep itself runs inside clustering.py (sweep_K_dispersion), on the same tree
the run uses, and writes 2_clustering/validity_K_dispersion.csv. This script only
reads that table and prints the marginal gain per added group, so that the knee
can be judged without re-running the clustering.

    python sweep_k_dispersion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config  # noqa: E402


def report_knee(table: pd.DataFrame, column: str = "nrmsd_p50_pod_weighted") -> None:
    #Lorenzo Giannuzzo: the gain per added group, read as a marginal gain rather than as a
    # fitted curvature, since the sweep has fewer than twenty points
    values = table[column].to_numpy()
    ks = table["K"].to_numpy()
    gains = -np.diff(values)
    print(f"\nmarginal gain in {column} per added group")
    for i, gain in enumerate(gains):
        share = 100.0 * gain / values[i] if values[i] else float("nan")
        print(f"  {ks[i]:>3} -> {ks[i + 1]:<3}  {gain:+.4f}  ({share:+.1f}%)")
    print("\nunweighted and pod-weighted columns must be read together: a K where the "
          "unweighted figure improves and the weighted one does not is adding small "
          "tight groups without representing more of the population.")


def main() -> None:
    path = load_config().results_dir("clustering") / "validity_K_dispersion.csv"
    if not path.exists():
        print(f"  {path} not found: run  python main.py --stage clustering  first")
        return
    table = pd.read_csv(path)
    print(table.round(3).to_string(index=False))
    report_knee(table)


if __name__ == "__main__":
    main()
