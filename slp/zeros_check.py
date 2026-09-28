"""Diagnostic, not a stage.

Does the partition of the users follow the mixture of forms, or the pattern of
structural zeros in it?

A frequency vector is compositional and most of its coordinates are empty: a user
never realizes every codeword of the dictionary. Those zeros are replaced before
the CLR by a constant delta, and after the transform they all collapse onto the
same value within a user, namely the minimum of that user's vector. The more of
them a user carries, the further its vector sits from a user carrying few, for a
reason that has nothing to do with behaviour.

The concern is specific to the unit of the first stage. Over a year a user
distributes some three hundred days across the dictionary and fills nearly every
coordinate, so there is little for the replacement to govern. Over months it
distributes eighteen or nineteen observations across twenty codewords, so most
coordinates are structurally empty and the replacement decides a large part of the
geometry. If the groups line up with the number of empty coordinates, the
partition is reading how many months a POD happens to have in the archive rather
than how it consumes.

Reads only the cache the clustering stage already wrote. Run from slp/.

    python zeros_check.py

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

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.config import load_config  # noqa: E402

TOL = 1e-9


def clr_columns(v: pd.DataFrame) -> list[str]:
    """The CLR block, whatever it is called in this build."""
    for pat in ("clr", "code", "form", "f_"):
        cols = [c for c in v.columns if str(c).lower().startswith(pat)]
        if len(cols) >= 4:
            return cols
    num = v.select_dtypes("number")
    return list(num.columns)


def main() -> None:
    cfg = load_config()
    cache = cfg.cache_dir
    v = pd.read_parquet(cache / "user_vectors.parquet")
    g = pd.read_parquet(cache / "groups.parquet")
    unit = str(cfg.get("clustering.shape_unit", "day"))
    delta = cfg.get("clustering.zero_replacement")

    cols = clr_columns(v)
    X = v[cols].to_numpy(dtype="float64")
    D = X.shape[1]
    print(f"\n{'=' * 78}\nSTRUCTURAL ZEROS\n{'=' * 78}\n")
    print(f"  first-stage unit  {unit}     delta {delta}")
    print(f"  {len(v):,} users, {D} coordinates read from "
          f"{cols[0]}..{cols[-1]}")

    #Lorenzo Giannuzzo: a replaced zero lands on the row minimum, and every replaced zero of a user
    # lands on the same value, so counting the ties at the minimum counts them
    lo = X.min(axis=1, keepdims=True)
    n_zero = (np.abs(X - lo) <= TOL).sum(axis=1)
    filled = D - n_zero
    print(f"\n  coordinates actually realized per user")
    print(f"    median {np.median(filled):.0f} of {D}    "
          f"min {filled.min()}    max {filled.max()}")
    print(f"    users with more than half the vector empty: "
          f"{int((filled <= D / 2).sum()):,} "
          f"({100 * (filled <= D / 2).mean():.0f}%)")

    if filled.std() == 0:
        print("\n  every user realizes the same number of codewords; the "
              "replacement cannot be driving the geometry.\n")
        return

    key = "pod" if "pod" in v.columns else v.columns[0]
    d = pd.DataFrame({"pod": v[key].astype(str), "filled": filled})
    d = d.merge(g[["pod", "group"]].assign(pod=lambda x: x["pod"].astype(str)),
                on="pod", how="inner")

    print(f"\n  realized coordinates, by group")
    tab = d.groupby("group")["filled"].agg(["size", "median", "min", "max"])
    print(tab.to_string())

    #Lorenzo Giannuzzo: how much of the variation in the count the partition explains: if the
    # groups were blind to it this would sit near zero
    grand = d["filled"].to_numpy()
    ss_tot = float(((grand - grand.mean()) ** 2).sum())
    ss_within = float(sum(((x["filled"] - x["filled"].mean()) ** 2).sum()
                          for _, x in d.groupby("group")))
    eta2 = 1.0 - ss_within / ss_tot if ss_tot > 0 else np.nan
    print(f"\n  share of the variation in the count explained by the groups: "
          f"{eta2:.3f}")
    if eta2 >= 0.25:
        print("    -> the partition is reading the pattern of empty coordinates")
        print("       to a degree that has to be reported, not assumed away.")
    elif eta2 >= 0.10:
        print("    -> partly. Worth varying delta before the result is relied on.")
    else:
        print("    -> the groups are not separating users by how empty their")
        print("       vectors are; the geometry rests on the mixture itself.")
    print()


if __name__ == "__main__":
    main()