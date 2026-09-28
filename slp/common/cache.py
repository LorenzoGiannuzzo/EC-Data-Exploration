"""What configuration the cache was written by.

The stages communicate through fixed file names under cache/: the dictionary,
the codeword of every day, the user vectors, the groups. Nothing in those names
records the configuration that produced them, so a run with one first-stage unit
leaves behind files a later run with another unit will read without complaint,
and the comparison and mapping stages can report numbers belonging to a
dictionary that no longer exists.

Renaming the files per configuration would solve it and would also break every
stage that reads them by name. What is done instead is to record, next to the
files, the parameters they depend on, and to have each stage check that record
before trusting what it finds. The check is cheap, it fails loudly, and it
leaves the layout of cache/ untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST = "manifest.json"

#Lorenzo Giannuzzo: the parameters a cached artefact depends on. Anything that
# changes the content of dictionary.npy or of the vectors built on it belongs
# here; anything that only changes how a result is reported does not.
KEYS = (
    "shape_unit",
    "dictionary_resolution",
    "dictionary_scope",
    "dictionary_method",
    "n_micro",
    "n_codewords",
    "scale_weight",
    "block_normalisation",
    "zero_replacement",
    "n_profiles",
)


def signature(cl: dict, resolved: dict | None = None) -> dict:
    """The configuration of the clustering stage, reduced to what the cache
    depends on. `resolved` carries the values the run actually settled on, which
    for D and K differ from the configuration whenever they were left to be
    chosen."""
    sig = {k: cl.get(k) for k in KEYS}
    sig.update(resolved or {})
    return sig


def write_manifest(cache_dir: Path, sig: dict) -> None:
    with open(Path(cache_dir) / MANIFEST, "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2, sort_keys=True, default=str)


def read_manifest(cache_dir: Path) -> dict | None:
    path = Path(cache_dir) / MANIFEST
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_manifest(cache_dir: Path, cl: dict, stage: str,
                   strict: bool = True) -> dict | None:
    """Verify that the cache was written by the configuration now in force.

    Raises by default: a stage that reports numbers built on another
    configuration is worse than a stage that refuses to run, since the first
    failure is silent and reaches the paper.
    """
    man = read_manifest(cache_dir)
    if man is None:
        msg = (f"{stage}: cache/ carries no manifest, so what wrote it is "
               f"unknown. Run the clustering stage before this one.")
        if strict:
            raise FileNotFoundError(msg)
        print(f"  WARNING: {msg}")
        return None

    want = signature(cl)
    differ = {k: (man.get(k), want[k]) for k in want
              if k not in ("n_codewords", "n_profiles")
              and str(man.get(k)) != str(want[k])}
    if differ:
        lines = "\n".join(f"      {k}: cache {a!r}, config {b!r}"
                          for k, (a, b) in sorted(differ.items()))
        msg = (f"{stage}: cache/ was written by a different configuration:\n"
               f"{lines}\n    Re-run the clustering stage.")
        if strict:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")
    return man
