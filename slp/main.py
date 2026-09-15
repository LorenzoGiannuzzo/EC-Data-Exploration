"""SLP framework — pipeline runner.

    python main.py                     every stage, in order
    python main.py --stage preprocessing
    python main.py --from clustering   from that stage onwards
    python main.py --config alt.yaml
    python main.py --stage numbers     every number of the paper in one table

Each stage reads the cache the previous one wrote, so re-running a late stage
after changing a parameter does not recompute the dictionary.

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
import os

#Lorenzo Giannuzzo: joblib cannot count the physical cores on this Windows machine and prints a
# traceback every run; the logical count is what it falls back to anyway
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#Lorenzo Giannuzzo: stage name -> (module, paper section)
STAGES: dict[str, tuple[str, str]] = {
    "preprocessing": ("preprocessing", "2.2"),
    "clustering":    ("clustering",    "2.3"),
    "generation":    ("generation",    "2.4"),
    "comparison":    ("comparison",    "2.5"),
    "mapping":       ("mapping",       "2.6"),
    "numbers":       ("numerical_results", "3"),
    "figures":       ("figures",       "2.5"),
}


def run_stage(name: str) -> float:
    mod_name, section = STAGES[name]
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        #Lorenzo Giannuzzo: Only a missing stage module means the stage is not written yet. A stage that
        # exists but fails to import because one of *its* dependencies is missing must
        # raise: reporting that as "not implemented" would let a broken stage be silently
        # skipped while the pipeline reports success and the cache keeps stale results.
        if exc.name != mod_name:
            raise
        print(f"\n  [{name}] not implemented yet (Section {section}) — skipped\n")
        return 0.0
    t0 = time.time()
    mod.main()
    return time.time() - t0


def apply_config_override(path: str) -> None:
    """Point every stage at an alternative configuration file.

    The stages import load_config by name when they are imported, so rebinding the
    function inside common.config reaches none of them. The path is handed over through
    the environment instead, which common.config.load_config reads whenever it is called
    without an explicit path, and it is set before the first stage is imported.
    """
    import os
    os.environ["SLP_CONFIG"] = str(Path(path).resolve())
    print(f"  configuration overridden: {path}")


def print_timings(timings: list[tuple[str, float]], failed: str | None = None) -> None:
    print(f"\n{'#'*78}\n#  TIMINGS")
    for s, t in timings:
        if t:
            print(f"#    {s:16s} {t:7.1f} s")
    print(f"#    {'total':16s} {sum(t for _, t in timings):7.1f} s")
    if failed:
        print(f"#  FAILED during: {failed}")
    print(f"{'#'*78}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="SLP framework pipeline")
    ap.add_argument("--stage", choices=list(STAGES), help="run one stage only")
    ap.add_argument("--from", dest="from_stage", choices=list(STAGES),
                    help="run from this stage to the end")
    ap.add_argument("--config", default=None, help="path to an alternative config.yaml")
    args = ap.parse_args()

    if args.config:
        if not Path(args.config).exists():
            ap.error(f"configuration file not found: {args.config}")
        apply_config_override(args.config)

    order = list(STAGES)
    if args.stage:
        todo = [args.stage]
    elif args.from_stage:
        todo = order[order.index(args.from_stage):]
    else:
        todo = order

    print(f"\n{'#'*78}\n#  SLP framework — {len(todo)} stage(s): {', '.join(todo)}\n{'#'*78}")
    timings: list[tuple[str, float]] = []
    for s in todo:
        try:
            timings.append((s, run_stage(s)))
        except Exception:
            #Lorenzo Giannuzzo: The traceback is left to propagate. What is added here is the timing of the
            # stages that did complete, which is otherwise lost and is the first thing
            # wanted when a long run dies halfway.
            print_timings(timings, failed=s)
            raise

    print_timings(timings)


if __name__ == "__main__":
    main()