"""SLP framework — pipeline runner.

    python main.py                     every stage, in order
    python main.py --stage preprocessing
    python main.py --from clustering   from that stage onwards
    python main.py --config alt.yaml

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
    """Point common.config at an alternative configuration file.

    The original function is captured before the module attribute is rebound. Rebinding
    first and reaching for the original through the same name afterwards would find the
    replacement instead, so the call would either fail or recurse. `__wrapped__` is
    consulted in case the loader is cached, and the cache is cleared so that a value read
    from the default file earlier in the process is not served again.
    """
    import common.config as C

    original = getattr(C.load_config, "__wrapped__", C.load_config)
    cache_clear = getattr(C.load_config, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()

    def load_config(p: str = path):
        return original(p)

    C.load_config = load_config  # type: ignore[assignment]
    print(f"  configuration overridden: {path}")
    #Lorenzo Giannuzzo: A stage that did `from common.config import load_config` at import time holds its own
    # reference and is unaffected. Stages in this pipeline call `config.load_config()`
    # through the module, which is what makes the override work.


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