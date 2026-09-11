"""Configuration loader.

The YAML is the single source of truth for every parameter that the paper
declares. Nothing in the pipeline hard-codes a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

#Lorenzo Giannuzzo: one folder per stage of the framework figure, in pipeline order. The
#numeric prefixes are what keep that order in a file browser, which the stage names alone
#would not: alphabetically the comparison would come first and the pre-processing fourth.
#A reader holding the paper open should reach the artefact behind any statement by walking
#the tree in the order the method is described.
STAGE_FOLDERS: dict[str, str] = {
    "preprocessing": "1_preprocessing",
    "clustering": "2_clustering",
    "generation": "3_standard_lp_generation",
    "comparison": "4_lp_comparison",
    "mapping": "5_lp_mapping",
}

#Lorenzo Giannuzzo: the mapping stage splits further because the paper reports its three
#metrics in three separate sections, and a reader following Section 3.3 should not have to
#pick the aggregation tables out of a directory holding all three.
MAPPING_PARTS: tuple[str, ...] = ("multiplicity", "aggregation", "coverage")


@dataclass
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted access: cfg.get('clustering.sample_size')."""
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # ── resolved paths ───────────────────────────────────────────────────────
    @property
    def data_dir(self) -> Path:
        return (ROOT / self.raw["data"]["root"]).resolve()

    @property
    def cache_dir(self) -> Path:
        p = ROOT / self.raw["output"]["cache_dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def results_dir(self, stage: str, part: str | None = None) -> Path:
        """Results folder of a stage, created on demand.

        A stage outside the framework keeps the old `<stage>_results` name rather than
        raising, so that a helper or an exploratory script that writes somewhere of its
        own does not have to be registered here to run.
        """
        base = ROOT / self.raw["output"]["results_dir"]
        p = base / STAGE_FOLDERS.get(stage, f"{stage}_results")
        if part is not None:
            if stage != "mapping" or part not in MAPPING_PARTS:
                raise KeyError(f"{part!r} is not a part of stage {stage!r}; "
                               f"expected one of {MAPPING_PARTS} under 'mapping'")
            p = p / part
        p.mkdir(parents=True, exist_ok=True)
        return p

    def figures_dir(self, stage: str, part: str | None = None,
                    fmt: str | None = None) -> Path:
        """Where the figures of a stage go, one level below its tables.

        Each format sits in its own sub-folder. A manuscript pulls the PNGs and a
        camera-ready submission pulls the PDFs, and keeping them apart means either can
        be selected, zipped or ignored as a whole rather than by extension.
        """
        p = self.results_dir(stage, part) / "figures"
        if fmt is not None:
            p = p / fmt
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def year(self) -> int:
        return int(self.raw["data"]["year"])


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))