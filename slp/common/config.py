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

    def results_dir(self, stage: str) -> Path:
        """paper_results/<stage>_results/, created on demand."""
        p = ROOT / self.raw["output"]["results_dir"] / f"{stage}_results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def year(self) -> int:
        return int(self.raw["data"]["year"])


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
