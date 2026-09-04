"""Configuration access.

One YAML file, read once, addressed with dotted keys so that a caller asking for
`data.meas_cols.pod` does not have to know how the file is nested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config_synthgen.yaml"


class Config:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_CONFIG
        if not self.path.exists():
            raise SystemExit(f"\n  configuration not found: {self.path}\n")
        with open(self.path, "r", encoding="utf-8") as fh:
            self._d = yaml.safe_load(fh) or {}

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    #Lorenzo Giannuzzo: ── resolved paths ───────────────────────────────────────────────────────
    @property
    def data_dir(self) -> Path:
        return (ROOT / str(self.get("data.root", "data"))).resolve()

    @property
    def cache_dir(self) -> Path:
        p = ROOT / str(self.get("output.cache_dir", "cache/synthgen"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_dir(self) -> Path:
        #Lorenzo Giannuzzo: the estimated models sit beside `results`, not inside
        # it. They are the one artefact of the pipeline that is
        # expensive to rebuild and that a delivery may ship on
        # its own, so emptying `results` must never remove them.
        p = ROOT / str(self.get("output.models_dir", "models"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def results_dir(self, sub: str = "") -> Path:
        p = ROOT / str(self.get("output.results_dir", "results"))
        if sub:
            p = p / sub
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: Path | str | None = None) -> Config:
    return Config(path)