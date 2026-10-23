"""v3_crosshop configuration: extends v2 Config with CrossHopConfig."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from v2_redesign.config import (
    PathsConfig, IOConfig, PreprocConfig, ModelConfig,
    FGWConfig, CandidateConfig, TrainConfig, InferConfig, Config as V2Config,
)


@dataclass
class CrossHopConfig:
    """Configuration for cross-species direct edges."""
    cross_hop_k: int = 32              # top-k mouse neighbors per human gene per modality
    modalities: List[str] = field(
        default_factory=lambda: ["BP", "MF", "Disease"]
    )
    weight_scale: float = 1.0          # multiply cross-hop edge weights by this factor
    min_weight_threshold: float = 1e-6 # drop edges with P[i,j] below this


@dataclass
class V3Config(V2Config):
    """v3 config: all v2 fields + crosshop section."""
    crosshop: CrossHopConfig = field(default_factory=CrossHopConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "V3Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        # Load all v2 sections
        for section_name in [
            "paths", "io", "preproc", "model", "fgw",
            "candidates", "train", "infer",
        ]:
            if section_name in raw:
                section = getattr(cfg, section_name)
                for k, v in raw[section_name].items():
                    if hasattr(section, k):
                        setattr(section, k, v)
        # Load crosshop section
        if "crosshop" in raw:
            for k, v in raw["crosshop"].items():
                if hasattr(cfg.crosshop, k):
                    setattr(cfg.crosshop, k, v)
        return cfg
