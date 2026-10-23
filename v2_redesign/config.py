"""Typed configuration using dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class PathsConfig:
    raw_dir: str = "dataset"
    proc_dir: str = "data/processed"


@dataclass
class IOConfig:
    human_gobp: str = "human_GOBP_ensembl.csv"
    human_gomf: str = "human_GOMF_ensembl.csv"
    mouse_gobp: str = "mouse_GOBP_ensembl.csv"
    mouse_gomf: str = "mouse_GOMF_ensembl.csv"
    human_dis: str = "HumanDiseaseGene.csv"
    mouse_dis: str = "MouseDiseaseGene.csv"


@dataclass
class PreprocConfig:
    go_roots: list = field(default_factory=lambda: ["GO:0008150", "GO:0003674", "GO:0005575"])
    disease_winsor_q: float = 0.995
    row_l2: bool = True
    svd_dim: int = 128
    heldout_ratio: float = 0.1


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.3
    edge_weight_blend: float = 0.1
    fusion_entropy_reg: float = 0.001


@dataclass
class FGWConfig:
    alpha: float = 0.5
    epsilon: float = 0.05
    max_iter: int = 50
    tau_row: float = 0.05
    tau_col: float = 0.15
    use_unbalanced: bool = True
    lambda_prior: float = 0.5
    lambda_memory: float = 0.3
    col_entropy_weight: float = 0.1
    cov_reg_weight: float = 0.01
    align_loss_weight: float = 0.0  # FGW-driven embedding alignment loss
    feature_weights: Dict[str, float] = field(
        default_factory=lambda: {"bp": 0.25, "mf": 0.15, "disease": 0.60}
    )


@dataclass
class CandidateConfig:
    topc: int = 256
    max_candidates: int = 2048
    soft_temperature: float = 0.1
    explore_ratio: float = 0.05
    T_init: float = 0.07
    mmr_lambda: float = 0.25
    svd_score_weight: float = 0.7


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip_grad: float = 2.0
    seed: int = 3407
    amp: bool = False
    val_every: int = 1
    save_every: int = 5
    memory_momentum: float = 0.9
    global_refresh_every: int = 5
    qk_lr_multiplier: float = 1.0
    hgt_emb_lr_multiplier: float = 0.1
    train_tau_col: float = 0.3
    residual_scale_init: float = -2.2  # sigmoid(-2.2)≈0.10, HGT starts as 10% refinement
    lr_schedule: str = "constant"  # "constant" or "cosine"
    fusion_mode: str = "attention"  # "attention" or "concat"


@dataclass
class InferConfig:
    topk_genes: int = 20
    topk_bp: int = 20
    topk_mf: int = 10
    topk_disease: int = 20
    fgw_refine_iters: int = 20


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    io: IOConfig = field(default_factory=IOConfig)
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fgw: FGWConfig = field(default_factory=FGWConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        for section_name, section_cls in [
            ("paths", PathsConfig), ("io", IOConfig), ("preproc", PreprocConfig),
            ("model", ModelConfig), ("fgw", FGWConfig), ("candidates", CandidateConfig),
            ("train", TrainConfig), ("infer", InferConfig),
        ]:
            if section_name in raw:
                section = getattr(cfg, section_name)
                for k, v in raw[section_name].items():
                    if hasattr(section, k):
                        setattr(section, k, v)
        return cfg

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
