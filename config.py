import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Hyperparameter Domains
TEXT_FEATURE_DIM_DOMAIN: List[int] = [64, 128, 256]
IMAGE_FEATURE_DIM_DOMAIN: List[int] = [64, 128, 256]
HIDDEN_DIM_DOMAIN: List[int] = [32, 64, 128]
FUSION_TYPE_DOMAIN: List[int] = [0, 1, 2]  # 0: concat, 1: gated, 2: late
DROPOUT_DOMAIN: List[float] = [0.0, 0.1, 0.2, 0.3]
LEARNING_RATE_DOMAIN: List[float] = [0.001, 0.0005, 0.0001]
WEIGHT_DECAY_DOMAIN: List[float] = [0.0, 0.0001, 0.001]
MODALITY_WEIGHT_DOMAIN: List[float] = [0.25, 0.5, 0.75]
ROBUSTNESS_STRENGTH_DOMAIN: List[float] = [0.0, 0.05, 0.10, 0.15]
EARLY_STOPPING_PATIENCE_DOMAIN: List[int] = [3, 5, 8]

PARAM_DOMAINS: List[List[Any]] = [
    TEXT_FEATURE_DIM_DOMAIN,
    IMAGE_FEATURE_DIM_DOMAIN,
    HIDDEN_DIM_DOMAIN,
    FUSION_TYPE_DOMAIN,
    DROPOUT_DOMAIN,
    LEARNING_RATE_DOMAIN,
    WEIGHT_DECAY_DOMAIN,
    MODALITY_WEIGHT_DOMAIN,
    ROBUSTNESS_STRENGTH_DOMAIN,
    EARLY_STOPPING_PATIENCE_DOMAIN,
]

GENE_NAMES: List[str] = [
    "text_feature_dim",
    "image_feature_dim",
    "hidden_dim",
    "fusion_type",
    "dropout",
    "learning_rate",
    "weight_decay",
    "modality_weight",
    "robustness_strength",
    "early_stopping_patience",
]

FUSION_TYPE_MAP: Dict[int, str] = {0: "concat", 1: "gated", 2: "late"}


@dataclass
class EvoForgeConfig:
    data_root: str = "data/crisismmd"
    random_seed: int = 42
    population_size: int = 8
    generations: int = 5
    elite_count: int = 1
    initial_mutation_rate: float = 0.20
    max_archive_size: int = 10
    runtime_budget_sec: float = 30.0
    ood_event_name: str = "hurricane_irma"
    synthetic_test_mode: bool = False
    daam_target_low: float = 0.20
    daam_target_high: float = 0.50
    mutation_scale_up: float = 1.35
    mutation_scale_down: float = 0.85
    min_mutation_rate: float = 0.05
    max_mutation_rate: float = 0.50

    @classmethod
    def load_from_json(cls, json_path: str) -> "EvoForgeConfig":
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            return cls(**data)
        return cls()
