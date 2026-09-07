"""Working configuration plumbing. Numerical implementation lives elsewhere."""

from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path
from typing import Optional

PREDICTIONS = ("x", "eps", "v")
PAPER_DIMENSIONS = (2, 8, 16, 512)


@dataclass(frozen=True)
class ExperimentConfig:
    # Paper settings. Five layers means five Linear layers in this scaffold.
    observed_dim: int = 2
    intrinsic_dim: int = 2
    prediction: str = "x"
    hidden_dim: int = 256
    linear_layers: int = 5
    # Explicit practice defaults, NOT recovered toy hyperparameters.
    dataset: str = "spiral"
    train_size: int = 8192
    eval_size: int = 2048
    spiral_turns: float = 2.0
    spiral_radius: float = 2.0
    seed: int = 0
    projection_seed: int = 123
    device: str = "cpu"
    batch_size: int = 256
    train_steps: int = 10000
    learning_rate: float = 0.001
    log_every: int = 100
    # Borrow main_jit.py's P_mean/P_std; toy-specific values are unspecified.
    time_mean: float = -0.8
    time_std: float = 0.8
    time_eps: float = 0.001
    solver: str = "heun"
    sampling_steps: int = 50

    def __post_init__(self) -> None:
        positive_ints = (
            "observed_dim", "intrinsic_dim", "hidden_dim", "linear_layers",
            "train_size", "eval_size", "batch_size", "train_steps", "log_every",
            "sampling_steps",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "projection_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**63 - 4:
                raise ValueError(f"{name} must be an integer in [0, 2**63 - 4)")
        for name in ("spiral_turns", "spiral_radius", "learning_rate", "time_std", "time_eps"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if type(self.time_mean) not in (int, float) or not math.isfinite(self.time_mean):
            raise ValueError("time_mean must be finite")
        if not self.time_eps < 0.5:
            raise ValueError("time_eps must be less than 0.5")
        if self.intrinsic_dim != 2 or self.observed_dim < self.intrinsic_dim:
            raise ValueError("the spiral requires intrinsic_dim=2 and observed_dim>=2")
        if self.linear_layers < 2:
            raise ValueError("linear_layers must be at least 2")
        for name, choices in (
            ("prediction", PREDICTIONS), ("dataset", ("spiral",)),
            ("device", ("cpu", "mps", "cuda")), ("solver", ("euler", "heun")),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Optional[Path] = None) -> ExperimentConfig:
    """Load partial JSON overrides, rejecting misspelled or unknown fields."""
    if path is None:
        return ExperimentConfig()
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = set(values) - {field.name for field in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    return ExperimentConfig(**values)


def experiment_matrix(base: ExperimentConfig) -> list[ExperimentConfig]:
    """Twelve independent runs, sharing seeds and budget across prediction types."""
    return [
        replace(base, observed_dim=dim, prediction=prediction)
        for dim in PAPER_DIMENSIONS for prediction in PREDICTIONS
    ]
