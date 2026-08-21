"""Strict paper configuration loading with stable content hashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    dataset: str
    variant: str
    input_dim: int
    modalities: int
    classes: int
    dense_width: int
    protocol: str
    scope: str
    frozen_outputs: str
    required_inputs: tuple[str, ...]
    allowed_stages: tuple[str, ...]
    backbone_optimizer: Mapping[str, Any]
    head_optimizer: Mapping[str, Any]
    seed: int

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, path: Path | None = None) -> "ConfigSpec":
        required = (
            "name", "dataset", "variant", "input_dim", "modalities", "classes", "dense_width",
            "protocol", "scope", "frozen_outputs", "required_inputs", "training", "seed",
        )
        missing = [key for key in required if key not in row]
        if missing:
            where = f" in {path}" if path else ""
            raise ValueError(f"CONFIG_MISSING{where}:{','.join(missing)}")
        training = row["training"]
        if not isinstance(training, Mapping):
            raise ValueError("CONFIG_TRAINING_MAPPING_REQUIRED")
        allowed = tuple(str(item) for item in training.get("allowed_stages", ()))
        backbone_optimizer = dict(training.get("backbone_optimizer", {}))
        head_optimizer = dict(training.get("head_optimizer", {}))
        output = cls(
            name=str(row["name"]),
            dataset=str(row["dataset"]),
            variant=str(row["variant"]),
            input_dim=int(row["input_dim"]),
            modalities=int(row["modalities"]),
            classes=int(row["classes"]),
            dense_width=int(row["dense_width"]),
            protocol=str(row["protocol"]),
            scope=str(row["scope"]),
            frozen_outputs=str(row["frozen_outputs"]),
            required_inputs=tuple(str(item) for item in row["required_inputs"]),
            allowed_stages=allowed,
            backbone_optimizer=backbone_optimizer,
            head_optimizer=head_optimizer,
            seed=int(row["seed"]),
        )
        output.validate()
        return output

    def validate(self) -> None:
        if self.classes != 5:
            raise ValueError("CONFIG_CLASSES_MUST_BE_5")
        if self.input_dim not in (128, 320):
            raise ValueError(f"CONFIG_INPUT_DIM_UNSUPPORTED:{self.input_dim}")
        if self.modalities not in (4, 10):
            raise ValueError(f"CONFIG_MODALITIES_UNSUPPORTED:{self.modalities}")
        if self.variant != "linear_bim":
            raise ValueError(f"CONFIG_VARIANT_UNSUPPORTED:{self.variant}")
        if self.dense_width != 64:
            raise ValueError("CONFIG_DENSE_WIDTH_MUST_BE_64")
        if self.input_dim == 128 and self.modalities != 4:
            raise ValueError("CONFIG_S20_S78_MODALITY_MISMATCH")
        if self.input_dim == 320 and self.modalities != 10:
            raise ValueError("CONFIG_S3_MODALITY_MISMATCH")
        if "head" not in self.allowed_stages:
            raise ValueError("CONFIG_HEAD_STAGE_REQUIRED")
        for optimizer_name, optimizer in (("backbone", self.backbone_optimizer), ("head", self.head_optimizer)):
            if str(optimizer.get("name", "Adam")) != "Adam":
                raise ValueError(f"CONFIG_{optimizer_name.upper()}_OPTIMIZER_UNSUPPORTED")
            if float(optimizer.get("eps", -1)) <= 0:
                raise ValueError(f"CONFIG_{optimizer_name.upper()}_EPS_REQUIRED")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "variant": self.variant,
            "input_dim": self.input_dim,
            "modalities": self.modalities,
            "classes": self.classes,
            "dense_width": self.dense_width,
            "protocol": self.protocol,
            "scope": self.scope,
            "frozen_outputs": self.frozen_outputs,
            "required_inputs": list(self.required_inputs),
            "training": {
                "allowed_stages": list(self.allowed_stages),
                "backbone_optimizer": dict(self.backbone_optimizer),
                "head_optimizer": dict(self.head_optimizer),
            },
            "seed": self.seed,
        }


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:  # YAML files used here are also JSON-compatible.
        return json.loads(text)
    value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError(f"CONFIG_MAPPING_REQUIRED:{path}")
    return value


def load_config(path: Path | str) -> ConfigSpec:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CONFIG_MISSING:{path}")
    return ConfigSpec.from_mapping(_load_mapping(path), path=path)


def config_hash(config: ConfigSpec | Mapping[str, Any]) -> str:
    value = config.as_dict() if isinstance(config, ConfigSpec) else dict(config)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()
