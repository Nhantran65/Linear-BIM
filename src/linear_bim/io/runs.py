"""Immutable frozen-output and run-directory helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    if not RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(f"RUN_ID_INVALID:{run_id}")
    return run_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenOutputs:
    path: Path
    arrays: dict[str, np.ndarray]
    sha256: str


def load_frozen_outputs(path: Path | str) -> FrozenOutputs:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"FROZEN_SYMLINK_NOT_ALLOWED:{path}")
    if not path.is_file():
        raise FileNotFoundError(f"FROZEN_OUTPUTS_MISSING:{path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    required = ("labels",)
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"FROZEN_OUTPUTS_SCHEMA_MISSING:{','.join(missing)}")
    for canonical, historical in (("u0_logits", "ulw_logits"), ("b0_logits", "bim_logits")):
        has_canonical, has_historical = canonical in arrays, historical in arrays
        if not has_canonical and not has_historical:
            raise ValueError(f"FROZEN_OUTPUTS_SCHEMA_MISSING:{canonical}|{historical}")
        if has_canonical and has_historical:
            if arrays[canonical].shape != arrays[historical].shape or not np.array_equal(arrays[canonical], arrays[historical]):
                raise ValueError(f"FROZEN_OUTPUTS_CONTRADICTORY_ALIAS:{canonical}:{historical}")
        if not has_canonical:
            arrays[canonical] = arrays[historical]
    labels = arrays["labels"]
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("FROZEN_LABELS_SCHEMA")
    for key in ("u0_logits", "b0_logits"):
        if arrays[key].shape != (len(labels), 5):
            raise ValueError(f"FROZEN_LOGITS_SCHEMA:{key}")
    if not np.all(np.isfinite(arrays["u0_logits"])) or not np.all(np.isfinite(arrays["b0_logits"])):
        raise ValueError("FROZEN_LOGITS_NONFINITE")
    if not np.all((labels >= 0) & (labels < 5)):
        raise ValueError("FROZEN_LABELS_RANGE")
    if "subjects" in arrays and len(arrays["subjects"]) != len(labels):
        raise ValueError("FROZEN_SUBJECT_IDENTITY")
    if "epoch_indices" in arrays and len(arrays["epoch_indices"]) != len(labels):
        raise ValueError("FROZEN_EPOCH_IDENTITY")
    return FrozenOutputs(path, arrays, sha256_file(path))


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    config_hash: str
    protocol: str
    stage: str
    input_hashes: dict[str, str]
    output: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "protocol": self.protocol,
            "stage": self.stage,
            "input_hashes": dict(sorted(self.input_hashes.items())),
        }


def create_run(
    root: Path | str,
    run_id: str,
    *,
    config_hash_value: str,
    protocol: str,
    stage: str,
    input_paths: tuple[Path, ...] = (),
    resume: bool = False,
) -> tuple[Path, RunManifest]:
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError("RUN_ROOT_SYMLINK_NOT_ALLOWED")
    root = root_path.resolve()
    if root.name != "runs" or any(part in {"frozen", "archive", "legacy"} for part in root.parts):
        raise ValueError(f"RUN_ROOT_NOT_CANONICAL:{root}")
    run_id = validate_run_id(run_id)
    output = (root / run_id).resolve()
    if root not in output.parents:
        raise ValueError("RUN_ROOT_ESCAPE")
    input_hashes = {str(Path(path).resolve()): sha256_file(Path(path)) for path in input_paths}
    manifest = RunManifest(run_id, config_hash_value, protocol, stage, input_hashes, output)
    manifest_path = output / "run.json"
    if resume:
        raise RuntimeError("RUN_RESUME_UNSUPPORTED: optimizer/RNG restore is not implemented")
    if output.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"RUN_EXISTS_USE_VALIDATED_RESUME:{output}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("config_hash") != config_hash_value
            or existing.get("input_hashes") != manifest.as_dict()["input_hashes"]
            or existing.get("run_id") != run_id
            or existing.get("protocol") != protocol
            or existing.get("stage") != stage
        ):
            raise ValueError("RUN_RESUME_HASH_MISMATCH")
        raise RuntimeError("RUN_RESUME_UNSUPPORTED: optimizer/RNG restore is not implemented")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, manifest
