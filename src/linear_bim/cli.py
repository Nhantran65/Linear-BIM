"""Explicit Linear-BIM command-line entry point.

Commands never download data or fall back to historical package paths. Full
backbone replay is intentionally unsupported until a reviewed recipe provides
the required inputs and protocol implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .data.npz import load_npz
from .data.padding import pad_by_sequence
from .evaluation.frozen import evaluate_frozen
from .explain.frozen import frozen_coordinate_explain
from .io.config import config_hash, load_config
from .io.runs import create_run, load_frozen_outputs, sha256_file as _sha256
from .models.backbone import ConfigurableLinearULW, parameter_ledger
from .models.bim import BIM, parameter_count
from .training.head import resolve_device, train_bim_head


REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(args: argparse.Namespace):
    return load_config(Path(args.config).resolve())


def _repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (REPO_ROOT / path).resolve()
    if REPO_ROOT not in resolved.parents and resolved != REPO_ROOT:
        raise ValueError(f"PATH_OUTSIDE_WORKSPACE:{value}")
    if any(part in {"archive", "legacy"} for part in resolved.parts):
        raise ValueError(f"PATH_LEGACY_NOT_SUPPORTED:{value}")
    return resolved


def _frozen_path(args: argparse.Namespace, config) -> Path:
    value = args.frozen or config.frozen_outputs
    path = _repo_path(value)
    if "frozen" not in path.parts:
        raise ValueError(f"FROZEN_PATH_MUST_BE_CANONICAL:{path}")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _effective_config_hash(config, args: argparse.Namespace) -> str:
    effective = {
        "config": config_hash(config),
        "stage": getattr(args, "stage", None),
        "fold": getattr(args, "fold", None),
        "seed": getattr(args, "effective_seed", None),
        "updates": getattr(args, "updates", None),
        "device": getattr(args, "device", None),
        "allow_cpu": getattr(args, "allow_cpu", None),
    }
    return hashlib.sha256(json.dumps(effective, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _emit(value: Any, output: str | None = None) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    if output is None:
        print(encoded, end="")
        return
    path = _repo_path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"OUTPUT_EXISTS:{path}")
    if any(part in {"frozen", "archive", "legacy"} for part in path.parts):
        raise ValueError(f"OUTPUT_PROTECTED:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _model_ledger(config) -> dict[str, Any]:
    backbone = ConfigurableLinearULW(config.seed, config.modalities)
    head = BIM(config.input_dim)
    backbone_row = parameter_ledger(backbone)
    complete = parameter_count(backbone) + parameter_count(head)
    expected = {
        (128, 4): (16_449, 3_112),
        (320, 10): (31_809, 6_184),
    }[(config.input_dim, config.modalities)]
    if complete != expected[0] or parameter_count(head) != expected[1]:
        raise RuntimeError(f"MODEL_LEDGER_FAIL:{complete}:{parameter_count(head)}")
    return {
        "backbone": backbone_row,
        "backbone_parameters": parameter_count(backbone),
        "head_parameters": parameter_count(head),
        "complete_parameters": complete,
        "expected_complete_parameters": expected[0],
        "expected_head_parameters": expected[1],
    }


def doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    missing = []
    for value in config.required_inputs:
        path = _repo_path(value)
        if not path.is_file():
            missing.append(str(path))
    frozen = _repo_path(config.frozen_outputs)
    input_ready = not missing
    evaluation_ready = frozen.is_file()
    result = {
        "status": ("FROZEN_EVALUATION_READY_HEAD_PAYLOAD_REQUIRED" if input_ready and evaluation_ready else "HEAD_INPUTS_READY_FROZEN_EVALUATION_BLOCKED" if input_ready else "INPUTS_MISSING"),
        "config": config.name,
        "config_hash": config_hash(config),
        "protocol": config.protocol,
        "scope": config.scope,
        "model": _model_ledger(config),
        "required_inputs": {str(_repo_path(value)): _repo_path(value).is_file() for value in config.required_inputs},
        "frozen_outputs": {"path": str(frozen), "present": frozen.is_file()},
        "readiness": {
            "frozen_evaluation": evaluation_ready,
            "audit": evaluation_ready,
            "head": "requires_explicit_train_npz_and_held_npz",
            "coordinate_explain": "requires_explicit_explain_input_npz_q_weight_bias",
            "full_backbone": False,
        },
        "supported_training_stages": list(config.allowed_stages),
        "full_backbone_training": "unsupported_until_reviewed_recipe",
    }
    _emit(result, args.output)
    return 0 if input_ready else 2


def audit(args: argparse.Namespace) -> int:
    config = _config(args)
    frozen = load_frozen_outputs(_frozen_path(args, config))
    arrays = frozen.arrays
    identity_pass = (
        ("subjects" not in arrays or len(arrays["subjects"]) == len(arrays["labels"]))
        and ("epoch_indices" not in arrays or len(arrays["epoch_indices"]) == len(arrays["labels"]))
        and ("folds" not in arrays or len(arrays["folds"]) == len(arrays["labels"]))
    )
    result = {
        "status": "SCHEMA_PASS" if identity_pass else "SCHEMA_FAIL",
        "evidence_audit": "not_a_substitute_for_historical_audit",
        "config": config.name,
        "config_hash": config_hash(config),
        "frozen_path": str(frozen.path),
        "frozen_sha256": frozen.sha256,
        "arrays": {key: {"dtype": str(value.dtype), "shape": list(value.shape)} for key, value in arrays.items()},
        "identity": {
            "subjects_present": "subjects" in arrays,
            "epoch_indices_present": "epoch_indices" in arrays,
            "folds_present": "folds" in arrays,
        },
    }
    _emit(result, args.output)
    return 0 if identity_pass else 2


def evaluate(args: argparse.Namespace) -> int:
    config = _config(args)
    result = evaluate_frozen(_frozen_path(args, config))
    result["config"] = config.name
    result["config_hash"] = config_hash(config)
    _emit(result, args.output)
    return 0


def explain(args: argparse.Namespace) -> int:
    config = _config(args)
    if not args.explain_input:
        raise ValueError("EXPLAIN_INPUT_REQUIRED:q/weight/bias NPZ")
    frozen = _frozen_path(args, config)
    payload = load_npz(Path(args.explain_input).resolve(), ("q", "weight", "bias"))
    raw_result = frozen_coordinate_explain(
        frozen,
        q=payload["q"],
        weight=payload["weight"],
        bias=payload["bias"],
    )
    result = dict(raw_result)
    result["config"] = config.name
    result["config_hash"] = config_hash(config)
    result["contributions"] = {
        "dtype": str(result["contributions"].dtype),
        "shape": list(result["contributions"].shape),
    }
    if "counterfactual_logits" in result:
        result["counterfactual_logits"] = {
            "dtype": str(result["counterfactual_logits"].dtype),
            "shape": list(result["counterfactual_logits"].shape),
        }
    for key in ("margins", "positive_temporal_share", "innovation_absolute_share", "class_pair_first", "class_pair_second"):
        if key in result:
            result[key] = {"dtype": str(result[key].dtype), "shape": list(result[key].shape)}
    if args.explain_npz:
        output_path = _repo_path(args.explain_npz)
        if output_path.suffix != ".npz":
            raise ValueError("EXPLAIN_ARTIFACT_MUST_END_NPZ")
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(f"OUTPUT_EXISTS:{output_path}")
        if any(part in {"frozen", "archive", "legacy"} for part in output_path.parts):
            raise ValueError(f"OUTPUT_PROTECTED:{output_path}")
        manifest_path = output_path.with_suffix(output_path.suffix + ".json")
        if manifest_path.exists() or manifest_path.is_symlink():
            raise FileExistsError(f"OUTPUT_EXISTS:{manifest_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"contributions": raw_result["contributions"], "group_names": np.asarray(result["group_names"])}
        original = raw_result
        for key in ("counterfactual_logits", "margins", "positive_temporal_share", "innovation_absolute_share", "class_pair_first", "class_pair_second"):
            if key in original:
                arrays[key] = original[key]
        np.savez_compressed(output_path, **arrays)
        result["explain_artifact"] = str(output_path)
        result["explain_artifact_sha256"] = _sha256(output_path)
        manifest_path.write_text(json.dumps({"schema_version": 1, "config": config.name, "config_hash": config_hash(config), "frozen_sha256": load_frozen_outputs(frozen).sha256, "artifact_sha256": result["explain_artifact_sha256"], "group_names": result["group_names"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result["explain_manifest"] = str(manifest_path)
    _emit(result, args.output)
    return 0


def _training_data(path: str, config, device: str):
    payload = load_npz(Path(path).resolve(), ("sequence_ids", "epoch_indices", "embeddings", "u0_logits", "labels"))
    if payload["embeddings"].shape[1] != config.input_dim:
        raise ValueError(f"TRAINING_INPUT_DIM_MISMATCH:{payload['embeddings'].shape[1]}:{config.input_dim}")
    return pad_by_sequence(
        payload["sequence_ids"], payload["epoch_indices"], payload["embeddings"],
        payload["u0_logits"], payload["labels"], device=device,
    )


def train(args: argparse.Namespace) -> int:
    config = _config(args)
    if args.stage != "head":
        raise RuntimeError("TRAIN_STAGE_UNSUPPORTED: use explicit --stage head")
    if args.device is None:
        raise RuntimeError("TRAIN_DEVICE_REQUIRED: explicit --device cpu or cuda:N")
    if "head" not in config.allowed_stages:
        raise RuntimeError("TRAIN_STAGE_NOT_IN_CONFIG:head")
    if float(config.head_optimizer.get("eps")) != 1e-8 or bool(config.head_optimizer.get("amsgrad")):
        raise RuntimeError("TRAIN_HEAD_OPTIMIZER_CONFIG_MISMATCH")
    if args.train_npz is None or args.held_npz is None:
        raise RuntimeError("TRAIN_SELECTION_DATA_REQUIRED: --train-npz and --held-npz")
    if args.fold not in range(10):
        raise ValueError("TRAIN_FOLD_MUST_BE_0_TO_9")
    if args.resume:
        raise RuntimeError("TRAIN_RESUME_UNSUPPORTED: optimizer/RNG restore is not implemented")
    target = resolve_device(args.device, allow_cpu=args.allow_cpu)
    args.effective_seed = config.seed + args.fold
    train_data = _training_data(args.train_npz, config, args.device)
    held_data = _training_data(args.held_npz, config, args.device)
    runs_root = _repo_path(args.runs_root or "artifacts/runs")
    if runs_root.name != "runs":
        raise ValueError(f"RUN_ROOT_NOT_CANONICAL:{runs_root}")
    run_dir, manifest = create_run(
        runs_root,
        args.run_id,
        config_hash_value=_effective_config_hash(config, args),
        protocol=config.protocol,
        stage=args.stage,
        input_paths=(Path(args.train_npz).resolve(), Path(args.held_npz).resolve()),
        resume=args.resume,
    )
    checkpoint = run_dir / "bim-head.pt"
    result = train_bim_head(
        train_data, held_data, input_dim=config.input_dim,
        updates=args.updates, learning_rate=float(config.head_optimizer.get("lr", 1e-3)),
        seed=args.effective_seed, device=str(target), allow_cpu=args.allow_cpu, checkpoint_path=checkpoint,
    )
    row = {
        "status": "HEAD_TRAINING_COMPLETE",
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "config": config.name,
        "config_hash": _effective_config_hash(config, args),
        "protocol": config.protocol,
        "selected_update": result.selected_update,
        "fold": args.fold,
        "seed": args.effective_seed,
        "effective": {"stage": args.stage, "device": str(target), "updates": args.updates, "allow_cpu": args.allow_cpu},
        "state_hash": result.state_hash,
        "checkpoint": str(checkpoint),
        "history": list(result.history),
    }
    output = run_dir / "training.json"
    if output.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output}")
    output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(row, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linear-bim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("doctor", doctor), ("audit", audit), ("evaluate", evaluate), ("explain", explain), ("train", train)):
        sub = subparsers.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument("--config", required=True)
        sub.add_argument("--output")
        if name in {"audit", "evaluate", "explain"}:
            sub.add_argument("--frozen")
        if name == "explain":
            sub.add_argument("--explain-input", help="NPZ with q, weight, bias arrays")
            sub.add_argument("--explain-npz", help="new NPZ artifact for exact contribution arrays")
        if name == "train":
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--runs-root")
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--stage")
            sub.add_argument("--device")
            sub.add_argument("--allow-cpu", action="store_true")
            sub.add_argument("--train-npz")
            sub.add_argument("--held-npz")
            sub.add_argument("--updates", type=int, default=200)
            sub.add_argument("--fold", type=int, default=0)
        if name == "doctor":
            pass
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
