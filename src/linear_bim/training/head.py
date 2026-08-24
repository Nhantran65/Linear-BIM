"""Safe BIM-head training over supplied, already-computed embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Must be set before the first CUDA context/matmul is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

from ..data.padding import PaddedSequences
from ..models.bim import BIM


@dataclass(frozen=True)
class HeadTrainingResult:
    model: BIM
    selected_update: int
    history: tuple[dict[str, Any], ...]
    state_hash: str


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape)).encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str, *, allow_cpu: bool = False) -> torch.device:
    device = torch.device(name)
    if device.type == "cpu":
        if not allow_cpu:
            raise RuntimeError("DEVICE_CPU_NOT_ALLOWED: pass --allow-cpu for rehearsal/head stage")
        return device
    if device.type != "cuda":
        raise RuntimeError(f"DEVICE_UNSUPPORTED:{name}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA_UNAVAILABLE:{name}")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA_DEVICE_UNAVAILABLE:{name}")
    return device


def _validate_padded(data: PaddedSequences, input_dim: int) -> None:
    if data.embeddings.ndim != 3 or data.embeddings.shape[-1] != input_dim:
        raise ValueError(f"TRAINING_EMBEDDING_SHAPE:{tuple(data.embeddings.shape)} expected *x*{input_dim}")
    if data.u0_logits.shape != data.labels.shape + (5,):
        raise ValueError("TRAINING_U0_LOGITS_SHAPE")
    if data.mask.shape != data.labels.shape:
        raise ValueError("TRAINING_MASK_SHAPE")
    if not torch.all(data.mask == (torch.arange(data.labels.shape[1], device=data.mask.device)[None] < data.lengths[:, None])):
        raise ValueError("TRAINING_MASK_INCONSISTENT")
    if not torch.all((data.labels[data.mask] >= 0) & (data.labels[data.mask] < 5)):
        raise ValueError("TRAINING_LABEL_RANGE")
    for name, value in (("embeddings", data.embeddings), ("u0_logits", data.u0_logits)):
        if not torch.isfinite(value).all():
            raise ValueError(f"TRAINING_NONFINITE:{name}")


def train_bim_head(
    train_data: PaddedSequences,
    held_data: PaddedSequences,
    *,
    input_dim: int = 128,
    seed: int = 32,
    updates: int = 200,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    allow_cpu: bool = True,
    checkpoint_path: Path | None = None,
    overwrite: bool = False,
) -> HeadTrainingResult:
    """Train only the BIM head on supplied embedding/U0-logit arrays.

    No data is fetched and no backbone is initialized.  ``held_data`` is
    used only for selection, matching the historical best-held-out head
    protocol. Existing checkpoints are never
    overwritten unless ``overwrite=True`` is explicitly supplied.
    """

    if updates <= 0:
        raise ValueError("updates must be positive")
    target = resolve_device(device, allow_cpu=allow_cpu)
    _validate_padded(train_data, input_dim)
    _validate_padded(held_data, input_dim)
    set_determinism(seed)
    model = BIM(input_dim=input_dim).to(target)
    def move(data: PaddedSequences) -> PaddedSequences:
        return PaddedSequences(
            embeddings=data.embeddings.to(target),
            u0_logits=data.u0_logits.to(target),
            labels=data.labels.to(target),
            lengths=data.lengths.to(target),
            mask=data.mask.to(target),
            rows=data.rows,
            n_flat=data.n_flat,
        )
    train_data = move(train_data)
    held_data = move(held_data)
    # Historical BIM head: Adam eps=1e-8, amsgrad disabled.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_update = -1
    best_state: dict[str, torch.Tensor] | None = None
    labels_for_score = held_data.labels[held_data.mask]
    eval_data = held_data
    for update in range(1, updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual = model(train_data.embeddings, train_data.lengths)
        logits = train_data.u0_logits + residual
        loss = nn.functional.cross_entropy(logits[train_data.mask], train_data.labels[train_data.mask])
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            held_logits = eval_data.u0_logits + model(eval_data.embeddings, eval_data.lengths)
            prediction = held_logits[eval_data.mask].argmax(-1)
            score = int((prediction == labels_for_score).sum().item()) / int(labels_for_score.numel())
        row = {
            "update": update,
            "loss": float(loss.detach().cpu()),
            "gradient_norm_before_clip": float(grad),
            "selection_accuracy": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_update = update
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("TRAINING_NO_CHECKPOINT")
    model.load_state_dict(best_state, strict=True)
    digest = state_hash(best_state)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists() and not overwrite:
            raise FileExistsError(f"CHECKPOINT_EXISTS:{checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)
    return HeadTrainingResult(model, best_update, tuple(history), digest)
