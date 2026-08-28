"""BIM-Explain adapters for supplied/frozen arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    GROUP_NAMES,
    class_pair,
    contribution_logits,
    counterfactual_logits as coordinate_counterfactuals,
    enforce_exact_partition,
    group_margin,
    innovation_absolute_share,
    positive_temporal_share,
)
from .full_core import (
    PURE_GROUP_NAMES,
    counterfactual_logits as history_counterfactuals,
    pure_local_history_decomposition,
)
from ..io.runs import load_frozen_outputs


def coordinate_explain(
    q: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    final_logits: np.ndarray | None = None,
    u0_logits: np.ndarray | None = None,
    bim_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute exact 14-group contributions and coordinate deletions."""

    contribution = contribution_logits(q, weight, bias)
    current = contribution[:, 0] + contribution[:, -1]
    exact, allocation = enforce_exact_partition(contribution, residual, current)
    result: dict[str, Any] = {
        "group_names": list(GROUP_NAMES),
        "contributions": exact,
        "roundoff_allocation_max_abs": allocation,
    }
    if final_logits is not None:
        result["counterfactual_logits"] = coordinate_counterfactuals(final_logits, exact)
    if u0_logits is not None and bim_logits is not None:
        if labels is None:
            raise ValueError("EXPLAIN_LABELS_REQUIRED_FOR_CLASS_PAIR")
        first, second = class_pair(np.asarray(labels), u0_logits, bim_logits)
        margins = group_margin(exact, first, second)
        result["class_pair_first"] = first
        result["class_pair_second"] = second
        result["margins"] = margins
        result["positive_temporal_share"] = positive_temporal_share(margins)
        result["innovation_absolute_share"] = innovation_absolute_share(margins)
    return result


def history_explain(
    q: np.ndarray,
    residual: np.ndarray,
    folds: np.ndarray,
    weights: dict[int, np.ndarray],
    biases: dict[int, np.ndarray],
    lambdas: dict[int, np.ndarray],
    *,
    final_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the distinct exact 8-group local/history decomposition."""

    contribution, diagnostics = pure_local_history_decomposition(
        q, residual, folds, weights, biases, lambdas
    )
    result: dict[str, Any] = {
        "group_names": list(PURE_GROUP_NAMES),
        "contributions": contribution,
        "diagnostics": diagnostics,
    }
    if final_logits is not None:
        result["counterfactual_logits"] = history_counterfactuals(final_logits, contribution)
    return result


def checkpoint_state(path: Path | str) -> dict[str, Any]:
    """Load a state-dict checkpoint without unpickling arbitrary objects."""

    import torch

    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("CHECKPOINT_STATE_DICT_REQUIRED")
    return value


def frozen_coordinate_explain(
    frozen_path: Path | str,
    *,
    q: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> dict[str, Any]:
    """Explain an immutable output file with an explicitly supplied feature matrix."""

    frozen = load_frozen_outputs(frozen_path)
    if len(q) != len(frozen.arrays["labels"]):
        raise ValueError("EXPLAIN_Q_ROW_IDENTITY")
    if "folds" in frozen.arrays and len(np.unique(frozen.arrays["folds"])) != 1:
        raise ValueError("EXPLAIN_POOLED_MULTIFOLD_REQUIRES_FOLD_CHECKPOINTS")
    residual = frozen.arrays["b0_logits"] - frozen.arrays["u0_logits"]
    q64 = np.asarray(q, dtype=np.float64)
    expected = q64 @ np.asarray(weight, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)
    if expected.shape != residual.shape or not np.allclose(expected, residual, rtol=2e-5, atol=5e-6):
        raise ValueError("EXPLAIN_CHECKPOINT_Q_RESIDUAL_MISMATCH")
    result = coordinate_explain(
        q.astype(np.float32, copy=False),
        residual,
        weight,
        bias,
        labels=frozen.arrays["labels"],
        final_logits=frozen.arrays["b0_logits"],
        u0_logits=frozen.arrays["u0_logits"],
        bim_logits=frozen.arrays["b0_logits"],
    )
    result["frozen_sha256"] = frozen.sha256
    return result
