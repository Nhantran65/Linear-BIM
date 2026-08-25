"""Read-only evaluation of immutable frozen output arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from ..io.runs import FrozenOutputs, load_frozen_outputs


CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")


def metric_bundle(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    prediction = np.asarray(prediction)
    if labels.ndim != 1 or prediction.shape != labels.shape:
        raise ValueError("EVALUATION_LABEL_PREDICTION_SHAPE")
    stages = f1_score(labels, prediction, labels=range(5), average=None, zero_division=0)
    return {
        "n_epochs": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(stages.mean()),
        "kappa": float(cohen_kappa_score(labels, prediction, labels=range(5))),
        "n1_f1": float(stages[1]),
        "per_class_f1": {name: float(stages[index]) for index, name in enumerate(CLASS_NAMES)},
        "confusion_matrix_true_rows": confusion_matrix(labels, prediction, labels=range(5)).tolist(),
    }


def evaluate_frozen(value: FrozenOutputs | str) -> dict[str, Any]:
    """Return U0/B0/Z0 metrics without mutating the frozen source."""

    frozen = value if isinstance(value, FrozenOutputs) else load_frozen_outputs(value)
    labels = frozen.arrays["labels"]
    result: dict[str, Any] = {
        "frozen_path": str(frozen.path),
        "frozen_sha256": frozen.sha256,
        "metrics": {},
        "n_epochs": int(len(labels)),
    }
    for name, key in (("U0", "u0_logits"), ("B0", "b0_logits"), ("Z0", "z0_logits")):
        if key in frozen.arrays:
            logits = frozen.arrays[key]
            result["metrics"][name] = metric_bundle(labels, logits.argmax(axis=1))
    return result
