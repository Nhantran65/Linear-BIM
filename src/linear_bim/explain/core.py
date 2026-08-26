"""Small, deterministic helpers for coordinate-level BIM-Explain."""

from __future__ import annotations

import numpy as np


CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")
GROUP_NAMES = (
    "current",
    "past_memory_0", "past_memory_1", "past_memory_2",
    "future_memory_0", "future_memory_1", "future_memory_2",
    "past_innovation_0", "past_innovation_1", "past_innovation_2",
    "future_innovation_0", "future_innovation_1", "future_innovation_2",
    "bias",
)
GROUP_SLICES = (
    (0, 16),
    (16, 32), (32, 48), (48, 64),
    (64, 80), (80, 96), (96, 112),
    (112, 128), (128, 144), (144, 160),
    (160, 176), (176, 192), (192, 208),
    None,
)
TEMPORAL_GROUPS = tuple(range(1, 13))
INNOVATION_GROUPS = tuple(range(7, 13))
DELETION_GROUPS = {
    "context_all": TEMPORAL_GROUPS,
    "past_all": (1, 2, 3, 7, 8, 9),
    "future_all": (4, 5, 6, 10, 11, 12),
    "memory_all": (1, 2, 3, 4, 5, 6),
    "innovation_all": INNOVATION_GROUPS,
    "scale_0": (1, 4, 7, 10),
    "scale_1": (2, 5, 8, 11),
    "scale_2": (3, 6, 9, 12),
}
DELETION_NAMES = tuple(DELETION_GROUPS)


def contribution_logits(q: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return ``[N,14,5]`` exact additive residual-logit contributions."""

    q = q.astype(np.float64)
    weight = weight.astype(np.float64)
    bias = bias.astype(np.float64)
    rows = []
    for bounds in GROUP_SLICES[:-1]:
        start, stop = bounds
        rows.append(q[:, start:stop] @ weight[:, start:stop].T)
    rows.append(np.broadcast_to(bias, (len(q), len(bias))).copy())
    return np.stack(rows, axis=1)


def enforce_exact_partition(
    contributions: np.ndarray,
    residual: np.ndarray,
    current_plus_bias_residual: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Allocate float32 association round-off without changing model outputs."""

    output = contributions.astype(np.float64, copy=True)
    residual = residual.astype(np.float64)
    current_plus_bias_residual = current_plus_bias_residual.astype(np.float64)
    current_adjustment = current_plus_bias_residual - (output[:, 0] + output[:, -1])
    output[:, 0] += current_adjustment
    temporal_target = residual - current_plus_bias_residual
    temporal_adjustment = temporal_target - output[:, TEMPORAL_GROUPS].sum(axis=1)
    output[:, TEMPORAL_GROUPS[0]] += temporal_adjustment
    maximum = float(max(np.max(np.abs(current_adjustment)), np.max(np.abs(temporal_adjustment))))
    return output, maximum


def counterfactual_logits(final_logits: np.ndarray, contributions: np.ndarray) -> np.ndarray:
    return np.stack(
        [final_logits - contributions[:, DELETION_GROUPS[name]].sum(axis=1) for name in DELETION_NAMES],
        axis=1,
    )


def transition_mask(labels: np.ndarray, entries: np.ndarray, epoch_indices: np.ndarray) -> np.ndarray:
    result = np.zeros(len(labels), dtype=bool)
    for entry in np.unique(entries):
        indices = np.flatnonzero(entries == entry)
        indices = indices[np.argsort(epoch_indices[indices], kind="stable")]
        values = labels[indices]
        local = np.zeros(len(indices), dtype=bool)
        local[1:] |= values[1:] != values[:-1]
        local[:-1] |= values[:-1] != values[1:]
        result[indices] = local
    return result


def class_pair(labels: np.ndarray, u0_logits: np.ndarray, bim_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair used for completeness: truth-vs-U0 when U0 errs, else BIM-vs-runner."""

    u0_prediction = u0_logits.argmax(axis=1)
    bim_prediction = bim_logits.argmax(axis=1)
    first = np.empty(len(labels), dtype=np.int64)
    second = np.empty(len(labels), dtype=np.int64)
    for index in range(len(labels)):
        if u0_prediction[index] != labels[index]:
            first[index], second[index] = labels[index], u0_prediction[index]
        else:
            first[index] = bim_prediction[index]
            order = np.argsort(bim_logits[index], kind="stable")
            second[index] = order[-2] if order[-1] == first[index] else order[-1]
    return first, second


def group_margin(contributions: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    rows = np.arange(len(first))[:, None]
    groups = np.arange(contributions.shape[1])[None, :]
    return contributions[rows, groups, first[:, None]] - contributions[rows, groups, second[:, None]]


def positive_temporal_share(margins: np.ndarray) -> np.ndarray:
    positive = np.maximum(margins, 0.0)
    denominator = positive.sum(axis=1)
    numerator = positive[:, TEMPORAL_GROUPS].sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def innovation_absolute_share(margins: np.ndarray) -> np.ndarray:
    absolute = np.abs(margins[:, :13])
    denominator = absolute.sum(axis=1)
    numerator = absolute[:, INNOVATION_GROUPS].sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def deterministic_example_indices(
    labels: np.ndarray,
    subjects: np.ndarray,
    entries: np.ndarray,
    epoch_indices: np.ndarray,
    folds: np.ndarray,
    u0_prediction: np.ndarray,
    bim_prediction: np.ndarray,
    transitions: np.ndarray,
) -> list[tuple[int, str, int]]:
    result = []
    for fold in range(10):
        candidates = {
            "corrected_n1": (folds == fold) & (labels == 1) & (u0_prediction != labels) & (bim_prediction == labels),
            "harmed": (folds == fold) & (u0_prediction == labels) & (bim_prediction != labels),
            "stable_correct": (folds == fold) & (~transitions) & (u0_prediction == labels) & (bim_prediction == labels),
        }
        for kind, mask in candidates.items():
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            order = np.lexsort((epoch_indices[indices], entries[indices], subjects[indices]))
            result.append((fold, kind, int(indices[order[0]])))
    return result
