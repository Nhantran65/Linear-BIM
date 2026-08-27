"""Deterministic primitives for the exact full BIM-Explain analysis.

This module deliberately remains separate from :mod:`linear_bim.explain.core`:
the former collapses local/current terms and reports eight groups, while the
latter reports fourteen coordinate groups.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")
PURE_GROUP_NAMES = (
    "local_total",
    "past_history_0", "past_history_1", "past_history_2",
    "future_history_0", "future_history_1", "future_history_2",
    "bias",
)
PURE_HISTORY_GROUPS = tuple(range(1, 7))
PURE_DELETION_GROUPS = {
    "all_true_history": PURE_HISTORY_GROUPS,
    "past_true_history": (1, 2, 3),
    "future_true_history": (4, 5, 6),
    "true_history_scale_0": (1, 4),
    "true_history_scale_1": (2, 5),
    "true_history_scale_2": (3, 6),
    "local_total": (0,),
}
PURE_DELETION_NAMES = tuple(PURE_DELETION_GROUPS)
OUTCOME_NAMES = ("correction", "harm", "persistent_error", "stable_correct")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def metric_bundle(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
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


def runner_up(logits: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    order = np.argsort(logits, axis=1, kind="stable")
    result = order[:, -1].copy()
    chosen_is_top = result == prediction
    result[chosen_is_top] = order[chosen_is_top, -2]
    return result.astype(np.int64)


def group_margin(contribution: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    rows = np.arange(len(first))[:, None]
    groups = np.arange(contribution.shape[1])[None, :]
    return contribution[rows, groups, first[:, None]] - contribution[rows, groups, second[:, None]]


def positive_share(margins: np.ndarray, groups: tuple[int, ...]) -> tuple[np.ndarray, int]:
    positive = np.maximum(margins, 0.0)
    denominator = positive.sum(axis=1)
    numerator = positive[:, groups].sum(axis=1)
    share = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return share, int(np.sum(denominator <= 0))


def transition_mask(labels: np.ndarray, entries: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    result = np.zeros(len(labels), dtype=bool)
    for entry in np.unique(entries):
        indices = np.flatnonzero(entries == entry)
        indices = indices[np.argsort(epochs[indices], kind="stable")]
        values = labels[indices]
        local = np.zeros(len(indices), dtype=bool)
        local[1:] |= values[1:] != values[:-1]
        local[:-1] |= values[:-1] != values[1:]
        result[indices] = local
    return result


def outcome_codes(labels: np.ndarray, ulw_prediction: np.ndarray, bim_prediction: np.ndarray) -> np.ndarray:
    ulw_correct = ulw_prediction == labels
    bim_correct = bim_prediction == labels
    output = np.empty(len(labels), dtype=np.int8)
    output[(~ulw_correct) & bim_correct] = 0
    output[ulw_correct & (~bim_correct)] = 1
    output[(~ulw_correct) & (~bim_correct)] = 2
    output[ulw_correct & bim_correct] = 3
    return output


def outcome_counts(codes: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int]:
    if mask is None:
        mask = np.ones(len(codes), dtype=bool)
    return {name: int(np.sum(mask & (codes == index))) for index, name in enumerate(OUTCOME_NAMES)}


def pure_local_history_decomposition(
    q: np.ndarray,
    residual: np.ndarray,
    folds: np.ndarray,
    weights: dict[int, np.ndarray],
    biases: dict[int, np.ndarray],
    lambdas: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    """Return exact ``[N,8,5]`` local/history residual contributions.

    The local-history identity and float64 roundoff allocation are retained
    exactly so deletion outputs sum to the observed residual logits.
    """

    q64 = q.astype(np.float64, copy=False)
    residual64 = residual.astype(np.float64, copy=False)
    output = np.zeros((len(q), len(PURE_GROUP_NAMES), residual.shape[1]), dtype=np.float64)
    identity_error = 0.0
    allocation = 0.0
    for fold in sorted(weights):
        index = np.flatnonzero(folds == fold)
        if not len(index):
            continue
        local_q = q64[index]
        p = local_q[:, :16]
        mf = local_q[:, 16:64].reshape(-1, 3, 16)
        mb = local_q[:, 64:112].reshape(-1, 3, 16)
        df = local_q[:, 112:160].reshape(-1, 3, 16)
        db = local_q[:, 160:208].reshape(-1, 3, 16)
        weight = weights[fold].astype(np.float64, copy=False)
        lam = lambdas[fold].astype(np.float64, copy=False)
        local_contribution = p @ weight[:, :16].T
        for memory, innovation, memory_offset, innovation_offset, output_offset in (
            (mf, df, 16, 112, 1),
            (mb, db, 64, 160, 4),
        ):
            for scale in range(3):
                start_m = memory_offset + scale * 16
                start_d = innovation_offset + scale * 16
                wm = weight[:, start_m:start_m + 16]
                wd = weight[:, start_d:start_d + 16]
                local_weight = (1.0 - lam[scale]) * wm + lam[scale] * wd
                local_pair = p @ local_weight.T
                pair_direct = memory[:, scale] @ wm.T + innovation[:, scale] @ wd.T
                neighbor = (memory[:, scale] - (1.0 - lam[scale]) * p) / lam[scale]
                history_direct = neighbor @ (lam[scale] * (wm - wd)).T
                identity_error = max(identity_error, float(np.max(np.abs(pair_direct - local_pair - history_direct))))
                local_contribution += local_pair
                output[index, output_offset + scale] = history_direct
        output[index, 0] = local_contribution
        output[index, -1] = biases[fold].astype(np.float64, copy=False)
        adjustment = residual64[index] - output[index].sum(axis=1)
        allocation = max(allocation, float(np.max(np.abs(adjustment))))
        output[index, 0] += adjustment
    completeness = float(np.max(np.abs(output.sum(axis=1) - residual64)))
    return output, {
        "pair_identity_max_abs_error": identity_error,
        "roundoff_allocation_max_abs": allocation,
        "residual_completeness_max_abs_error": completeness,
    }


def counterfactual_logits(final_logits: np.ndarray, contribution: np.ndarray) -> np.ndarray:
    return np.stack(
        [final_logits - contribution[:, PURE_DELETION_GROUPS[name]].sum(axis=1) for name in PURE_DELETION_NAMES],
        axis=1,
    )


def prediction_flow(
    labels: np.ndarray,
    ulw_prediction: np.ndarray,
    bim_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> dict[str, Any]:
    corrected = (ulw_prediction != labels) & (bim_prediction == labels)
    harmed = (ulw_prediction == labels) & (bim_prediction != labels)
    correction_rows = {
        "count": int(corrected.sum()),
        "still_correct": int(np.sum(corrected & (candidate_prediction == labels))),
        "equal_original_ulw_wrong_class": int(np.sum(corrected & (candidate_prediction == ulw_prediction))),
        "different_wrong_class": int(np.sum(corrected & (candidate_prediction != labels) & (candidate_prediction != ulw_prediction))),
    }
    harm_rows = {
        "count": int(harmed.sum()),
        "repaired": int(np.sum(harmed & (candidate_prediction == labels))),
        "unchanged_wrong_class": int(np.sum(harmed & (candidate_prediction == bim_prediction))),
        "different_wrong_class": int(np.sum(harmed & (candidate_prediction != labels) & (candidate_prediction != bim_prediction))),
    }
    return {
        "changed_predictions": int(np.sum(candidate_prediction != bim_prediction)),
        "changed_prediction_rate": float(np.mean(candidate_prediction != bim_prediction)),
        "prediction_equal_ulw": int(np.sum(candidate_prediction == ulw_prediction)),
        "prediction_equal_ulw_rate": float(np.mean(candidate_prediction == ulw_prediction)),
        "corrections": correction_rows,
        "harms": harm_rows,
    }


def contribution_summary(margins: np.ndarray, names: tuple[str, ...], mask: np.ndarray) -> dict[str, Any]:
    selected = margins[mask]
    if not len(selected):
        return {"count": 0, "groups": {}}
    return {
        "count": int(len(selected)),
        "groups": {
            name: {
                "signed_mean": float(selected[:, index].mean()),
                "signed_median": float(np.median(selected[:, index])),
                "absolute_mean": float(np.abs(selected[:, index]).mean()),
                "absolute_median": float(np.median(np.abs(selected[:, index]))),
            }
            for index, name in enumerate(names)
        },
    }


def subject_statistic(
    values: np.ndarray,
    mask: np.ndarray,
    subjects: np.ndarray,
    reducer: Callable[[np.ndarray], float] = lambda row: float(np.median(row)),
) -> tuple[np.ndarray, np.ndarray]:
    subject_ids, statistics = [], []
    for subject in np.unique(subjects):
        local = mask & (subjects == subject)
        if not np.any(local):
            continue
        subject_ids.append(int(subject))
        statistics.append(float(reducer(values[local])))
    return np.asarray(subject_ids, dtype=np.int64), np.asarray(statistics, dtype=np.float64)


def bootstrap_subject_values(
    values: np.ndarray, seed: int, samples: int, reference: float | None = None
) -> tuple[dict[str, Any], np.ndarray]:
    if not len(values):
        return ({"n_subjects": 0, "estimate": None, "ci95": [None, None], "reference": reference}, np.empty(0))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    draws = values[indices].mean(axis=1)
    report = {
        "n_subjects": int(len(values)),
        "estimate": float(values.mean()),
        "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
        "reference": reference,
        "samples": int(samples),
        "seed": int(seed),
    }
    return report, draws


def subject_transition_difference(
    values: np.ndarray, transitions: np.ndarray, subjects: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ids, differences = [], []
    for subject in np.unique(subjects):
        local = subjects == subject
        if not np.any(local & transitions) or not np.any(local & ~transitions):
            continue
        ids.append(int(subject))
        differences.append(float(values[local & transitions].mean() - values[local & ~transitions].mean()))
    return np.asarray(ids, dtype=np.int64), np.asarray(differences, dtype=np.float64)


def subject_metric_differences(
    labels: np.ndarray,
    full_prediction: np.ndarray,
    deleted_prediction: np.ndarray,
    subjects: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    ids, differences = [], []
    for subject in np.unique(subjects):
        local = subjects == subject
        if metric == "n1_f1" and not np.any(labels[local] == 1):
            continue
        full = metric_bundle(labels[local], full_prediction[local])[metric]
        deleted = metric_bundle(labels[local], deleted_prediction[local])[metric]
        ids.append(int(subject))
        differences.append(float(full - deleted))
    return np.asarray(ids, dtype=np.int64), np.asarray(differences, dtype=np.float64)


def stratum_rows(labels: np.ndarray, outcomes: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family, array in values.items():
        rows = {}
        for value in np.unique(array):
            mask = array == value
            key = CLASS_NAMES[int(value)] if family == "class" else str(int(value))
            rows[key] = {"epochs": int(mask.sum()), "outcomes": outcome_counts(outcomes, mask)}
        result[family] = rows
    result["pooled"] = {"epochs": int(len(labels)), "outcomes": outcome_counts(outcomes)}
    return result


def deterministic_indices(
    labels: np.ndarray,
    folds: np.ndarray,
    subjects: np.ndarray,
    entries: np.ndarray,
    epochs: np.ndarray,
    outcomes: np.ndarray,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    kinds = {
        "corrected_n1": (outcomes == 0) & (labels == 1),
        "harm": outcomes == 1,
        "persistent_error": outcomes == 2,
        "stable_correct": outcomes == 3,
    }
    for fold in range(10):
        for kind, base in kinds.items():
            index = np.flatnonzero(base & (folds == fold))
            if not len(index):
                continue
            order = np.lexsort((epochs[index], entries[index], subjects[index]))
            rows.append({"fold": fold, "kind": kind, "index": int(index[order[0]])})
    return rows
