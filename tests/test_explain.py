from __future__ import annotations

import numpy as np

from linear_bim.explain.core import DELETION_GROUPS, GROUP_NAMES, contribution_logits, enforce_exact_partition
from linear_bim.explain.full_core import PURE_GROUP_NAMES, pure_local_history_decomposition


def test_coordinate_partition_and_full_history_are_distinct() -> None:
    generator = np.random.default_rng(12)
    q = generator.normal(size=(7, 208)).astype(np.float32)
    weight = generator.normal(size=(5, 208)).astype(np.float32)
    bias = generator.normal(size=5).astype(np.float32)
    residual = q @ weight.T + bias
    contribution = contribution_logits(q, weight, bias)
    exact, adjustment = enforce_exact_partition(contribution, residual, contribution[:, 0] + contribution[:, -1])
    assert len(GROUP_NAMES) == 14
    assert len(PURE_GROUP_NAMES) == 8
    assert adjustment < 2e-5
    np.testing.assert_allclose(exact.sum(1), residual, rtol=0, atol=1e-6)
    for name, groups in DELETION_GROUPS.items():
        np.testing.assert_allclose(residual - exact[:, groups].sum(1), residual - exact[:, groups].sum(1))


def test_full_history_decomposition_reports_fidelity() -> None:
    generator = np.random.default_rng(20260830)
    count = 12
    folds = np.repeat(np.arange(2), count // 2).astype(np.int8)
    q = np.zeros((count, 208), dtype=np.float32)
    weights: dict[int, np.ndarray] = {}
    biases: dict[int, np.ndarray] = {}
    lambdas: dict[int, np.ndarray] = {}
    residual = np.zeros((count, 5), dtype=np.float32)
    for fold in range(2):
        index = np.flatnonzero(folds == fold)
        p = generator.normal(size=(len(index), 16))
        lam = np.asarray([.55, .75, .88], dtype=np.float64)
        rows = [p]
        memories, innovations = [], []
        for _ in range(2):
            memory = np.empty((len(index), 3, 16), dtype=np.float64)
            innovation = np.empty_like(memory)
            for scale in range(3):
                neighbor = generator.normal(size=(len(index), 16))
                memory[:, scale] = lam[scale] * neighbor + (1 - lam[scale]) * p
                innovation[:, scale] = p - memory[:, scale]
            memories.append(memory)
            innovations.append(innovation)
        rows.extend([memories[0].reshape(len(index), -1), memories[1].reshape(len(index), -1)])
        rows.extend([innovations[0].reshape(len(index), -1), innovations[1].reshape(len(index), -1)])
        local_q = np.concatenate(rows, axis=1).astype(np.float32)
        weight = generator.normal(size=(5, 208)).astype(np.float32)
        bias = generator.normal(size=5).astype(np.float32)
        q[index] = local_q
        residual[index] = local_q @ weight.T + bias
        weights[fold], biases[fold], lambdas[fold] = weight, bias, lam.astype(np.float32)
    output, diagnostics = pure_local_history_decomposition(q, residual, folds, weights, biases, lambdas)
    assert output.shape == (count, 8, 5)
    assert diagnostics["pair_identity_max_abs_error"] < 1e-5
    assert diagnostics["residual_completeness_max_abs_error"] <= 1e-12
    np.testing.assert_allclose(output.sum(1), residual, rtol=0, atol=1e-7)
