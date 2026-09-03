from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from linear_bim.data.padding import pad_by_sequence
from linear_bim.io.runs import create_run, load_frozen_outputs
from linear_bim.training.head import train_bim_head


def _padded(seed: int = 1):
    generator = np.random.default_rng(seed)
    sequence_ids = np.repeat(np.arange(3), 4)
    epochs = np.tile(np.arange(4), 3)
    embeddings = generator.normal(size=(12, 128)).astype(np.float32)
    logits = generator.normal(size=(12, 5)).astype(np.float32)
    labels = generator.integers(0, 5, size=12, dtype=np.int64)
    return pad_by_sequence(sequence_ids, epochs, embeddings, logits, labels)


def test_head_uses_constant_lr_beyond_first_restart() -> None:
    train = _padded()
    held = _padded(2)
    result = train_bim_head(train, held, updates=12, seed=32, device="cpu")
    assert len(result.history) == 12
    assert {row["learning_rate"] for row in result.history} == {1e-3}
    assert result.selected_update >= 1


def test_frozen_aliases_and_contradictions(tmp_path: Path) -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    ulw = np.zeros((2, 5), dtype=np.float32)
    bim = np.ones((2, 5), dtype=np.float32)
    source = tmp_path / "frozen.npz"
    np.savez(source, labels=labels, ulw_logits=ulw, bim_logits=bim)
    frozen = load_frozen_outputs(source)
    assert np.array_equal(frozen.arrays["u0_logits"], ulw)
    contradictory = tmp_path / "contradictory.npz"
    np.savez(contradictory, labels=labels, u0_logits=ulw, ulw_logits=np.ones_like(ulw), b0_logits=bim)
    with pytest.raises(ValueError, match="CONTRADICTORY"):
        load_frozen_outputs(contradictory)


def test_run_resume_requires_optimizer_and_rng_state(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "runs"
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"input")
    output, _ = create_run(root, "safe-run", config_hash_value="a", protocol="p", stage="head", input_paths=(input_path,))
    assert json.loads((output / "run.json").read_text())["run_id"] == "safe-run"
    with pytest.raises(RuntimeError, match="RESUME_UNSUPPORTED"):
        create_run(root, "safe-run", config_hash_value="a", protocol="p", stage="head", input_paths=(input_path,), resume=True)
    with pytest.raises(ValueError, match="RUN_ID_INVALID"):
        create_run(root, "../escape", config_hash_value="a", protocol="p", stage="head")
