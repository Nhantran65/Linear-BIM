from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from linear_bim import cli


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "s20.yaml"
    path.write_text(
        """name: s20
dataset: sleep-edf-20
variant: linear_bim
input_dim: 128
modalities: 4
classes: 5
dense_width: 64
protocol: test
scope: test
frozen_outputs: artifacts/frozen/s20/frozen_outputs.npz
required_inputs: []
training:
  allowed_stages: [head]
  backbone_optimizer: {name: Adam, eps: 1.0e-7, amsgrad: true}
  head_optimizer: {name: Adam, eps: 1.0e-8, amsgrad: false, lr: 0.001}
seed: 32
""",
        encoding="utf-8",
    )
    return path


def _frozen_inputs(tmp_path: Path) -> tuple[Path, Path]:
    frozen_dir = tmp_path / "artifacts" / "frozen" / "s20"
    frozen_dir.mkdir(parents=True)
    generator = np.random.default_rng(7)
    q = generator.normal(size=(4, 208)).astype(np.float32)
    weight = generator.normal(size=(5, 208)).astype(np.float32)
    bias = generator.normal(size=5).astype(np.float32)
    u0 = generator.normal(size=(4, 5)).astype(np.float32)
    residual = q @ weight.T + bias
    frozen = frozen_dir / "frozen_outputs.npz"
    np.savez(frozen, labels=np.asarray([0, 1, 2, 3]), u0_logits=u0, b0_logits=u0 + residual, folds=np.zeros(4, dtype=np.int8))
    explain = tmp_path / "explain.npz"
    np.savez(explain, q=q, weight=weight, bias=bias)
    return frozen, explain


def test_cli_evaluate_and_explain_write_only_new_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    frozen, explain = _frozen_inputs(tmp_path)
    assert cli.main(["evaluate", "--config", str(config), "--frozen", str(frozen)]) == 0
    artifact = tmp_path / "artifacts" / "runs" / "explain.npz"
    assert cli.main([
        "explain", "--config", str(config), "--frozen", str(frozen),
        "--explain-input", str(explain), "--explain-npz", str(artifact),
    ]) == 0
    assert artifact.is_file()
    assert artifact.with_suffix(".npz.json").is_file()
    with np.load(artifact, allow_pickle=False) as archive:
        assert archive["contributions"].shape == (4, 14, 5)
        assert archive["counterfactual_logits"].shape[1] == 8


def test_cli_train_requires_explicit_stage_device_and_selection_data(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    code = cli.main(["train", "--config", str(config), "--run-id", "test-run"])
    assert code == 2
    assert "TRAIN_STAGE_UNSUPPORTED" in capsys.readouterr().err


def test_cli_doctor_reports_missing_inputs_without_writes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert cli.main(["doctor", "--config", str(config)]) == 0
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert before == after
