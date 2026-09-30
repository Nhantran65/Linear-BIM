# Linear-BIM: source code and compact numerical review evidence

This repository contains an implementation of Linear-BIM, a bidirectional temporal residual head over a frozen ULW-SleepNet backbone, and BIM-Explain for exact decomposition of that residual. It accompanies **`Linear_BIM_Paper.pdf` supplied on 2026-09-30** (`Linear-BIM: Bidirectional Innovation Memory with Exact Attribution for Sleep Stage Scoring`; PDF SHA-256 and source hashes are recorded in `review_evidence/paper_v1.json`). The PDF is a local input to this audit, not part of the small Git review package. The complete manuscript source is unavailable here; the checked-in historical Table I TeX refers to a different, official-checkpoint comparison. See [the item-by-item audit](review_evidence/README.md).

## Check the paper numbers from a clean clone

Python 3.10+ and NumPy are sufficient for the compact checker:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy
.venv/bin/python scripts/check_review_evidence.py
```

This checks the three dataset confusion matrices, matched gains, fold directions, Table II intervention metrics, subject bootstrap confidence intervals and Figure 2 margin arithmetic. If the original ignored source files are present, use `--check-sources` to verify their SHA-256 as well. The JSON contains the source paths and hashes. This is a **numerical review check**, not a rerun of signal preprocessing, checkpoint training or model evaluation. It does not independently recover event-level pooled history medians or FLOP measurements. The PDF's Sleep-EDF FLOPs differ from the report's operation ledger; the exact comparison is in the audit.

## Layout

| Path | Role in a clean clone |
|---|---|
| `src/linear_bim/` | Supported model, head training, frozen evaluation and attribution primitives |
| `configs/paper/` | S20, S78 and S3 configuration, including local input paths and protocol labels |
| `experiments/recipes/` | Protocol mapping and limits of the supported runner |
| `review_evidence/` | Small derived JSON, source hashes and PDF-to-repo audit |
| `scripts/check_review_evidence.py` | Independent calculation from the committed bundle |
| `scripts/export_review_evidence.py` | Local-only deterministic export from frozen sources |
| `tests/` | Focused package tests |
| `data/`, `artifacts/frozen/`, `artifacts/runs/`, `checkpoints/`, `results/` | Local inputs and large outputs; absent from the current review tree |
| `paper/` | Existing paper tables and figure assets retained from prior commits; their source provenance is recorded in the audit |

The original historical runners live in `archive/legacy/` in the full workspace and are **not** part of this Git repository. They are named in the audit for provenance. `MIGRATION_MANIFEST.json` and relocation metadata document the workspace reorganization but are not needed for the clean-clone check.

## Environment and data provenance

The package declares Python >=3.10, NumPy, scikit-learn, PyTorch and PyYAML in `pyproject.toml`; optional `data`, `plot` and `test` extras add their respective libraries. These declarations are **not a pinned CUDA/environment lockfile**. The frozen source reports record Python 3.12.3, NumPy 2.4.4 and PyTorch 2.12.1+cu130 for S20; other report environments may differ. Install the package and tests with `.venv/bin/python -m pip install -e '.[test]'` when network access is available.

The experiments used Sleep-EDF-20 (32,843 30-second epochs), Sleep-EDF-78 (195,099) and ISRUC-S3 (8,589). Sleep-EDF processing follows the author-compatible recording-wise tenfold protocol, **with subject overlap and held-out checkpoint selection**. S3 uses ten subject-disjoint folds but also held-out selection. The parent is a **self-trained** ULW-SleepNet per dataset, frozen before BIM head training. The Sleep-EDF parent has 13,337 parameters; BIM adds 3,112. S3 has 25,625 parent parameters and adds 6,184. The historical S3 cVAN source manifest identifies upstream commit `b0fa1e2d6b97d2dec1457b7572441cd01e25a333`; these are not cVAN performance results. Dataset release identifiers, redistribution rights and a fully pinned preprocessing environment are **not established** by the committed files. Consult the original dataset providers and the local cache manifests before a full replay.

## What can actually be run

From a clean clone, run the numerical check above and the focused code tests after installing `.[test]`:

```sh
.venv/bin/python -m pytest -q tests
```

With the **original frozen outputs** supplied at paths in `configs/paper/*.yaml`, these supported read-only commands run:

```sh
.venv/bin/python -m linear_bim evaluate --config configs/paper/s20.yaml
.venv/bin/python -m linear_bim audit --config configs/paper/s20.yaml
```

The `doctor` command reports missing local inputs in a clean clone and exits 2; it is a diagnostic, not a reproduction step. `evaluate`/`audit` cannot run without the ignored frozen NPZ. Head training is supported only from explicitly supplied train and held-out NPZ files containing `sequence_ids`, `epoch_indices`, `embeddings`, `u0_logits`, and `labels`; a valid invocation is:

```sh
.venv/bin/python -m linear_bim train --config configs/paper/s20.yaml --stage head --run-id local-s20-fold0 --fold 0 --device cpu --allow-cpu --train-npz /path/to/train.npz --held-npz /path/to/held.npz
```

It writes to ignored `artifacts/runs/`. This command is conditional on those NPZ inputs and **does not reconstruct the paper pipeline**. Full backbone training, raw data preparation, all tenfold checkpoint selection, external-method reproduction, and paper figure generation from a clean clone are not supported by the compact runner. Recreating the paper requires licensed/source datasets, preprocessing specifications and source files, frozen parent weights or their training pipeline, historical experiment runners, a pinned compute environment, and a reviewed end-to-end protocol with an untouched outer test if generalization is claimed. No automatic download or silent fallback is implemented.
