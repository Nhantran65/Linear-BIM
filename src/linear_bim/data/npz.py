"""Strict offline NPZ loading for supplied caches and frozen outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DATA_INPUT_MISSING:{path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"DATA_SCHEMA_MISSING:{path}:{','.join(missing)}")
        return {name: archive[name].copy() for name in archive.files}
