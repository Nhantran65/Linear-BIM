"""Padding helpers that preserve subject/entry order and identity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PaddedSequences:
    """Padded arrays plus the exact flat row mapping used to create them."""

    embeddings: torch.Tensor
    u0_logits: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor
    rows: tuple[np.ndarray, ...]
    n_flat: int


def _ordered_groups(subjects: np.ndarray, epoch_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    if len(subjects) != len(epoch_indices):
        raise ValueError("subjects and epoch_indices must have equal length")
    groups = []
    for subject in sorted(np.unique(subjects).tolist()):
        index = np.flatnonzero(subjects == subject)
        order = np.argsort(epoch_indices[index], kind="stable")
        groups.append(index[order])
    if not groups:
        raise ValueError("cannot pad an empty dataset")
    return tuple(groups)


def pad_by_sequence(
    sequence_ids: np.ndarray,
    epoch_indices: np.ndarray,
    embeddings: np.ndarray,
    u0_logits: np.ndarray,
    labels: np.ndarray,
    device: torch.device | str = "cpu",
) -> PaddedSequences:
    """Pad each recording/entry sequence without blind concatenation.

    Inputs are flat rows. Rows are grouped by sorted recording/entry ID and
    ordered by stable ``epoch_indices`` within each sequence. The mapping is
    retained so predictions can be returned to the original flat order.
    """

    arrays = (sequence_ids, epoch_indices, embeddings, u0_logits, labels)
    if any(len(value) != len(sequence_ids) for value in arrays[1:]):
        raise ValueError("all flat arrays must have equal length")
    if embeddings.ndim != 2 or u0_logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("embeddings/logits/labels must be rank 2/2/1")
    if not np.isfinite(embeddings).all() or not np.isfinite(u0_logits).all():
        raise ValueError("DATA_NONFINITE_EMBEDDINGS_OR_LOGITS")
    try:
        identities = set(zip(sequence_ids.tolist(), epoch_indices.tolist(), strict=True))
    except TypeError as error:
        raise ValueError("DATA_SEQUENCE_IDENTITIES_NOT_HASHABLE") from error
    if len(identities) != len(sequence_ids):
        raise ValueError("DATA_DUPLICATE_SEQUENCE_EPOCH_IDENTITY")
    rows = _ordered_groups(sequence_ids, epoch_indices)
    maximum = max(len(row) for row in rows)
    embedding_pad = np.zeros((len(rows), maximum, embeddings.shape[1]), dtype=embeddings.dtype)
    logits_pad = np.zeros((len(rows), maximum, u0_logits.shape[1]), dtype=u0_logits.dtype)
    labels_pad = np.zeros((len(rows), maximum), dtype=labels.dtype)
    lengths = []
    for index, row in enumerate(rows):
        lengths.append(len(row))
        embedding_pad[index, :len(row)] = embeddings[row]
        logits_pad[index, :len(row)] = u0_logits[row]
        labels_pad[index, :len(row)] = labels[row]
    target = torch.device(device)
    length_tensor = torch.as_tensor(lengths, dtype=torch.long, device=target)
    mask = torch.arange(maximum, device=target)[None] < length_tensor[:, None]
    return PaddedSequences(
        embeddings=torch.as_tensor(embedding_pad, device=target),
        u0_logits=torch.as_tensor(logits_pad, device=target),
        labels=torch.as_tensor(labels_pad, device=target),
        lengths=length_tensor,
        mask=mask,
        rows=rows,
        n_flat=len(labels),
    )


def unpad_by_sequence(value: np.ndarray, padded: PaddedSequences) -> np.ndarray:
    """Restore ``[N,...]`` flat order from a padded ``[subjects,steps,...]`` value."""

    if value.ndim < 2 or value.shape[0] != len(padded.rows):
        raise ValueError("padded value has incompatible subject/step dimensions")
    output = np.empty((padded.n_flat,) + value.shape[2:], dtype=value.dtype)
    for index, row in enumerate(padded.rows):
        output[row] = value[index, :len(row)]
    return output
