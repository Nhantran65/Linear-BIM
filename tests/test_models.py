from __future__ import annotations

import torch

from linear_bim.data.padding import pad_by_sequence, unpad_by_sequence
from linear_bim.models.backbone import ConfigurableLinearULW, LinearPaperULW, parameter_ledger
from linear_bim.models.bim import BIM, CurrentOnly, EXPECTED_LEDGER, direct_memories, masked_memories, parameter_count


def test_parameter_ledgers_and_state_dict_keys() -> None:
    assert parameter_count(CurrentOnly()) == EXPECTED_LEDGER["C0_head"]
    assert parameter_count(BIM()) == EXPECTED_LEDGER["B0_head"]
    s20 = ConfigurableLinearULW(32, 4)
    s3 = ConfigurableLinearULW(32, 10)
    assert parameter_count(s20) + parameter_count(BIM(128)) == 16_449
    assert parameter_count(s3) + parameter_count(BIM(320)) == 31_809
    assert list(LinearPaperULW(32).state_dict()) == list(s20.state_dict())
    assert parameter_ledger(s3)["dense_input_dim"] == 320


def test_masked_memories_match_direct_and_reset() -> None:
    torch.manual_seed(3)
    model = BIM().double()
    x = torch.randn(3, 7, 128, dtype=torch.float64)
    lengths = torch.tensor([7, 4, 2])
    q, values = model.features(x, lengths)
    for index, length in enumerate(lengths.tolist()):
        mf, mb = direct_memories(values["p"][index, :length], model.poles)
        torch.testing.assert_close(values["mf"][index, :length], mf, rtol=0, atol=1e-12)
        torch.testing.assert_close(values["mb"][index, :length], mb, rtol=0, atol=1e-12)
        isolated, _ = model.features(x[index:index + 1, :length], torch.tensor([length]))
        torch.testing.assert_close(q[index, :length], isolated[0], rtol=0, atol=1e-12)


def test_sequence_padding_preserves_entry_order() -> None:
    sequence_ids = torch.tensor([0, 1, 0, 1]).numpy()
    epoch_indices = torch.tensor([1, 0, 0, 1]).numpy()
    embeddings = torch.arange(16, dtype=torch.float32).reshape(4, 4).numpy()
    logits = torch.zeros(4, 5).numpy()
    labels = torch.tensor([1, 2, 3, 4]).numpy()
    padded = pad_by_sequence(sequence_ids, epoch_indices, embeddings, logits, labels)
    assert [row.tolist() for row in padded.rows] == [[2, 0], [1, 3]]
    restored = unpad_by_sequence(padded.embeddings.numpy(), padded)
    torch.testing.assert_close(torch.from_numpy(restored), torch.from_numpy(embeddings))
