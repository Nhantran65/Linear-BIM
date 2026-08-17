"""Current-only and CUDA-scanned Bidirectional Innovation Memory heads."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

try:  # Optional accelerator dependency; CPU remains the reference path.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:  # pragma: no cover - requires Triton/CUDA

    @triton.jit
    def _iir_forward_kernel(
        p, lengths, lambdas, output, steps: tl.constexpr, width: tl.constexpr,
        poles: tl.constexpr, reverse: tl.constexpr,
    ):
        program = tl.program_id(0)
        hidden = program % width
        pole = (program // width) % poles
        batch = program // (width * poles)
        length = tl.load(lengths + batch)
        lam = tl.load(lambdas + pole)
        state = 0.0
        for offset in tl.range(0, steps, loop_unroll_factor=1):
            active = offset < length
            time = length - 1 - offset if reverse else offset
            p_index = (batch * steps + time) * width + hidden
            value = tl.load(p + p_index, mask=active, other=0.0)
            state = tl.where(active, lam * state + (1.0 - lam) * value, state)
            out_index = ((batch * steps + time) * poles + pole) * width + hidden
            tl.store(output + out_index, state, mask=active)

    @triton.jit
    def _iir_backward_kernel(
        p, lengths, lambdas, memory, grad_memory, grad_p_part, grad_l_part,
        steps: tl.constexpr, width: tl.constexpr, poles: tl.constexpr,
        reverse: tl.constexpr,
    ):
        program = tl.program_id(0)
        hidden = program % width
        pole = (program // width) % poles
        batch = program // (width * poles)
        length = tl.load(lengths + batch)
        lam = tl.load(lambdas + pole)
        carry = 0.0
        grad_l = 0.0
        for offset in tl.range(0, steps, loop_unroll_factor=1):
            active = offset < length
            time = offset if reverse else length - 1 - offset
            memory_index = ((batch * steps + time) * poles + pole) * width + hidden
            direct = tl.load(grad_memory + memory_index, mask=active, other=0.0)
            total = direct + carry
            p_index = (batch * steps + time) * width + hidden
            value = tl.load(p + p_index, mask=active, other=0.0)
            neighbor_time = time + 1 if reverse else time - 1
            neighbor_valid = active & (neighbor_time >= 0) & (neighbor_time < length)
            neighbor_index = ((batch * steps + neighbor_time) * poles + pole) * width + hidden
            neighbor = tl.load(memory + neighbor_index, mask=neighbor_valid, other=0.0)
            grad_l += tl.where(active, total * (neighbor - value), 0.0)
            part_index = ((batch * steps + time) * poles + pole) * width + hidden
            tl.store(grad_p_part + part_index, (1.0 - lam) * total, mask=active)
            carry = tl.where(active, lam * total, carry)
        tl.store(grad_l_part + (batch * poles + pole) * width + hidden, grad_l)


def direct_memories(
    p: torch.Tensor, lambdas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference scan for a single unpadded ``[T, H]`` input."""

    steps, width = p.shape
    forward = []
    state = torch.zeros(3, width, dtype=p.dtype, device=p.device)
    for time in range(steps):
        state = lambdas[:, None] * state + (1.0 - lambdas[:, None]) * p[time]
        forward.append(state)
    backward: list[torch.Tensor] = [torch.empty(0, device=p.device)] * steps
    state = torch.zeros(3, width, dtype=p.dtype, device=p.device)
    for time in range(steps - 1, -1, -1):
        state = lambdas[:, None] * state + (1.0 - lambdas[:, None]) * p[time]
        backward[time] = state
    return torch.stack(forward), torch.stack(backward)


class _CudaIIR(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, p: torch.Tensor, lengths: torch.Tensor, lambdas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, width = p.shape
        poles = int(lambdas.numel())
        mf = torch.zeros(batch, steps, poles, width, dtype=p.dtype, device=p.device)
        mb = torch.zeros_like(mf)
        grid = (batch * poles * width,)
        _iir_forward_kernel[grid](p, lengths, lambdas, mf, steps=steps, width=width, poles=poles, reverse=False)
        _iir_forward_kernel[grid](p, lengths, lambdas, mb, steps=steps, width=width, poles=poles, reverse=True)
        ctx.save_for_backward(p, lengths, lambdas, mf, mb)
        return mf, mb

    @staticmethod
    def backward(
        ctx: Any, grad_mf: torch.Tensor, grad_mb: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        p, lengths, lambdas, mf, mb = ctx.saved_tensors
        batch, steps, width = p.shape
        poles = int(lambdas.numel())
        shape = (batch, steps, poles, width)
        pf = torch.zeros(shape, dtype=p.dtype, device=p.device)
        pb = torch.zeros_like(pf)
        lf = torch.zeros(batch, poles, width, dtype=p.dtype, device=p.device)
        lb = torch.zeros_like(lf)
        grid = (batch * poles * width,)
        _iir_backward_kernel[grid](p, lengths, lambdas, mf, grad_mf.contiguous(), pf, lf, steps=steps, width=width, poles=poles, reverse=False)
        _iir_backward_kernel[grid](p, lengths, lambdas, mb, grad_mb.contiguous(), pb, lb, steps=steps, width=width, poles=poles, reverse=True)
        return (pf + pb).sum(dim=2), None, (lf + lb).sum(dim=(0, 2))


def masked_memories(
    p: torch.Tensor, lengths: torch.Tensor, lambdas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan padded sequences while resetting state after each valid length."""

    if p.ndim != 3:
        raise ValueError(f"expected [batch, steps, width], got {tuple(p.shape)}")
    if lengths.ndim != 1 or lengths.shape[0] != p.shape[0]:
        raise ValueError("lengths must be one value per batch row")
    if torch.any(lengths < 0) or torch.any(lengths > p.shape[1]):
        raise ValueError("sequence lengths must be within padded sequence bounds")
    if p.is_cuda:
        if triton is None or p.dtype != torch.float32:
            raise RuntimeError("CUDA BIM requires Triton float32")
        return _CudaIIR.apply(p.contiguous(), lengths.contiguous(), lambdas.contiguous())
    batch, steps, width = p.shape
    valid = torch.arange(steps, device=p.device)[None] < lengths[:, None]
    mf_rows = []
    state = torch.zeros(batch, 3, width, dtype=p.dtype, device=p.device)
    for time in range(steps):
        updated = lambdas[None, :, None] * state + (1.0 - lambdas[None, :, None]) * p[:, time, None]
        state = torch.where(valid[:, time, None, None], updated, torch.zeros_like(updated))
        mf_rows.append(state)
    mb_rows: list[torch.Tensor] = [torch.empty(0, device=p.device)] * steps
    state = torch.zeros(batch, 3, width, dtype=p.dtype, device=p.device)
    for time in range(steps - 1, -1, -1):
        updated = lambdas[None, :, None] * state + (1.0 - lambdas[None, :, None]) * p[:, time, None]
        state = torch.where(valid[:, time, None, None], updated, torch.zeros_like(updated))
        mb_rows[time] = state
    return torch.stack(mf_rows, 1), torch.stack(mb_rows, 1)


class CurrentOnly(nn.Module):
    def __init__(self, input_dim: int = 128) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, 16)
        self.residual = nn.Linear(16, 5)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        return self.residual(torch.tanh(self.projection(embeddings)))


class BIM(nn.Module):
    def __init__(self, input_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, 16)
        initial = torch.exp(-1.0 / torch.tensor([2.0, 4.0, 8.0]))
        self.beta = nn.Parameter(torch.logit(initial))
        self.residual = nn.Linear(208, 5)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @property
    def poles(self) -> torch.Tensor:
        return torch.sigmoid(self.beta)

    def features(
        self, embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p = torch.tanh(self.projection(embeddings))
        mf, mb = masked_memories(p, lengths, self.poles)
        expanded = p[:, :, None]
        df, db = expanded - mf, expanded - mb
        valid = torch.arange(p.shape[1], device=p.device)[None] < lengths[:, None]
        mask = valid[:, :, None, None]
        mf, mb, df, db = (torch.where(mask, value, torch.zeros_like(value)) for value in (mf, mb, df, db))
        q = torch.cat((p, mf.flatten(2), mb.flatten(2), df.flatten(2), db.flatten(2)), -1)
        q = torch.where(valid[:, :, None], q, torch.zeros_like(q))
        return q, {"p": p, "mf": mf, "mb": mb, "df": df, "db": db, "valid": valid}

    def forward(
        self, embeddings: torch.Tensor, lengths: torch.Tensor, context_zero: bool = False
    ) -> torch.Tensor:
        q, _ = self.features(embeddings, lengths)
        if context_zero:
            q = torch.cat((q[..., :16], torch.zeros_like(q[..., 16:])), -1)
        return self.residual(q)

    def effective_tau(self) -> torch.Tensor:
        return -1.0 / torch.log(self.poles)


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


EXPECTED_LEDGER = {"C0_head": 2_149, "C0_complete": 15_486, "B0_head": 3_112, "B0_complete": 16_449}
