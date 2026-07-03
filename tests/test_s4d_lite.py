from __future__ import annotations

import torch

from src.models.s4d_lite import S4DLiteMixer


def _naive_forward(mixer: S4DLiteMixer, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, d_model = x.shape

    A_c = -torch.exp(mixer.log_A)
    delta = torch.exp(mixer.log_delta).clamp_min(1e-6)
    A_d = torch.exp(delta * A_c)
    A_c_safe = torch.where(A_c.abs() < 1e-8, torch.full_like(A_c, -1e-8), A_c)
    B_d = ((A_d - 1.0) / A_c_safe) * mixer.B

    state = torch.zeros((bsz, d_model), dtype=x.dtype, device=x.device)
    outputs = []
    for t in range(seq_len):
        x_t = x[:, t, :]
        m_t = attention_mask[:, t].to(dtype=x.dtype).unsqueeze(-1)
        state_candidate = A_d.unsqueeze(0) * state + B_d.unsqueeze(0) * x_t
        state = m_t * state_candidate + (1.0 - m_t) * state
        y_t = state * mixer.C.unsqueeze(0) + mixer.skip.unsqueeze(0) * x_t
        outputs.append(y_t)
    return torch.stack(outputs, dim=1)


def test_s4d_lite_mixer_vectorized_matches_naive() -> None:
    torch.manual_seed(123)

    bsz, seq_len, d_model = 3, 64, 32
    x = torch.randn(bsz, seq_len, d_model)
    attention_mask = (torch.rand(bsz, seq_len) > 0.2).long()

    mixer = S4DLiteMixer(d_model=d_model)

    y_fast = mixer(x, attention_mask)
    y_ref = _naive_forward(mixer, x, attention_mask)

    assert torch.allclose(y_fast, y_ref, atol=1e-5, rtol=1e-4)
