from __future__ import annotations

import torch

from src.ssm.selective_scan_parallel import selective_scan_naive, selective_scan_parallel


def test_selective_scan_parallel_matches_naive() -> None:
    torch.manual_seed(0)
    b, l, p, n = 2, 24, 3, 5

    # Positive A values expected after discretization.
    A = torch.exp(torch.randn(b, l, p, n) * -0.1).clamp_min(1e-4)
    Bdisc = torch.randn(b, l, p, n) * 0.1
    C = torch.randn(b, l, n) * 0.1
    Dskip = torch.randn(p) * 0.1
    u = torch.randn(b, l, p)

    y_naive = selective_scan_naive(A, Bdisc, C, Dskip=Dskip, u=u)
    y_parallel = selective_scan_parallel(A, Bdisc, C, Dskip=Dskip, u=u)

    assert torch.allclose(y_parallel, y_naive, atol=1e-5, rtol=1e-4)
