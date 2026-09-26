"""Tetrahedron weights and band re-orthonormalisation."""
import numpy as np
import pytest
import torch

from gpucohp.engine import _hermitian_power, _loewdin_orthonormalise
from gpucohp.tetra import state_weights


def _cubic_mesh(n):
    idx = lambda i, j, k: ((i % n) * n + (j % n)) * n + (k % n)
    cube = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    split = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    tets = np.array([[[idx(i + a, j + b, k + c) for a, b, c in cube][t] for t in s]
                     for i in range(n) for j in range(n) for k in range(n) for s in split])
    ks = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)]) / n
    return np.where(ks >= 0.5, ks - 1, ks), tets


@pytest.mark.parametrize("scheme", ["lobster", "linear", "blochl"])
def test_tetrahedron_sum_rule_and_free_electrons(scheme):
    kf, tets = _cubic_mesh(12)
    eig = np.stack([(kf ** 2).sum(1) * 10, (kf ** 2).sum(1) * 10 + 1], 1)
    E = np.array([0.6, 1.2, 1.8, 2.4, 40.0])
    W = state_weights(eig, (1 / len(tets), tets, np.ones(len(tets))), E, scheme=scheme).numpy()
    N = W.sum(axis=0)                                  # integrated DOS per band
    assert np.allclose(N[:, -1], 1.0, atol=1e-12)      # every band holds one state per k weight
    exact = 4 / 3 * np.pi * (E[:4] / 10) ** 1.5        # free-electron sphere inside the zone
    assert np.all(np.abs(N[0, :4] - exact) / exact < 0.1)


def test_tetrahedron_schemes_agree_on_totals():
    kf, tets = _cubic_mesh(8)
    eig = ((kf ** 2).sum(1) * 10)[:, None]
    E = np.linspace(0, 3, 7)
    tet = (1 / len(tets), tets, np.ones(len(tets)))
    tot = {s: state_weights(eig, tet, E, scheme=s).numpy().sum(0) for s in ("lobster", "linear", "blochl")}
    assert np.allclose(tot["lobster"], tot["linear"]) and np.allclose(tot["linear"], tot["blochl"])


def _problem(n=6, singular=False):
    torch.manual_seed(0)
    A = torch.randn(n, n, dtype=torch.complex128)
    S = A @ A.mH + n * torch.eye(n, dtype=torch.complex128)
    Cn = torch.randn(n, n, dtype=torch.complex128)
    if singular:
        Cn[:, -1] = 0.7 * Cn[:, 0]
    return 0.5 * (S + S.mH), Cn


def test_loewdin_matches_symmetric_orthonormalisation():
    S, Cn = _problem()
    Sh = _hermitian_power(S, 0.5)
    C1, _ = _loewdin_orthonormalise(Cn, S, Sh)
    ref = Cn @ _hermitian_power(Cn.mH @ S @ Cn, -0.5)
    assert torch.allclose(C1, ref, atol=1e-12)


def test_loewdin_rank_deficient_complete_and_drop():
    S, Cn = _problem(singular=True)
    Sh = _hermitian_power(S, 0.5)
    C1, sv = _loewdin_orthonormalise(Cn, S, Sh, "complete")
    assert sv.item() < 1e-10
    assert torch.allclose(C1.mH @ S @ C1, torch.eye(6, dtype=S.dtype), atol=1e-10)
    C1d, _ = _loewdin_orthonormalise(Cn, S, Sh, "drop")
    O = C1d.mH @ S @ C1d
    assert abs(torch.linalg.eigvalsh(O).min().item()) < 1e-8            # one direction left empty
