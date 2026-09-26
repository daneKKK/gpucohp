"""Linear tetrahedron integration on VASP's tetrahedron list.

VASP writes the tetrahedra it used (ISMEAR = -5) to vasprun.xml: a common
volume weight and, per tetrahedron, a multiplicity and four k-point indices.
For a state n and energy E, the integrated weight of corner k of a
tetrahedron follows Bloechl, Jepsen and Andersen, Phys. Rev. B 49, 16223
(1994), Appendix B, optionally with their correction term. LOBSTER instead
multiplies the occupied fraction of a tetrahedron by the mean of its four
corner values (scheme "lobster", the default here; checked against LOBSTER
5.1.1 COHPCAR curves to 5e-4). Summing the corner
weights over all tetrahedra that contain k gives W_nk(E), which replaces
w_k * Theta(E - eps_nk) of Gaussian integration: sum_nk W_nk(E) q_nk is the
integral of q up to E. Curves are obtained from it by differentiation.
"""
import re
import numpy as np
import torch


def read_tetrahedra(vasprun="vasprun.xml"):
    """(volume weight, tets (ntet, 4) 0-based k indices, multiplicity (ntet,)) or None."""
    txt = open(vasprun).read()
    m = re.search(r'<varray name="tetrahedronlist"[^>]*>(.*?)</varray>', txt, re.S)
    if not m:
        return None
    blk = m.group(1)
    vw = float(re.search(r'<i name="volumeweight"\s*>\s*(\S+)\s*</i>', txt).group(1))
    rows = np.array([[int(x) for x in v.split()] for v in re.findall(r"<v[^>]*>(.*?)</v>", blk)])
    return vw, rows[:, 1:5] - 1, rows[:, 0].astype(float)


def _corner_weights(e, E, blochl):
    """Integrated corner weights for sorted corner energies e (..., 4) at energies E (nE,).
    Returns (..., 4, nE) for a tetrahedron of unit volume."""
    e1, e2, e3, e4 = (e[..., i, None] for i in range(4))             # (..., 1)
    x = E                                                           # (nE,)
    z = torch.zeros(e.shape[:-1] + (E.shape[0],), dtype=e.dtype, device=e.device)
    w = [z.clone() for _ in range(4)]
    d21, d31, d41, d32, d42, d43 = e2 - e1, e3 - e1, e4 - e1, e3 - e2, e4 - e2, e4 - e3
    # region 1: e1 <= E < e2
    r = (x >= e1) & (x < e2)
    c = 0.25 * (x - e1) ** 3 / (d21 * d31 * d41)
    w1 = c * (4 - (x - e1) * (1 / d21 + 1 / d31 + 1 / d41))
    ws = [w1, c * (x - e1) / d21, c * (x - e1) / d31, c * (x - e1) / d41]
    dos = 3 * (x - e1) ** 2 / (d21 * d31 * d41)
    D = torch.where(r, dos, z)
    for i in range(4):
        w[i] = torch.where(r, ws[i], w[i])
    # region 2: e2 <= E < e3
    r = (x >= e2) & (x < e3)
    c1 = 0.25 * (x - e1) ** 2 / (d41 * d31)
    c2 = 0.25 * (x - e1) * (x - e2) * (e3 - x) / (d41 * d32 * d31)
    c3 = 0.25 * (x - e2) ** 2 * (e4 - x) / (d42 * d32 * d41)
    ws = [c1 + (c1 + c2) * (e3 - x) / d31 + (c1 + c2 + c3) * (e4 - x) / d41,
          c1 + c2 + c3 + (c2 + c3) * (e3 - x) / d32 + c3 * (e4 - x) / d42,
          (c1 + c2) * (x - e1) / d31 + (c2 + c3) * (x - e2) / d32,
          (c1 + c2 + c3) * (x - e1) / d41 + c3 * (x - e2) / d42]
    dos = (3 * d21 + 6 * (x - e2) - 3 * (d31 + d42) * (x - e2) ** 2 / (d32 * d42)) / (d31 * d41)
    D = torch.where(r, dos, D)
    for i in range(4):
        w[i] = torch.where(r, ws[i], w[i])
    # region 3: e3 <= E < e4
    r = (x >= e3) & (x < e4)
    c = 0.25 * (e4 - x) ** 3 / (d41 * d42 * d43)
    ws = [0.25 - c * (e4 - x) / d41, 0.25 - c * (e4 - x) / d42, 0.25 - c * (e4 - x) / d43,
          0.25 - c * (4 - (e4 - x) * (1 / d41 + 1 / d42 + 1 / d43))]
    dos = 3 * (e4 - x) ** 2 / (d41 * d42 * d43)
    D = torch.where(r, dos, D)
    for i in range(4):
        w[i] = torch.where(r, ws[i], w[i])
    # region 4: E >= e4
    r = x >= e4
    for i in range(4):
        w[i] = torch.where(r, torch.full_like(z, 0.25), w[i])
    if blochl:
        es = [e1, e2, e3, e4]
        for i in range(4):
            w[i] = w[i] + D / 40.0 * sum(es[j] - es[i] for j in range(4))
    return torch.stack(w, dim=-2)


def state_weights(eig, tet, energies, scheme="lobster", device="cpu", max_elems=8_000_000):
    """Integrated tetrahedron weights W[k, n, E] (nk, Nb, nE), sum_k W -> 1 per band.

    eig: (nk, Nb) band energies; tet = (volume weight, (ntet, 4) indices, multiplicity);
    energies: (nE,) grid on the same energy scale as eig.
    scheme: "lobster" - occupied fraction of each tetrahedron times the mean of
              its corner values (what LOBSTER 5.1.1 does; reproduces its curves);
            "linear"  - linear interpolation of the quantity inside the tetrahedron;
            "blochl"  - linear plus Bloechl's correction."""
    vw, idx, mult = tet
    dev = torch.device(device)
    eig_t = torch.tensor(eig, device=dev, dtype=torch.float64)
    idx_t = torch.tensor(idx, device=dev, dtype=torch.long)
    vol = torch.tensor(vw * mult, device=dev, dtype=torch.float64)                # (ntet,)
    E = torch.tensor(energies, device=dev, dtype=torch.float64)
    nk, nb = eig.shape
    W = torch.zeros((nk, nb, len(energies)), device=dev, dtype=torch.float64)
    e = eig_t[idx_t]                                                              # (ntet, 4, Nb)
    e = e.permute(0, 2, 1)                                                        # (ntet, Nb, 4)
    es, order = torch.sort(e, dim=-1)
    es = es + torch.arange(4, device=dev, dtype=torch.float64) * 1e-9           # lift exact degeneracies
    kk = torch.gather(idx_t[:, None, :].expand(-1, nb, -1), 2, order)            # (ntet, Nb, 4) k of sorted corner
    band = torch.arange(nb, device=dev)[None, :, None].expand_as(kk)
    flat = (kk * nb + band).reshape(-1)                                           # (ntet*Nb*4,)
    echunk = max(1, int(max_elems // (idx.shape[0] * nb * 4)))                   # ~20 temporaries of this size
    for a in range(0, len(energies), echunk):
        Ec = E[a:a + echunk]
        w = _corner_weights(es, Ec, scheme == "blochl") * vol[:, None, None, None]   # (ntet, Nb, 4, nEc)
        if scheme == "lobster":
            w = w.sum(dim=2, keepdim=True).expand_as(w) / 4.0                       # corner-averaged quantity
        Wc = torch.zeros((nk * nb, len(Ec)), device=dev, dtype=torch.float64)
        Wc.index_add_(0, flat, w.reshape(-1, len(Ec)))
        W[:, :, a:a + echunk] = Wc.reshape(nk, nb, -1)
    return W
