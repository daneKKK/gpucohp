"""Per-k projection of PAW wavefunctions onto the local basis (numpy reference
implementation).

  T_{mu n}(k) = <chi_mu^k | psi_nk>
             = sum_G c_mu(G)^* C_n(G)  +  sum_{a,i} A_{mu,ai}(k) P_{ai,n}(k)
  c_mu(G)    = (4 pi / sqrt(Omega)) (-i)^l Y_lm(q^) Rt_mu(|q|) e^{-i q.tau_mu}
"""
import numpy as np
from .sph import real_ylm


def basis_pw(basis, structure, qcart, tables=None):
    """Plane-wave coefficients of the Bloch sums of all basis functions.
    Returns (nbasis, nplane) complex."""
    omega = structure.volume
    qn = np.linalg.norm(qcart, axis=1)
    nb = basis.nbasis
    out = np.zeros((nb, len(qcart)), dtype=complex)
    lmax = int(max(basis.l))
    Y = {l: real_ylm(l, qcart) for l in range(lmax + 1)}
    phase = {ia: np.exp(-1j * (qcart @ structure.cart[ia])) for ia in range(structure.natoms)}
    Rt = {}
    for j, f in enumerate(basis.functions):
        key = (f.element, f.radial.label)
        if key not in Rt:
            Rt[key] = f.radial.Rt(qn)
        out[j] = (4 * np.pi / np.sqrt(omega)) * (-1j) ** f.radial.l * Y[f.radial.l][:, f.m] * Rt[key] * phase[f.iatom]
    return out


def transfer_matrix(basis, structure, paw, aug, wav, ispin, ik, nbands=None):
    G = wav.gvectors(ik)
    kf = wav.kpoints[ik]
    q = (G + kf) @ wav.rec_lattice
    C = wav.coeffs(ispin, ik)
    if nbands is not None:
        C = C[:nbands]
    cmu = basis_pw(basis, structure, q)
    P = paw.projections(q, kf @ wav.rec_lattice, C)          # (nproj, nb)
    A = aug.Ak(kf)                                           # (nbasis, nproj)
    T_pw = cmu.conj() @ C.T                                  # (nbasis, nb)
    T = T_pw + A @ P
    return T, T_pw, P, A
