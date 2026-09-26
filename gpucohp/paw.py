"""PAW projector overlaps <p~_i^a | psi~_nk> and the PAW orthonormality test.

Conventions
-----------
Pseudo wavefunction  psi~_nk(r) = 1/sqrt(Omega) sum_G C_n(G) exp(i(k+G).r)
Atom-centred function f(r) = R(r) Y_lm(r^) at tau has plane-wave coefficients
    c(G) = (4 pi / sqrt(Omega)) (-i)^l Y_lm(q^) Rt_l(|q|) exp(-i q.tau),  q = k+G
    Rt_l(q) = int r^2 R(r) j_l(qr) dr
VASP's POTCAR tabulates  pq(q) = 4 pi Rt_l(q)  for the projectors.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from .sph import real_ylm


class PawSetup:
    """Per-structure PAW data: species per atom, projector index tables."""

    def __init__(self, structure, species):
        self.structure = structure
        self.species = species                       # list of PawSpecies (POTCAR order)
        assert len(species) == len(structure.symbols), "POTCAR/POSCAR species mismatch"
        self.splines = []
        for sp in species:
            spl = []
            for c in sp.channels:
                for ip in range(c["nproj"]):
                    spl.append(CubicSpline(sp.qgrid, c["pq"][ip]))
            self.splines.append(spl)
        # global projector list: (iatom, ispecies, iproj(channel-wise index), l, m)
        self.plist = []
        self.atom_proj_slices = []
        for ia, isp in enumerate(structure.species_of_atom):
            sp = species[isp]
            start = len(self.plist)
            for ip, (ic, l, ii) in enumerate(sp.proj_table):
                for m in range(2 * l + 1):
                    self.plist.append((ia, isp, ip, l, m))
            self.atom_proj_slices.append(slice(start, len(self.plist)))
        self.nproj = len(self.plist)
        # Q_ij expanded to lm-resolved (block diagonal in l, delta_mm')
        self.Qfull = np.zeros((self.nproj, self.nproj))
        for ia, sl in enumerate(self.atom_proj_slices):
            isp = structure.species_of_atom[ia]
            sp = species[isp]
            idx = list(range(sl.start, sl.stop))
            for a in idx:
                for b in idx:
                    _, _, ipa, la, ma = self.plist[a]
                    _, _, ipb, lb, mb = self.plist[b]
                    if la == lb and ma == mb:
                        self.Qfull[a, b] = sp.qij[ipa, ipb]

    # --------------------------------------------------------------
    def projector_pw(self, qcart, kcart):
        """Plane-wave coefficients of all projectors: array (nproj, nplane) complex.
        qcart: (nplane,3) Cartesian k+G (1/A)."""
        st = self.structure
        omega = st.volume
        qn = np.linalg.norm(qcart, axis=1)
        out = np.zeros((self.nproj, len(qcart)), dtype=complex)
        Y = {}
        lmax = max(p[3] for p in self.plist)
        for l in range(lmax + 1):
            Y[l] = real_ylm(l, qcart)                                # (npl, 2l+1)
        phase_atom = {ia: np.exp(-1j * (qcart @ st.cart[ia])) for ia in range(st.natoms)}
        for j, (ia, isp, ip, l, m) in enumerate(self.plist):
            pq = self.splines[isp][ip](qn)
            out[j] = (1.0 / np.sqrt(omega)) * (-1j) ** l * Y[l][:, m] * pq * phase_atom[ia]
        return out

    def projections(self, qcart, kcart, C):
        """P (nproj, nbands) = <p~ | psi~_n>  for coefficient matrix C (nbands, nplane)."""
        cp = self.projector_pw(qcart, kcart)
        return cp.conj() @ C.T

    def pw_overlap_matrix(self, C, P):
        """Full PAW overlap <psi_n|psi_m> = C C^H + P^H Q P."""
        return C.conj() @ C.T + P.conj().T @ self.Qfull @ P
