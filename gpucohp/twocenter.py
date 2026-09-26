"""Two-centre integrals between STO basis functions via Fourier-Bessel
expansion, and their lattice sums S(k).

S_{mu nu}(d) = int f_mu(r) f_nu(r - d) d3r
             = 8 sum_{LM} i^{l1-l2-L} G(l1m1,l2m2,LM) Y_LM(d^) I_L(d)
I_L(d)       = int_0^inf q^2 Rt_mu(q) Rt_nu(q) j_L(q d) dq

I_L(d) is tabulated on a uniform d grid per (radial pair, L) with an adaptive
q cut-off and interpolated with cubic splines for the actual inter-atomic
distances; d = 0 (on-site) is treated exactly.

Storage: the overlaps are kept as blocks S_{A0,B T} (nA x nB) for every
(atom pair, lattice image), grouped by species pair, so that memory scales
with (number of pair images) x (block size) and not with N_basis^2.
"""
import time
import numpy as np
from scipy.interpolate import CubicSpline
from .bessel import spherical_jn
from .sph import real_ylm, gaunt_real


class RadialFTTable:
    """Uniform q-grid tabulation of Rt(q) for all unique radial functions."""

    def __init__(self, radials, qmax=400.0, nq=10001):
        self.q = np.linspace(0.0, qmax, nq)
        self.dq = self.q[1] - self.q[0]
        self.keys = list(radials.keys())
        self.Rt = np.array([radials[k].Rt(self.q) for k in self.keys])   # (nrad, nq)
        self.l = np.array([radials[k].l for k in self.keys])

    def index(self, key):
        return self.keys.index(key)


class PairIntegralTable:
    """I_L(d) for one radial pair, all needed L, as cubic splines on a d grid."""

    def __init__(self, table, ir1, ir2, Lvals, dmax, dstep=0.02, tol=1e-9, chunk=512, device=None):
        q = table.q
        w = np.full_like(q, table.dq)
        w[0] = w[-1] = 0.5 * table.dq
        prod = w * q * q * table.Rt[ir1] * table.Rt[ir2]
        amax = np.abs(prod).max()
        nz = np.where(np.abs(prod) > tol * amax)[0]
        nq = min(len(q), nz[-1] + 2) if len(nz) else len(q)
        q, prod = q[:nq], prod[:nq]
        self.d = np.arange(0.0, dmax + 2 * dstep, dstep)
        self.Lvals = list(Lvals)
        self.splines = {}
        self.at_zero = {}
        use_torch = device is not None and str(device) != "cpu"
        if use_torch:
            import torch
            qt = torch.tensor(q, device=device, dtype=torch.float64)
            pt = torch.tensor(prod, device=device, dtype=torch.float64)
            dt = torch.tensor(self.d, device=device, dtype=torch.float64)
        for L in Lvals:
            vals = np.zeros(len(self.d))
            for i0 in range(0, len(self.d), chunk):
                if use_torch:
                    x = torch.outer(dt[i0:i0 + chunk], qt)
                    vals[i0:i0 + chunk] = (spherical_jn(L, x) @ pt).cpu().numpy()
                else:
                    dd = self.d[i0:i0 + chunk]
                    vals[i0:i0 + chunk] = spherical_jn(L, np.outer(dd, q)) @ prod
            self.splines[L] = CubicSpline(self.d, vals)
            self.at_zero[L] = prod.sum() if L == 0 else 0.0

    def __call__(self, L, dists):
        dists = np.asarray(dists, dtype=float)
        out = self.splines[L](dists)
        z = dists < 1e-10
        if z.any():
            out[z] = self.at_zero[L]
        return out


class TwoCenterOverlap:
    """Overlap blocks S_{A0,BT} for all atom pairs / lattice images, and S(k)."""

    def __init__(self, basis, structure, table, rmax_pair, gaunt=None, dstep=0.02, log=None, device=None):
        self.basis = basis
        self.device = device
        self.structure = structure
        self.table = table
        self.rmax_pair = rmax_pair
        self.gaunt = gaunt or gaunt_real(lmax=int(max(table.l)))
        self._log = log or (lambda *a: None)
        st = structure
        entries = st.distance_table(rmax_pair)          # (iA, iB, N, d = tau_B + T - tau_A, |d|)
        self.entries = entries
        self.ia = np.array([e[0] for e in entries])
        self.ib = np.array([e[1] for e in entries])
        self.Nimg = np.array([e[2] for e in entries])
        self.dvec = np.array([e[3] for e in entries])
        self.dist = np.array([e[4] for e in entries])
        # species layouts: local radial index list, identical for atoms of one species
        nsp = len(st.symbols)
        self.sp_layout = {}
        for s in range(nsp):
            atoms = np.where(st.species_of_atom == s)[0]
            if len(atoms) == 0:
                continue
            sl = basis.atom_slices[atoms[0]]
            self.sp_layout[s] = dict(irad=basis.iradial[sl], n=sl.stop - sl.start)
        spA = st.species_of_atom[self.ia]
        spB = st.species_of_atom[self.ib]
        self.groups = {}
        for sa in self.sp_layout:
            for sb in self.sp_layout:
                idx = np.where((spA == sa) & (spB == sb))[0]
                if len(idx) == 0:
                    continue
                nA, nB = self.sp_layout[sa]["n"], self.sp_layout[sb]["n"]
                self.groups[(sa, sb)] = dict(idx=idx, blocks=np.zeros((len(idx), nA, nB)))
        self._compute(dstep)
        self._build_index()

    # ------------------------------------------------------------------
    def _compute(self, dstep):
        tab = self.table
        Lmax = 2 * int(max(tab.l))
        Y = {L: real_ylm(L, self.dvec) for L in range(Lmax + 1)}
        nrad = len(tab.keys)
        t0 = time.time()
        cache = {}
        for ir1 in range(nrad):
            for ir2 in range(nrad):
                l1, l2 = int(tab.l[ir1]), int(tab.l[ir2])
                Lvals = [L for L in range(abs(l1 - l2), l1 + l2 + 1) if (l1 + l2 + L) % 2 == 0]
                pit = cache.get((ir2, ir1))          # I_L is symmetric in the radial pair
                for (sa, sb), g in self.groups.items():
                    mu_loc = np.where(self.sp_layout[sa]["irad"] == ir1)[0]
                    nu_loc = np.where(self.sp_layout[sb]["irad"] == ir2)[0]
                    if len(mu_loc) == 0 or len(nu_loc) == 0:
                        continue
                    if pit is None:
                        pit = PairIntegralTable(tab, ir1, ir2, Lvals, self.rmax_pair, dstep=dstep, device=self.device)
                        cache[(ir1, ir2)] = pit
                    sel = g["idx"]
                    for L in Lvals:
                        I = pit(L, self.dist[sel])
                        G = self.gaunt[(l1, l2, L)]
                        phase = ((1j) ** (l1 - l2 - L)).real
                        ang = np.einsum("abM,nM->nab", G, Y[L][sel])
                        val = 8.0 * phase * ang * I[:, None, None]          # (n, 2l1+1, 2l2+1)
                        g["blocks"][np.ix_(np.arange(len(sel)), mu_loc, nu_loc)] += val
                if pit is not None:
                    self._log(f"    radial pair {tab.keys[ir1]} x {tab.keys[ir2]} done ({time.time() - t0:.1f} s)")

    def _build_index(self):
        """Flat indices into the (nb*nb) matrix for every group block element."""
        nb = self.basis.nbasis
        starts = np.array([sl.start for sl in self.basis.atom_slices])
        for (sa, sb), g in self.groups.items():
            idx = g["idx"]
            nA, nB = g["blocks"].shape[1:]
            rows = starts[self.ia[idx]][:, None, None] + np.arange(nA)[None, :, None]
            cols = starts[self.ib[idx]][:, None, None] + np.arange(nB)[None, None, :]
            g["flat"] = (rows * nb + cols).reshape(-1)
            g["N"] = self.Nimg[idx]

    # ------------------------------------------------------------------
    def Sk(self, kfrac):
        """S(k) = sum_T e^{ik.T} S(T) for a single fractional k (numpy)."""
        nb = self.basis.nbasis
        S = np.zeros(nb * nb, dtype=complex)
        for g in self.groups.values():
            ph = np.exp(2j * np.pi * (g["N"] @ np.asarray(kfrac, dtype=float)))
            np.add.at(S, g["flat"], (ph[:, None, None] * g["blocks"]).reshape(-1))
        return S.reshape(nb, nb)

    @property
    def n_entries(self):
        return len(self.entries)
