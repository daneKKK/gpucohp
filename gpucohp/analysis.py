"""Energy-resolved bonding analysis from the projection results:
pDOS, pCOOP, pCOHP, COBI, their integrals and Mulliken/Loewdin populations.

Conventions (Bloch sum chi_mu^k = N^-1/2 sum_T e^{ikT} chi_mu(r - tau_mu - T)):
  real-space matrix   X_{A0,BN} = sum_k w_k e^{-ik.N} X_AB(k)
  pair contribution   v_{mu nu}(nk) = Re[ conj(C_mu n) C_nu n e^{ik.N} X_{mu0,nuN} ]
  pCOHP: X = H (Loewdin basis, C2) ; pCOOP: X = S (original basis, C1)
  COBI : X = conj(P)  with P the occupied real-space density matrix (Loewdin basis)
"""
import math
import numpy as np
import torch
from scipy.special import erf


class Pair:
    __slots__ = ("index", "iA", "iB", "cell", "dist", "orbitalwise")

    def __init__(self, index, iA, iB, cell, dist, orbitalwise):
        self.index, self.iA, self.iB, self.cell, self.dist, self.orbitalwise = index, iA, iB, tuple(cell), dist, orbitalwise


def build_pairs(structure, lobsterin):
    """Atom pairs from cohpBetween / cohpGenerator (LOBSTER semantics:
    generator lists every pair (i<=j over atoms, all images) with i in the home cell)."""
    st = structure
    pairs = []
    seen = set()

    def add(i, j, cell, orbw):
        key = (i, j, cell)
        if key in seen:
            return
        seen.add(key)
        d = st.cart[j] + np.array(cell) @ st.lattice - st.cart[i]
        pairs.append(Pair(len(pairs) + 1, i, j, cell, float(np.linalg.norm(d)), orbw))

    for (i, j, cell, orbw) in lobsterin.between:
        add(i - 1, j - 1, cell, orbw)
    for (d1, d2, ta, tb, orbw) in lobsterin.generators:
        table = st.distance_table(d2)
        cand = []
        for (a, b, N, dv, dd) in table:
            if dd < d1 or dd > d2 or dd < 1e-6:
                continue
            if a > b:
                continue
            ea, eb = st.symbol_of_atom[a], st.symbol_of_atom[b]
            if ta and tb and not ({ea, eb} == {ta, tb}):
                continue
            if ta and not tb and ta not in (ea, eb):
                continue
            cand.append((a, b, N, dd))
        # LOBSTER order: atom A, atom B, then translation (z, y, x ascending)
        cand.sort(key=lambda x: (x[0], x[1], x[2][2], x[2][1], x[2][0]))
        for a, b, N, dd in cand:
            add(a, b, N, orbw)
    return pairs


class EnergyGrid:
    def __init__(self, estart, eend, nsteps, efermi):
        # LOBSTER: step = (eend-estart)/(n-2); grid shifted so that E_F (0 eV) is a grid point
        n = nsteps
        self.step = (eend - estart) / (n - 2)
        i0 = int(math.ceil(-estart / self.step - 1e-9))
        self.emin = -i0 * self.step
        self.E = self.emin + self.step * np.arange(n)
        self.n = n
        self.efermi = efermi
        self.i0 = i0

    def gaussian(self, eps, sigma):
        """G[i, s] = g_sigma(E_i - eps_s) and its cumulative integral."""
        x = (self.E[:, None] - eps[None, :]) / sigma
        g = np.exp(-0.5 * x * x) / (sigma * math.sqrt(2 * math.pi))
        gi = 0.5 * (1.0 + erf(x / math.sqrt(2.0)))
        return g, gi


class Analysis:
    def __init__(self, engine, structure, basis, pairs, egrid, sigma, spin_factor=None, device=None):
        """sigma: LOBSTER's gaussianSmearingWidth; LOBSTER uses g(x) = exp(-x^2/sigma^2)/(sigma sqrt(pi)),
        i.e. a normal distribution with standard deviation sigma/sqrt(2)."""
        self.eng, self.st, self.basis, self.pairs, self.eg = engine, structure, basis, pairs, egrid
        self.sigma = sigma / math.sqrt(2.0)
        self.wav = engine.wav
        self.nspin = self.wav.nspin
        self.spin_factor = spin_factor if spin_factor is not None else (2.0 if self.nspin == 1 else 1.0)
        self.device = torch.device(device or engine.device)
        self.Nb = engine.Nb
        self.nb = basis.nbasis
        self.w = np.asarray(engine.kweights)
        self.efermi = self.wav.efermi
        self.tetra = None               # (volume weight, tets, multiplicity) -> tetrahedron integration
        self.tetra_scheme = "lobster"
        self._tetW = {}

    def use_tetrahedra(self, tet, scheme="lobster"):
        self.tetra, self.tetra_scheme, self._tetW = tet, scheme, {}

    def _tet_weights(self, s):
        """Integrated tetrahedron weights (nk*Nb, nE) on the energy grid, cached per spin."""
        if s not in self._tetW:
            from .tetra import state_weights
            eig = self.wav.eig[s, :, :self.Nb] - self.efermi
            W = state_weights(eig, self.tetra, self.eg.E, scheme=self.tetra_scheme, device=self.device)
            self._tetW[s] = W.reshape(-1, self.eg.n)
        return self._tetW[s]

    def _weights(self, s, k0, nkc, E):
        """Energy weights of the states of k-points k0..k0+nkc-1 (state index = k*Nb + n):
        G (nE, nst) for curves, Gi (nE, nst) for their integrals, occ_ef (nst,) up to E_F."""
        Nb, dev = self.Nb, self.device
        if self.tetra is not None:
            W = self._tet_weights(s)[k0 * Nb:(k0 + nkc) * Nb]                          # (nst, nE)
            Gi = W.T
            G = torch.diff(Gi, dim=0, prepend=Gi[:1]) / self.eg.step   # LOBSTER differentiates the integral;
            # its first grid point is 0 even when states below the window are already integrated
            return G, Gi, W[:, self.eg.i0]
        eps = torch.tensor(self.wav.eig[s, k0:k0 + nkc, :Nb].reshape(-1) - self.efermi, device=dev)
        wk = torch.tensor(np.repeat(self.w[k0:k0 + nkc], Nb), device=dev)
        sig = self.sigma
        x = (E[:, None] - eps[None, :]) / sig
        xc = x - 0.5 * self.eg.step / sig                                                # LOBSTER: curve at bin centre
        G = torch.exp(-0.5 * xc * xc) / (sig * math.sqrt(2 * math.pi)) * wk[None, :]
        Gi = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) * wk[None, :]
        occ_ef = 0.5 * (1.0 + torch.erf(-eps / (sig * math.sqrt(2.0)))) * wk
        return G, Gi, occ_ef

    # ------------------------------------------------------------------
    def _stack(self, s, key):
        return np.stack([self.eng.results[s][ik][key] for ik in range(self.wav.nkpts)])

    def realspace_blocks(self, s, key, occupied_density=False):
        """For every pair: real-space block X_{A0,BN} (nA x nB).
        key: 'H2' or 'S' matrices, or 'P2' (occupied Loewdin density matrix)."""
        nk = self.wav.nkpts
        kf = self.wav.kpoints
        out = []
        if key == "P2":
            C2 = self._stack(s, "C2")                                   # (nk, nb, Nb)
            occ = getattr(self, "occ_override", None)
            f = (self.wav.occ[s, :, : self.Nb] if occ is None else occ[s]) * self.spin_factor   # (nk, Nb)
            Pk = np.einsum("kmn,kn,kln->kml", C2, f, np.conj(C2))       # (nk, nb, nb)  P(k) = sum_n f C C^*
            X = Pk
        else:
            X = self._stack(s, key)                                     # (nk, nb, nb)
        for p in self.pairs:
            ph = np.exp(-2j * np.pi * (kf @ np.array(p.cell)))         # e^{-ik.N}
            slA, slB = self.basis.atom_slices[p.iA], self.basis.atom_slices[p.iB]
            blk = np.einsum("k,k,kab->ab", self.w, ph, X[:, slA, slB])
            out.append(blk)
        return out

    # ------------------------------------------------------------------
    def bonding(self, kind, s, chunk=128):
        """Energy-resolved curves for all pairs.
        kind in {'cohp','coop','cobi'}. Returns dict with per-pair arrays:
          'atom': (npairs, nE), 'atom_int': (npairs, nE),
          'orb': list of (nA, nB, nE) or None, 'orb_int': same,
          'at_ef': (npairs,), 'orb_at_ef': list of (nA,nB) or None."""
        Ckey = {"cohp": "C2", "coop": "C1", "cobi": "C2"}[kind]
        if kind == "cohp":
            blocks = self.realspace_blocks(s, "H2")
        elif kind == "coop":
            blocks = self.realspace_blocks(s, "S")
        else:
            blocks = [np.conj(b) for b in self.realspace_blocks(s, "P2")]
        nk, Nb, nE = self.wav.nkpts, self.Nb, self.eg.n
        dev = self.device
        kf = self.wav.kpoints
        eig = self.wav.eig[s, :, :Nb] - self.efermi                     # (nk, Nb) relative to E_F
        E = torch.tensor(self.eg.E, device=dev)
        npairs = len(self.pairs)
        acc = torch.zeros((npairs, nE), device=dev, dtype=torch.float64)
        acc_i = torch.zeros((npairs, nE), device=dev, dtype=torch.float64)
        acc_ef = np.zeros(npairs)
        orb = [torch.zeros((b.shape[0], b.shape[1], nE), device=dev, dtype=torch.float64) if p.orbitalwise else None
               for p, b in zip(self.pairs, blocks)]
        orb_i = [torch.zeros_like(o) if o is not None else None for o in orb]
        orb_ef = [np.zeros(b.shape) if p.orbitalwise else None for p, b in zip(self.pairs, blocks)]
        blocks_t = [torch.tensor(b, device=dev) for b in blocks]
        sig = self.sigma
        for k0 in range(0, nk, chunk):
            ks = range(k0, min(nk, k0 + chunk))
            C = torch.tensor(np.stack([self.eng.results[s][ik][Ckey] for ik in ks]), device=dev)  # (nkc, nb, Nb)
            G, Gi, occ_ef = self._weights(s, k0, len(ks), E)                                        # (nE, nst), (nst,)
            kcell = torch.tensor(kf[k0:k0 + len(ks)], device=dev)
            for ip, p in enumerate(self.pairs):
                slA, slB = self.basis.atom_slices[p.iA], self.basis.atom_slices[p.iB]
                ph = torch.exp(2j * math.pi * (kcell @ torch.tensor(np.array(p.cell, float), device=dev)))  # e^{ik.N}
                CA = C[:, slA, :]                                       # (nkc, nA, Nb)
                CB = C[:, slB, :]
                v = torch.real(torch.conj(CA)[:, :, None, :] * CB[:, None, :, :] * blocks_t[ip][None, :, :, None]
                               * ph[:, None, None, None])               # (nkc, nA, nB, Nb)
                v = v.permute(1, 2, 0, 3).reshape(v.shape[1], v.shape[2], -1)   # (nA, nB, nst)
                vsum = v.sum(dim=(0, 1))                                 # (nst,)
                acc[ip] += G @ vsum
                acc_i[ip] += Gi @ vsum
                acc_ef[ip] += float(vsum @ occ_ef)
                if p.orbitalwise:
                    orb[ip] += torch.einsum("es,abs->abe", G, v)
                    orb_i[ip] += torch.einsum("es,abs->abe", Gi, v)
                    orb_ef[ip] += (v @ occ_ef).cpu().numpy()
        sf = self.spin_factor
        return dict(atom=(acc * sf).cpu().numpy(), atom_int=(acc_i * sf).cpu().numpy(), at_ef=acc_ef * sf,
                    orb=[(o * sf).cpu().numpy() if o is not None else None for o in orb],
                    orb_int=[(o * sf).cpu().numpy() if o is not None else None for o in orb_i],
                    orb_at_ef=[o * sf if o is not None else None for o in orb_ef])

    # ------------------------------------------------------------------
    def dos(self, s, chunk=256, loewdin=False):
        """Mulliken pDOS per basis function (nb, nE) and its integral; total DOS."""
        nk, Nb, nE = self.wav.nkpts, self.Nb, self.eg.n
        dev = self.device
        eig = self.wav.eig[s, :, :Nb] - self.efermi
        E = torch.tensor(self.eg.E, device=dev)
        pdos = torch.zeros((self.nb, nE), device=dev, dtype=torch.float64)
        pdos_i = torch.zeros_like(pdos)
        tot = torch.zeros(nE, device=dev, dtype=torch.float64)
        tot_i = torch.zeros_like(tot)
        sig = self.sigma
        for k0 in range(0, nk, chunk):
            ks = range(k0, min(nk, k0 + chunk))
            if loewdin:
                C2 = torch.tensor(np.stack([self.eng.results[s][ik]["C2"] for ik in ks]), device=dev)
                q = torch.real(torch.conj(C2) * C2)                                     # (nkc, nb, Nb)
            else:
                C1 = torch.tensor(np.stack([self.eng.results[s][ik]["C1"] for ik in ks]), device=dev)
                S = torch.tensor(np.stack([self.eng.results[s][ik]["S"] for ik in ks]), device=dev)
                q = torch.real(torch.conj(C1) * (S @ C1))                               # Mulliken weights
            q = q.permute(1, 0, 2).reshape(self.nb, -1)                                 # (nb, nst)
            G, Gi, _ = self._weights(s, k0, len(ks), E)
            pdos += q @ G.T
            pdos_i += q @ Gi.T
            tot += G.sum(dim=1)
            tot_i += Gi.sum(dim=1)
        sf = self.spin_factor
        return dict(pdos=(pdos * sf).cpu().numpy(), pdos_int=(pdos_i * sf).cpu().numpy(),
                    total=(tot * sf).cpu().numpy(), total_int=(tot_i * sf).cpu().numpy())

    # ------------------------------------------------------------------
    def occupations(self, s):
        """Band occupations (nk, Nb) implied by the energy integration: tetrahedron
        weights up to E_F divided by the k weights, or the Gaussian step at E_F."""
        Nb = self.Nb
        if self.tetra is not None:
            W = self._tet_weights(s)[:, self.eg.i0].cpu().numpy().reshape(-1, Nb)
            return W / self.w[:, None]
        eps = self.wav.eig[s, :, :Nb] - self.efermi
        return 0.5 * (1.0 + erf(-eps / (self.sigma * math.sqrt(2.0))))

    def populations(self, s):
        """Mulliken and Loewdin gross populations per basis function (nb,)."""
        nk, Nb = self.wav.nkpts, self.Nb
        mull = np.zeros(self.nb)
        loew = np.zeros(self.nb)
        occ = self.occ_override[s] if getattr(self, "occ_override", None) is not None else None
        for ik in range(nk):
            r = self.eng.results[s][ik]
            f = self.wav.occ[s, ik, :Nb] if occ is None else occ[ik]
            wf = self.w[ik] * f
            C1, S, C2 = r["C1"], r["S"], r["C2"]
            mull += np.real(np.conj(C1) * (S @ C1)) @ wf
            loew += np.real(np.conj(C2) * C2) @ wf
        return mull * self.spin_factor, loew * self.spin_factor

    def atom_populations(self, mull, loew):
        nat = self.st.natoms
        m = np.array([mull[self.basis.atom_slices[a]].sum() for a in range(nat)])
        l = np.array([loew[self.basis.atom_slices[a]].sum() for a in range(nat)])
        return m, l
