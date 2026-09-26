"""Torch-based projection engine: loops over spins and k-points, projects the
PAW wavefunctions onto the local basis and reconstructs the LCAO Hamiltonian.

Per (spin, k) the following is produced (LOBSTER's scheme, Maintz et al.
J. Comput. Chem. 2013/2016):

  T  = <chi_mu^k | psi_n>                       (nbasis x Nb)   transfer matrix
  S  = <chi_mu^k | chi_nu^k>                    (nbasis x nbasis)
  C  = S^-1 T                                    coefficients of projected bands
  spill_n = 1 - Re(T_n^H C_n)
  O  = C^H S C ;  C1 = C O^-1/2                  re-orthonormalised projected bands
  H  = S C1 E C1^H S                             Hamiltonian in the non-orthogonal basis
  Loewdin basis:  C2 = S^1/2 C1 ,  H2 = S^-1/2 H S^-1/2 = S^1/2 C1 E C1^H S^1/2

Stored per (spin,k): eig[Nb], C1, C2, S, H2 (and H = S^1/2 H2 S^1/2 on demand).
"""
import cmath
import math
import time
import numpy as np
import torch
from .sph import real_ylm
from .constants import TPI


def _hermitian_power(M, power, eps=1e-12):
    """M^power for a batch of Hermitian positive matrices (torch)."""
    w, V = torch.linalg.eigh(M)
    w = torch.clamp(w, min=eps)
    return (V * w.pow(power).unsqueeze(-2).to(V.dtype)) @ V.mH


def _loewdin_orthonormalise(Cn, S, Sh, mode="complete", tol=1e-12):
    """Re-orthonormalise projected bands: C1 = Cn O^(-1/2), O = Cn^H S Cn.

    Regular k-points (the vast majority) use the eigendecomposition of O. Where
    O is singular - two retained bands project onto the same local combination
    and leave a basis direction uncovered - the polar factor of X = S^(1/2) Cn
    is used instead (X = U s V^H -> C1 = S^(-1/2) U V^H): identical for regular
    O, and it fills the uncovered direction with the orthogonal complement,
    which is what LOBSTER reaches by normalising the numerical remainder
    (mode "complete"); mode "drop" leaves it empty. Works on single (n, n) or
    batched (B, n, n) inputs. Returns C1 and sqrt(min eig O), a rank-deficiency
    indicator (the smallest singular value of X)."""
    O = Cn.mH @ S @ Cn
    O = 0.5 * (O + O.mH)
    w, V = torch.linalg.eigh(O)
    wmin = w[..., 0].clamp(min=0).sqrt()
    C1 = Cn @ ((V * w.clamp(min=tol).rsqrt().unsqueeze(-2).to(V.dtype)) @ V.mH)
    bad = w[..., 0] < tol * w[..., -1]
    if bad.any():
        single = Cn.dim() == 2
        Xb = (Sh @ Cn).unsqueeze(0) if single else (Sh @ Cn)[bad]
        U, sv, Vh = torch.linalg.svd(Xb, full_matrices=False)
        if mode == "drop":
            keep = (sv > 1e-6 * sv[..., :1]).to(Xb.dtype)
            W = (U * keep.unsqueeze(-2)) @ Vh + U @ ((sv.to(Xb.dtype) * (1 - keep)).unsqueeze(-1) * Vh)
        else:
            W = U @ Vh
        Shb = Sh.unsqueeze(0) if single else (Sh[bad] if Sh.dim() == 3 else Sh)
        fix = torch.linalg.solve(Shb, W)
        if single:
            C1 = fix[0]
        else:
            C1 = C1.clone()
            C1[bad] = fix
    return C1, wmin


class ProjectionEngine:
    def __init__(self, wav, structure, basis, paw, aug, overlap, device="cpu", dtype=torch.complex128,
                 nbands_use=None, log=print, rank_deficient="complete"):
        self.wav, self.st, self.basis, self.paw, self.aug, self.ov = wav, structure, basis, paw, aug, overlap
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.log = log
        self.rank_mode = rank_deficient
        nb = basis.nbasis
        self.nbasis = nb
        if nbands_use is None:
            nbands_use = min(nb, wav.nbands)
        if nbands_use > nb:
            raise ValueError("cannot use more bands than basis functions (Hamiltonian would be undetermined)")
        self.Nb = nbands_use
        self._prepare_tables()

    # ------------------------------------------------------------------
    def _prepare_tables(self):
        dev, rd, cd = self.device, self.rdtype, self.cdtype
        st, basis, paw = self.st, self.basis, self.paw
        gcut = math.sqrt(self.wav.encut / 3.80998212) * 1.02 + 0.5
        # radial FT tables of basis functions on a fine uniform q grid (linear interpolation on device)
        nq = 20001
        self.qtab = torch.linspace(0.0, gcut, nq, device=dev, dtype=rd)
        qnp = self.qtab.cpu().numpy()
        Rt = np.array([basis.radials[k].Rt(qnp) for k in basis.radial_keys])
        self.Rt_tab = torch.tensor(Rt, device=dev, dtype=rd)                      # (nrad, nq)
        pq = []
        self.proj_species_index = []
        for isp, sp in enumerate(paw.species):
            for ip in range(sp.nproj_total):
                pq.append(paw.splines[isp][ip](qnp))
        self.pq_tab = torch.tensor(np.array(pq), device=dev, dtype=rd)            # (nproj_types, nq)
        offs = np.cumsum([0] + [sp.nproj_total for sp in paw.species])
        self.proj_type = torch.tensor([offs[isp] + ip for (ia, isp, ip, l, m) in paw.plist], device=dev)
        self.proj_atom = torch.tensor([p[0] for p in paw.plist], device=dev)
        self.proj_l = torch.tensor([p[3] for p in paw.plist], device=dev)
        self.proj_m = torch.tensor([p[4] for p in paw.plist], device=dev)
        self.bas_rad = torch.tensor(basis.iradial, device=dev)
        self.bas_atom = torch.tensor(basis.iatom, device=dev)
        self.bas_l = torch.tensor(basis.l, device=dev)
        self.bas_m = torch.tensor(basis.m, device=dev)
        self.tau = torch.tensor(st.cart, device=dev, dtype=rd)
        self.Qfull = torch.tensor(paw.Qfull, device=dev, dtype=cd)
        self.rec = torch.tensor(self.wav.rec_lattice, device=dev, dtype=rd)
        self.omega = st.volume
        self.lmax = int(max(int(basis.l.max()), max(p[3] for p in paw.plist)))
        # overlap lattice sum data: species-pair groups of blocks with flat scatter indices
        self.S_groups = [(torch.tensor(g["blocks"], device=dev, dtype=cd),
                          torch.tensor(g["N"], device=dev, dtype=rd),
                          torch.tensor(g["flat"], device=dev, dtype=torch.long))
                         for g in self.ov.groups.values()]
        # augmentation blocks
        self.A_blocks = [(torch.tensor(b, device=dev, dtype=cd), N, ia_mu, ia_a)
                         for b, N, ia_mu, ia_a in zip(self.aug.blocks, self.aug.Nimg, self.aug.mu_atom, self.aug.a_atom)]

    # ------------------------------------------------------------------
    def _interp(self, table, q):
        """linear interpolation of rows of table (n, nq) at q (npl,) -> (n, npl)"""
        dq = self.qtab[1] - self.qtab[0]
        x = q / dq
        i0 = torch.clamp(x.floor().long(), 0, table.shape[1] - 2)
        t = (x - i0.to(x.dtype)).unsqueeze(0)
        return table[:, i0] * (1 - t) + table[:, i0 + 1] * t

    def _atom_centered_pw(self, qcart, which):
        """PW coefficients of Bloch sums of atom-centred functions.
        which='basis' or 'proj'. Returns (nfunc, npl) complex."""
        rd, cd = self.rdtype, self.cdtype
        qn = torch.linalg.norm(qcart, dim=1)
        Y = [real_ylm(l, qcart) for l in range(self.lmax + 1)]                  # each (npl, 2l+1)
        if which == "basis":
            rad = self._interp(self.Rt_tab, qn)[self.bas_rad]                     # (nb, npl)
            l, m, atom = self.bas_l, self.bas_m, self.bas_atom
            pref = 4 * math.pi / math.sqrt(self.omega)
        else:
            rad = self._interp(self.pq_tab, qn)[self.proj_type]                   # (nproj, npl)
            l, m, atom = self.proj_l, self.proj_m, self.proj_atom
            pref = 1.0 / math.sqrt(self.omega)
        ylm = torch.empty_like(rad)
        for ll in range(self.lmax + 1):
            sel = l == ll
            if sel.any():
                ylm[sel] = Y[ll][:, m[sel]].T
        phase = torch.exp(-1j * (self.tau[atom] @ qcart.T).to(cd))              # (nfunc, npl)
        il = torch.tensor([(-1j) ** int(x) for x in l.tolist()], device=self.device, dtype=cd).unsqueeze(1)
        return pref * il * ylm.to(cd) * rad.to(cd) * phase

    def Sk(self, kfrac):
        nb = self.nbasis
        S = torch.zeros(nb * nb, device=self.device, dtype=self.cdtype)
        for blocks, N, flat in self.S_groups:
            ph = torch.exp(2j * math.pi * (N @ kfrac).to(self.cdtype))
            S.index_add_(0, flat, (ph[:, None, None] * blocks).reshape(-1))
        return S.reshape(nb, nb)

    def Ak(self, kfrac):
        nb, npj = self.nbasis, self.paw.nproj
        A = torch.zeros((nb, npj), device=self.device, dtype=self.cdtype)
        for blk, N, ia_mu, ia_a in self.A_blocks:
            ph = cmath.exp(2j * math.pi * float(np.dot(N, kfrac.cpu().numpy())))
            A[self.basis.atom_slices[ia_mu], self.paw.atom_proj_slices[ia_a]] += ph * blk
        return A

    def _pw_chunk(self, npl):
        """Plane waves per chunk: about a quarter of free device memory for the
        ~6 temporaries of size (nfunc x chunk) built by _atom_centered_pw."""
        if self.device.type != "cuda":
            return npl
        free, _ = torch.cuda.mem_get_info(self.device)
        itemsize = 16 if self.cdtype == torch.complex128 else 8
        per_pw = 6 * itemsize * max(self.nbasis, self.paw.nproj)
        return int(max(4096, min(npl, 0.25 * free // per_pw)))

    # ------------------------------------------------------------------
    def project_k(self, ispin, ik):
        wav = self.wav
        dev, rd, cd = self.device, self.rdtype, self.cdtype
        G = torch.tensor(wav.gvectors(ik), device=dev, dtype=rd)
        kfrac = torch.tensor(wav.kpoints[ik], device=dev, dtype=rd)
        qcart = (G + kfrac) @ self.rec
        C = torch.tensor(wav.coeffs(ispin, ik)[: self.Nb], device=dev).to(cd)      # (Nb, npl)
        # <chi|psi~> and <p|psi~>, accumulated over plane-wave chunks so that the
        # (nfunc x npl) coefficient arrays of large vacuum cells fit on the device
        npl = qcart.shape[0]
        chunk = self._pw_chunk(npl)
        TB = torch.zeros((self.nbasis, self.Nb), device=dev, dtype=cd)
        P = torch.zeros((self.paw.nproj, self.Nb), device=dev, dtype=cd)          # (nproj, Nb)
        for a in range(0, npl, chunk):
            q, Cc = qcart[a:a + chunk], C[:, a:a + chunk]
            TB += self._atom_centered_pw(q, "basis").conj() @ Cc.T
            P += self._atom_centered_pw(q, "proj").conj() @ Cc.T
        T = TB + self.Ak(kfrac) @ P                                                # (nb, Nb)
        S = self.Sk(kfrac)
        S = 0.5 * (S + S.mH)
        Cn = torch.linalg.solve(S, T)                                              # C = S^-1 T
        spill = 1.0 - torch.real(torch.sum(T.conj() * Cn, dim=0))                  # (Nb,)
        Sh = _hermitian_power(S, 0.5)
        C1, min_sv = _loewdin_orthonormalise(Cn, S, Sh, self.rank_mode)
        Oorth = C1.mH @ S @ C1                                                     # LOBSTER's bandOverlaps check
        band_dev = torch.abs(Oorth - torch.eye(self.Nb, device=dev, dtype=cd)).max().item()
        E = torch.tensor(wav.eig[ispin, ik, : self.Nb], device=dev, dtype=rd)
        C2 = Sh @ C1                                                               # Loewdin-basis coefficients
        H2 = (C2 * E.to(cd)) @ C2.mH                                               # Hamiltonian in Loewdin basis
        H2 = 0.5 * (H2 + H2.mH)
        return dict(eig=E, C1=C1, C2=C2, S=S, H2=H2, spill=spill, band_dev=band_dev, Oorth=Oorth, Cn=Cn,
                    min_sv=min_sv.item())

    def project_batch(self, ispin, iks):
        """Batched version of project_k for a list of k-points (zero-padded plane-wave sets)."""
        wav = self.wav
        dev, rd, cd = self.device, self.rdtype, self.cdtype
        B = len(iks)
        Gs = [wav.gvectors(ik) for ik in iks]
        npl = max(len(g) for g in Gs)
        G = torch.zeros((B, npl, 3), device=dev, dtype=rd)
        C = torch.zeros((B, self.Nb, npl), device=dev, dtype=cd)
        for b, ik in enumerate(iks):
            n = len(Gs[b])
            G[b, :n] = torch.tensor(Gs[b], device=dev, dtype=rd)
            C[b, :, :n] = torch.tensor(wav.coeffs(ispin, ik)[: self.Nb], device=dev).to(cd)
        kfrac = torch.tensor(wav.kpoints[list(iks)], device=dev, dtype=rd)               # (B,3)
        qcart = (G + kfrac[:, None, :]) @ self.rec                                        # (B, npl, 3)
        cb = self._atom_centered_pw(qcart.reshape(-1, 3), "basis").reshape(self.nbasis, B, npl).permute(1, 0, 2)
        cp = self._atom_centered_pw(qcart.reshape(-1, 3), "proj").reshape(self.paw.nproj, B, npl).permute(1, 0, 2)
        P = cp.conj() @ C.mT                                                              # (B, nproj, Nb)
        A = torch.stack([self.Ak(kfrac[b]) for b in range(B)])
        T = cb.conj() @ C.mT + A @ P                                                      # (B, nb, Nb)
        S = torch.stack([self.Sk(kfrac[b]) for b in range(B)])
        S = 0.5 * (S + S.mH)
        Cn = torch.linalg.solve(S, T)
        spill = 1.0 - torch.real(torch.sum(T.conj() * Cn, dim=1))                         # (B, Nb)
        Sh = _hermitian_power(S, 0.5)
        C1, min_sv = _loewdin_orthonormalise(Cn, S, Sh, self.rank_mode)
        Oorth = C1.mH @ S @ C1
        eye = torch.eye(self.Nb, device=dev, dtype=cd)
        band_dev = torch.abs(Oorth - eye).flatten(1).max(dim=1).values
        E = torch.tensor(wav.eig[ispin, list(iks), : self.Nb], device=dev, dtype=rd)     # (B, Nb)
        C2 = Sh @ C1
        H2 = (C2 * E[:, None, :].to(cd)) @ C2.mH
        H2 = 0.5 * (H2 + H2.mH)
        out = []
        for b in range(B):
            out.append(dict(eig=E[b], C1=C1[b], C2=C2[b], S=S[b], H2=H2[b], spill=spill[b],
                            band_dev=band_dev[b].item(), Oorth=Oorth[b], Cn=Cn[b], min_sv=min_sv[b].item()))
        return out

    def run(self, kweights, progress=True, batch=None):
        wav = self.wav
        nk, ns = wav.nkpts, wav.nspin
        self.results = [[None] * nk for _ in range(ns)]
        self.kweights = np.asarray(kweights)
        if batch is None:
            if self.device.type == "cuda":
                free, total = torch.cuda.mem_get_info(self.device)
                npl_max = int(wav.nplane.max())
                itemsize = 16 if self.cdtype == torch.complex128 else 8
                # dominant per-k temporaries: C (Nb x npl), basis/proj coefficient sets, a few (nb x npl)
                per_k = itemsize * npl_max * (self.Nb + 2 * self.nbasis + 2 * self.paw.nproj + 8)
                batch = int(max(1, min(64, (0.5 * free) // per_k)))
            else:
                batch = 16
            self.log(f"  k-point batch size {batch}")
        t0 = time.time()
        for s in range(ns):
            for k0 in range(0, nk, batch):
                iks = list(range(k0, min(nk, k0 + batch)))
                rs = self.project_batch(s, iks) if batch > 1 else [self.project_k(s, iks[0])]
                for ik, r in zip(iks, rs):
                    self.results[s][ik] = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in r.items()}
                if progress and (k0 % max(batch, (nk // 10 // batch) * batch) == 0 or iks[-1] == nk - 1):
                    self.log(f"  spin {s + 1} k-point {iks[-1] + 1}/{nk}  ({time.time() - t0:.1f} s)")
        return self.results

    # ------------------------------------------------------------------
    def spilling(self, occ_threshold=1e-8, spins=None, occ=None):
        """(abs charge spilling, abs total spilling) as in LOBSTER, over the given spin channels."""
        wav = self.wav
        num_c = den_c = num_t = den_t = 0.0
        for s in (range(wav.nspin) if spins is None else spins):
            for ik in range(wav.nkpts):
                sp = self.results[s][ik]["spill"]
                f = wav.occ[s, ik, : self.Nb] if occ is None else occ[s][ik]
                w = self.kweights[ik]
                num_c += w * np.sum(f * np.abs(sp)); den_c += w * np.sum(f)
                num_t += w * np.sum(np.abs(sp)); den_t += w * self.Nb
        return num_c / max(den_c, 1e-30), num_t / max(den_t, 1e-30)

    def max_band_dev(self):
        return max(self.results[s][ik]["band_dev"] for s in range(self.wav.nspin) for ik in range(self.wav.nkpts))
