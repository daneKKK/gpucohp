"""PAW one-centre corrections for the local basis:

  A_{mu,ai}(k) = <chi_mu^k | phi_i^a - phi~_i^a>_Bloch = sum_T e^{ik.T} I_{mu,ai}(T)
  I_{mu,ai}(T) = int_{sphere a} chi_mu(r + D) [phi_i - phi~_i](r) d3r ,
                 D = tau_a + T - tau_mu

The STO chi_mu(r + D) is expanded in real spherical harmonics around the
sphere centre by angular quadrature on the POTCAR radial grid; the radial
integral with the partial-wave difference u_i(r) = r*phi_i(r) follows.
"""
import numpy as np
_trapz = getattr(np, "trapezoid", None) or np.trapz
from .sph import real_ylm, angular_grid


class Augmentation:
    def __init__(self, basis, structure, paw, rmax_pair=0.0, mode="onsite", ntheta=20, nphi=40, tol_wave=1e-8):
        """mode="onsite": augment each basis function only inside the PAW sphere of
        its own atom (this is what LOBSTER does; verified against its transfer
        matrices).  mode="full": include all spheres within rmax_pair."""
        self.basis = basis
        self.structure = structure
        self.paw = paw
        self.mode = mode
        st = structure
        pts, w = angular_grid(ntheta, nphi)                  # (nang,3), (nang,)
        self.entries = []                                    # (mu, a, N, D)
        nb = basis.nbasis
        # sphere data per species: radial grid within PAW radius and Delta u_i
        self.sphere = []
        for sp in paw.species:
            du = sp.ae_wf - sp.ps_wf                          # (nproj_sp, n)
            nz = np.where(np.abs(du).max(axis=0) > tol_wave)[0]
            nr = nz[-1] + 2 if len(nz) else 1
            r = sp.rgrid[:nr]
            self.sphere.append(dict(r=r, du=du[:, :nr], rc=r[-1]))
        # enumerate (mu, a, T) with |D| <= rmax_pair
        if mode == "onsite":
            table = [(ia, ia, (0, 0, 0), np.zeros(3), 0.0) for ia in range(st.natoms)]
        else:
            table = st.distance_table(rmax_pair)           # (iA=mu atom, iB=a, N, d, |d|) with d = tau_B + T - tau_A
        # A(T) blocks: for each entry index we store array (nb_on_atom_mu, nproj_on_atom_a)
        self.blocks = []
        self.Nimg = []
        self.mu_atom = []
        self.a_atom = []
        Ymax = max(int(max(basis.l)), max(p[3] for p in paw.plist))
        for (ia_mu, ia_a, N, D, dist) in table:
            isp = st.species_of_atom[ia_a]
            sph = self.sphere[isp]
            r = sph["r"]
            # points r_j * rhat_k + D  -> STO argument
            X = r[:, None, None] * pts[None, :, :] + D[None, None, :]      # (nr, nang, 3)
            blk = np.zeros((basis.atom_slices[ia_mu].stop - basis.atom_slices[ia_mu].start,
                            paw.atom_proj_slices[ia_a].stop - paw.atom_proj_slices[ia_a].start))
            mu_funcs = range(basis.atom_slices[ia_mu].start, basis.atom_slices[ia_mu].stop)
            if dist < 1e-12:
                # same centre: chi_mu = R_mu(r) Y_lm  ->  only matching (l,m) projectors
                for jmu, mu in enumerate(mu_funcs):
                    f = basis.functions[mu]
                    Rr = f.radial.R(r)
                    for jp, ip in enumerate(range(paw.atom_proj_slices[ia_a].start, paw.atom_proj_slices[ia_a].stop)):
                        _, _, ipr, l, m = paw.plist[ip]
                        if l == f.radial.l and m == f.m:
                            blk[jmu, jp] = _trapz(r * Rr * sph["du"][ipr], r)
                self.blocks.append(blk); self.Nimg.append(N); self.mu_atom.append(ia_mu); self.a_atom.append(ia_a)
                continue
            # evaluate each unique radial on atom mu once
            Xn = np.linalg.norm(X, axis=-1)                                 # (nr, nang)
            Yx = {l: real_ylm(l, X) for l in range(int(max(basis.l)) + 1)}   # (nr,nang,2l+1)
            Yr = {l: real_ylm(l, pts) for l in range(Ymax + 1)}            # (nang, 2l+1)
            Rcache = {}
            for jmu, mu in enumerate(mu_funcs):
                f = basis.functions[mu]
                key = (f.element, f.radial.label)
                if key not in Rcache:
                    Rcache[key] = f.radial.R(Xn)                           # (nr, nang)
                chi = Rcache[key] * Yx[f.radial.l][:, :, f.m]              # (nr, nang)
                # angular projections F_LM(r) for all projector (l,m) on atom a
                for jp, ip in enumerate(range(paw.atom_proj_slices[ia_a].start, paw.atom_proj_slices[ia_a].stop)):
                    _, _, ipr, l, m = paw.plist[ip]
                    F = (chi * Yr[l][None, :, m] * w[None, :]).sum(axis=1)  # (nr,)
                    blk[jmu, jp] = _trapz(r * F * sph["du"][ipr], r)
            self.blocks.append(blk)
            self.Nimg.append(N)
            self.mu_atom.append(ia_mu)
            self.a_atom.append(ia_a)
        self.Nimg = np.array(self.Nimg)

    def Ak(self, kfrac):
        """A(k) (nbasis, nproj) complex."""
        nb, npj = self.basis.nbasis, self.paw.nproj
        A = np.zeros((nb, npj), dtype=complex)
        phase = np.exp(2j * np.pi * (self.Nimg @ np.asarray(kfrac)))
        for blk, ph, ia_mu, ia_a in zip(self.blocks, phase, self.mu_atom, self.a_atom):
            sl_mu = self.basis.atom_slices[ia_mu]
            sl_a = self.paw.atom_proj_slices[ia_a]
            A[sl_mu, sl_a] += ph * blk
        return A
