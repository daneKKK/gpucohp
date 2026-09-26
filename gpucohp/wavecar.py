"""Reader for VASP WAVECAR (complex, non gamma-only, collinear).

Follows the well known record layout:
  record 1 : recl, nspin, rtag
  record 2 : nkpts, nbands, encut, lattice (3x3), efermi
  then for each spin, each k-point:
      header record : nplane, k (3), [energy, 0, occupation] * nbands
      nbands records: plane-wave coefficients (complex64 if rtag=45200,
                      complex128 if rtag=45210)
G-vector ordering follows VASP's loop over the FFT grid (z slowest? no:
x fastest? see gvectors()).  The ordering used by VASP is
   for iz in range(ngz): for iy in range(ngy): for ix in range(ngx)
with negative indices wrapped, and only |k+G|^2*HSQDTM <= ENCUT retained.
"""
import numpy as np
from .constants import HSQDTM, TPI


class Wavecar:
    def __init__(self, path="WAVECAR", verbose=False):
        self.path = path
        self._f = open(path, "rb")
        rec = np.fromfile(self._f, dtype=np.float64, count=3)
        self.recl, self.nspin, self.rtag = int(rec[0]), int(rec[1]), int(rec[2])
        if self.rtag == 45200:
            self.cdtype = np.complex64
        elif self.rtag == 45210:
            self.cdtype = np.complex128
        else:
            raise ValueError(f"unsupported WAVECAR rtag {self.rtag} (gamma-only / ncl not supported)")
        self._f.seek(self.recl)
        rec = np.fromfile(self._f, dtype=np.float64, count=12)
        self.nkpts, self.nbands = int(rec[0]), int(rec[1])
        self.encut = rec[2]
        self.lattice = rec[3:12].reshape(3, 3)          # rows = a1,a2,a3 (Angstrom)
        self._f.seek(self.recl + 12 * 8)
        self.efermi = float(np.fromfile(self._f, dtype=np.float64, count=1)[0])
        self.volume = float(np.linalg.det(self.lattice))
        self.rec_lattice = TPI * np.linalg.inv(self.lattice).T   # rows = b1,b2,b3 (1/Angstrom, incl 2pi)
        # FFT grid bounds (same recipe as VASP / pymatgen)
        self._ngmax = self._grid_bounds()
        self._read_headers()

    # ------------------------------------------------------------------
    def _grid_bounds(self):
        # maximum G index along each reciprocal vector so that |G| <= Gcut
        gcut = np.sqrt(self.encut / HSQDTM)   # 1/A (includes 2pi)
        blen = np.linalg.norm(self.rec_lattice, axis=1)
        # VASP: NGX >= 2*Gcut/|b|+1 etc. We use a generous bound: index range
        # -nmax..nmax where nmax = int(gcut / |b_i| * something).  Use the
        # exact geometric bound: max index along b_i is gcut * |a_i| / 2pi
        alen = np.linalg.norm(self.lattice, axis=1)
        nmax = np.ceil(gcut * alen / TPI).astype(int) + 1
        return nmax

    def _read_headers(self):
        nk, nb, ns = self.nkpts, self.nbands, self.nspin
        self.nplane = np.zeros((ns, nk), dtype=int)
        self.kpoints = np.zeros((nk, 3))
        self.eig = np.zeros((ns, nk, nb))
        self.occ = np.zeros((ns, nk, nb))
        for s in range(ns):
            for k in range(nk):
                irec = 2 + s * nk * (nb + 1) + k * (nb + 1)
                self._f.seek(irec * self.recl)
                rec = np.fromfile(self._f, dtype=np.float64, count=4 + 3 * nb)
                self.nplane[s, k] = int(rec[0])
                self.kpoints[k] = rec[1:4]
                tmp = rec[4:4 + 3 * nb].reshape(nb, 3)
                self.eig[s, k] = tmp[:, 0]
                self.occ[s, k] = tmp[:, 2]

    # ------------------------------------------------------------------
    def gvectors(self, ik, check=True):
        """Integer G indices (nplane,3) for k-point ik, in VASP's ordering."""
        kpt = self.kpoints[ik]
        nx, ny, nz = self._ngmax
        # VASP loop order: z outermost, y, x innermost; indices run 0..N-1 with
        # values ix<=N/2 positive, otherwise ix-N.  Reproduce with explicit
        # ordering: for each axis the sequence 0,1,..,nmax,-nmax,..,-1
        def seq(n):
            return np.concatenate([np.arange(0, n + 1), np.arange(-n, 0)])
        gx, gy, gz = seq(nx), seq(ny), seq(nz)
        GZ, GY, GX = np.meshgrid(gz, gy, gx, indexing="ij")
        G = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
        kg = (G + kpt) @ self.rec_lattice
        ekin = HSQDTM * np.einsum("ij,ij->i", kg, kg)
        mask = ekin <= self.encut
        npl = int(self.nplane[0, ik])
        if check and mask.sum() != npl:
            # boundary rounding differs from VASP's arithmetic: take exactly the
            # npl lowest kinetic energies (loop order is preserved)
            thr = np.sort(ekin)[npl - 1]
            mask = ekin <= thr * (1 + 1e-12)
            if mask.sum() != npl:
                raise RuntimeError(
                    f"k-point {ik}: generated {mask.sum()} G-vectors, WAVECAR has {npl}")
        return G[mask]

    def coeffs(self, ispin, ik, bands=None):
        """Plane-wave coefficients (nbands_sel, nplane) as complex128."""
        nb = self.nbands
        npl = self.nplane[ispin, ik]
        if bands is None:
            bands = range(nb)
        out = np.empty((len(bands), npl), dtype=np.complex128)
        base = 2 + ispin * self.nkpts * (nb + 1) + ik * (nb + 1) + 1
        for i, b in enumerate(bands):
            self._f.seek((base + b) * self.recl)
            out[i] = np.fromfile(self._f, dtype=self.cdtype, count=npl)
        return out

    def summary(self):
        return (f"WAVECAR: nspin={self.nspin} nk={self.nkpts} nbands={self.nbands} "
                f"encut={self.encut:.1f} eV efermi={self.efermi:.4f} eV "
                f"prec={'single' if self.rtag == 45200 else 'double'} vol={self.volume:.3f} A^3")
