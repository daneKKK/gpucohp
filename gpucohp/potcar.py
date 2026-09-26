"""Parser for VASP PAW POTCAR files (multiple species concatenated).

For each species we extract what the projection needs:
  * projector functions p~_i(q) in reciprocal space (tabulated on a uniform
    q grid 0..psmaxn, 100 points) and in real space (0..rmax, 100 points)
  * the PAW radial grid and the all-electron / pseudo partial waves
    u_i(r) = r*phi_i(r) for every channel
  * the augmentation charges Q_ij read from the file (checked against the
    partial waves)
Units (verified numerically against WAVECAR orthonormality and projector /
partial-wave duality): the "PAW radial sets" grid and u(r)=r*phi(r) are in
Angstrom; projector rmax in Angstrom; psmaxn in 1/Angstrom (incl. 2pi).
The reciprocal projector table is on q_i = i*psmaxn/100, i=0..99.
"""
import re
import numpy as np
_trapz = getattr(np, "trapezoid", None) or np.trapz
from .constants import BOHR

_float_re = re.compile(r"[-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?|[-+]?\d+(?:[EeDd][-+]?\d+)?")


def _floats(block):
    return np.array([float(x.replace("D", "E").replace("d", "e")) for x in _float_re.findall(block)])


class PawSpecies:
    def __init__(self, text):
        self.text = text
        head = text.split("END of PSCTR", 1)[0]
        self.titel = re.search(r"TITEL\s*=\s*(.*)", head).group(1).strip()
        self.symbol = re.sub(r"_.*", "", self.titel.split()[1])
        self.zval = float(re.search(r"ZVAL\s*=\s*([\d.]+)", head).group(1))
        self.enmax = float(re.search(r"ENMAX\s*=\s*([\d.]+)", head).group(1))
        self.rcore = float(re.search(r"RCORE\s*=\s*([\d.]+)", head).group(1)) * BOHR  # A
        self.rdep = float(re.search(r"RDEP\s*=\s*([\d.]+)", head).group(1)) * BOHR    # A
        m = re.search(r"RMAX\s*=\s*([\d.]+)", head)
        self.rmax_proj = float(m.group(1)) * BOHR if m else None
        self._parse_atomic_config(head)
        self._parse_nonlocal()
        self._parse_radial()

    # --------------------------------------------------------------
    def _parse_atomic_config(self, head):
        self.atomic_config = []
        m = re.search(r"Atomic configuration\s*\n\s*(\d+)\s+entries\s*\n.*\n((?:.*\n)+?)\s*Description", head)
        if m:
            for line in m.group(2).strip().splitlines():
                p = line.split()
                if len(p) >= 5:
                    self.atomic_config.append((int(p[0]), int(p[1]), float(p[3]), float(p[4])))

    def _parse_nonlocal(self):
        txt = self.text
        pre, _, rest = txt.partition(" Non local Part")
        lastline = pre.rstrip().splitlines()[-1]
        self.psmaxn = float(lastline.split()[0])           # 1/A
        blocks = ("Non local Part" + rest).split(" Non local Part")
        blocks = [b for b in blocks if b.strip()]
        self.channels = []
        for b in blocks:
            body = b.split("Non local Part", 1)[-1]
            body = body.split("PAW radial sets")[0]
            lines = body.strip("\n").splitlines()
            hdr = lines[0].split()
            l, nproj, rmax = int(hdr[0]), int(hdr[1]), float(hdr[2])
            after = "\n".join(lines[1:])
            parts = re.split(r" (Reciprocal Space Part|Real Space Part)\s*\n", after)
            dion = _floats(parts[0])[: nproj * nproj].reshape(nproj, nproj)
            pq, pr = [], []
            i = 1
            while i < len(parts) - 1:
                tag, data = parts[i], parts[i + 1]
                vals = _floats(data)[:100]
                (pq if tag.startswith("Reciprocal") else pr).append(vals)
                i += 2
            self.channels.append(dict(l=l, nproj=nproj, rmax=rmax, dion=dion,
                                      pq=np.array(pq), pr=np.array(pr)))
        self.nproj_total = sum(c["nproj"] for c in self.channels)
        self.proj_table = []
        for ic, c in enumerate(self.channels):
            for i in range(c["nproj"]):
                self.proj_table.append((ic, c["l"], i))
        # VASP: q_i = i * PSMAXN / NPSNL, i = 0..NPSNL-1  (spacing psmaxn/100, NOT /99)
        self.qgrid = np.arange(100) * self.psmaxn / 100.0

    def _parse_radial(self):
        txt = self.text
        _, _, rad = txt.partition(" PAW radial sets")
        lines = rad.splitlines()
        hdr = lines[1].split()
        self.nmax = int(hdr[0])
        n = self.nmax
        nt = self.nproj_total

        def block(name, count, start=0):
            idx = rad.find(name, start)
            if idx < 0:
                return None, start
            sub = rad[idx + len(name):]
            sub = sub.split("\n", 1)[1]
            vals = _floats(sub)[:count]
            return vals, idx + len(name)

        pos = 0
        self.qij, pos = block("augmentation charges", nt * nt, pos)
        self.qij = self.qij.reshape(nt, nt)
        occ, pos = block("uccopancies in atom", nt * nt, pos)
        self.occ_atom = occ.reshape(nt, nt) if occ is not None else None
        g, pos = block(" grid", n, pos)
        self.rgrid = g                               # already in Angstrom
        self.aepot, pos = block("aepotential", n, pos)
        self.core_rho, pos = block("core charge-density", n, pos)
        self.ps_wf = np.zeros((nt, n))
        self.ae_wf = np.zeros((nt, n))
        p = rad.find("pspotential")
        for i in range(nt):
            v, p = block("pseudo wavefunction", n, p)
            self.ps_wf[i] = v
            v, p = block("ae wavefunction", n, p)
            self.ae_wf[i] = v
        # stored as u(r) = r*phi(r) on the Angstrom grid (Angstrom^-1/2); no conversion

    # --------------------------------------------------------------
    def lmax(self):
        return max(c["l"] for c in self.channels)

    def qij_from_waves(self):
        """Recompute augmentation charges from partial waves (check)."""
        r = self.rgrid
        nt = self.nproj_total
        q = np.zeros((nt, nt))
        for i in range(nt):
            for j in range(nt):
                if self.proj_table[i][1] != self.proj_table[j][1]:
                    continue
                f = self.ae_wf[i] * self.ae_wf[j] - self.ps_wf[i] * self.ps_wf[j]
                q[i, j] = _trapz(f, r)
        return q

    def __repr__(self):
        ch = ",".join(f"l{c['l']}x{c['nproj']}" for c in self.channels)
        return (f"<PawSpecies {self.titel}: zval={self.zval} nproj={self.nproj_total} "
                f"[{ch}] nmax={self.nmax} psmaxn={self.psmaxn:.3f}>")


def read_potcar(path="POTCAR"):
    txt = open(path).read()
    parts = txt.split("End of Dataset")
    return [PawSpecies(p) for p in parts if "TITEL" in p]
