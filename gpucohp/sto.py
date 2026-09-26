"""Slater-type-orbital basis sets (LOBSTER style contractions).

A contracted radial function is
    R(r) = sum_a c_a N_a r^(n_a-1) exp(-alpha_a r)
with normalised primitives N_a = sqrt((2 alpha)^(2n+1) / (2n)!).
LOBSTER lists alpha in 1/bohr; internally everything is in Angstrom.

Two input formats are supported:
  * LOBSTER "basisFunctions.lobster" dumps (writeBasisFunctions keyword)
  * LOBSTER custom ".sto" files (see Custom_Basis/README.md of the LOBSTER
    distribution)
"""
import math
import re
import numpy as np
from scipy.special import hyp2f1, gammaln
from .constants import BOHR
from .sph import L_OF_LETTER, ORB_NAMES


class RadialSTO:
    """One contracted radial function (element, label like '3d', l)."""

    def __init__(self, element, label, prims):
        self.element = element
        self.label = label                      # e.g. "3d"
        self.l = L_OF_LETTER[label[-1]]
        self.n_principal = int(label[:-1])
        # prims: list of (n, alpha_bohr, coeff)
        self.n = np.array([p[0] for p in prims], dtype=int)
        self.alpha = np.array([p[1] for p in prims]) / BOHR      # 1/Angstrom
        self.coeff = np.array([p[2] for p in prims])
        self.norm = np.sqrt((2 * self.alpha) ** (2 * self.n + 1) /
                            np.array([math.factorial(2 * k) for k in self.n]))

    # radial function in real space (Angstrom)
    def R(self, r):
        r = np.asarray(r, dtype=float)
        out = np.zeros_like(r)
        for n, a, c, N in zip(self.n, self.alpha, self.coeff, self.norm):
            out += c * N * r ** (n - 1) * np.exp(-a * r)
        return out

    def selfoverlap(self):
        s = 0.0
        for n1, a1, c1, N1 in zip(self.n, self.alpha, self.coeff, self.norm):
            for n2, a2, c2, N2 in zip(self.n, self.alpha, self.coeff, self.norm):
                s += c1 * c2 * N1 * N2 * math.factorial(n1 + n2) / (a1 + a2) ** (n1 + n2 + 1)
        return s

    # radial Fourier-Bessel transform  Rt_l(q) = int r^2 R(r) j_l(qr) dr
    def Rt(self, q):
        q = np.asarray(q, dtype=float)
        l = self.l
        out = np.zeros_like(q)
        for n, a, c, N in zip(self.n, self.alpha, self.coeff, self.norm):
            out += c * N * _prim_ft(n, l, a, q)
        return out

    def rcut(self, tol=1e-7):
        """Radius beyond which |R(r)| r < tol (for lattice sums)."""
        r = np.linspace(0.5, 60.0, 6000)
        v = np.abs(self.R(r)) * r
        idx = np.where(v > tol)[0]
        return float(r[idx[-1]]) if len(idx) else 0.5


def _prim_ft(n, l, a, q):
    """int_0^inf r^(n+1) exp(-a r) j_l(q r) dr  (analytic, via 2F1).

    With mu = n+3/2, nu = l+1/2:
    sqrt(pi/(2q)) (q/2)^nu Gamma(mu+nu) / (a^(mu+nu) Gamma(nu+1)) 2F1((mu+nu)/2,(mu+nu+1)/2; nu+1; -q^2/a^2)
    """
    q = np.asarray(q, dtype=float)
    s = n + l + 2                     # mu + nu (integer)
    nu = l + 0.5
    pref = math.exp(gammaln(s) - gammaln(nu + 1)) / a ** s
    out = np.empty_like(q)
    small = q < 1e-12
    z = -(q[~small] / a) ** 2
    f = hyp2f1(s / 2.0, (s + 1) / 2.0, nu + 1, z)
    # sqrt(pi/(2q)) * (q/2)^(l+1/2) = sqrt(pi)/2^(l+1) * q^l
    out[~small] = math.sqrt(math.pi) / 2 ** (l + 1) * q[~small] ** l * pref * f
    if small.any():
        # q -> 0: j_l(qr) -> (qr)^l/(2l+1)!!  ; only l=0 finite
        if l == 0:
            out[small] = math.factorial(n + 1) / a ** (n + 2)
        else:
            out[small] = 0.0
    return out


class BasisFunction:
    """One (atom, radial, m) basis function."""
    __slots__ = ("iatom", "element", "radial", "m", "name")

    def __init__(self, iatom, element, radial, m):
        self.iatom = iatom
        self.element = element
        self.radial = radial
        self.m = m
        self.name = f"{radial.n_principal}{ORB_NAMES[radial.l][m]}" if radial.l > 0 else f"{radial.n_principal}s"


# ----------------------------------------------------------------------
# parsers
# ----------------------------------------------------------------------
def read_lobster_basisfunctions(path):
    """Parse basisFunctions.lobster -> dict element -> {label: RadialSTO}."""
    txt = open(path).read()
    blocks = re.split(r"\n(?=\S+\s+\d[spdf]\S*\s+at atom)", txt)
    out = {}
    for b in blocks:
        m = re.match(r"\s*(\S+)\s+(\d)([spdf])(\S*)\s+at atom\s+(\d+)", b)
        if not m:
            continue
        el, npr, lch = m.group(1), m.group(2), m.group(3)
        label = npr + lch
        if el in out and label in out[el]:
            continue
        prims = []
        for line in b.splitlines()[2:]:
            p = line.split()
            if len(p) == 3 and p[0].isdigit():
                prims.append((int(p[0]), float(p[1]), float(p[2])))
        out.setdefault(el, {})[label] = RadialSTO(el, label, prims)
    return out


def read_sto_file(path, element=None):
    """Parse a LOBSTER custom .sto file -> {label: RadialSTO}."""
    lines = [ln.rstrip("\\").strip() for ln in open(path) if ln.strip()]
    head = lines[0].split(",")[0].strip()
    el = element or head.capitalize()
    first = [int(x) for x in lines[1].split(":")[1].split()]
    norb = [int(x) for x in lines[2].split(":")[1].split()]
    nprim = [int(x) for x in lines[3].split(":")[1].split()]
    letters = "spdf"
    prims = {0: [], 1: [], 2: [], 3: []}
    for ln in lines[4:]:
        p = ln.split()
        l = L_OF_LETTER[p[0].lower()]
        prims[l].append((int(p[1]), float(p[2]), [float(x) for x in p[3:]]))
    out = {}
    for l in range(4):
        for io in range(norb[l]):
            label = f"{first[l] + io}{letters[l]}"
            pr = [(n, a, cs[io]) for (n, a, cs) in prims[l]]
            out[label] = RadialSTO(el, label, pr)
    return out
