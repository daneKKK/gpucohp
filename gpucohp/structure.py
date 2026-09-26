"""Crystal structure (POSCAR/CONTCAR reader) and lattice helpers."""
import numpy as np

_ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
             "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
             "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()
ATOMIC_NUMBER = {el: i + 1 for i, el in enumerate(_ELEMENTS)}


class Structure:
    def __init__(self, lattice, symbols, counts, cart):
        self.lattice = np.asarray(lattice, dtype=float)          # rows a1,a2,a3 (A)
        self.symbols = list(symbols)                              # per species
        self.counts = [int(c) for c in counts]
        self.cart = np.asarray(cart, dtype=float)                 # (nat,3) A
        self.species_of_atom = np.concatenate([[i] * c for i, c in enumerate(self.counts)]).astype(int)
        self.symbol_of_atom = [self.symbols[i] for i in self.species_of_atom]
        self.natoms = len(self.cart)
        self.volume = float(abs(np.linalg.det(self.lattice)))
        self.rec_lattice = 2 * np.pi * np.linalg.inv(self.lattice).T
        self.frac = self.cart @ np.linalg.inv(self.lattice)

    @classmethod
    def from_poscar(cls, path):
        lines = [ln.rstrip("\n") for ln in open(path)]
        scale = float(lines[1].split()[0])
        lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
        if scale < 0:
            scale = (-scale / abs(np.linalg.det(lat))) ** (1.0 / 3.0)
        lat *= scale
        i = 5
        toks = lines[i].split()
        if toks and not toks[0].isdigit():
            symbols = toks
            i += 1
        else:
            symbols = None
        counts = [int(x) for x in lines[i].split()]
        i += 1
        if lines[i].strip().lower().startswith("s"):
            i += 1
        mode = lines[i].strip().lower()
        i += 1
        nat = sum(counts)
        pos = np.array([[float(x) for x in lines[i + j].split()[:3]] for j in range(nat)])
        if mode.startswith("c") or mode.startswith("k"):
            cart = pos * scale
        else:
            cart = pos @ lat
        if symbols is None:
            symbols = [f"X{j}" for j in range(len(counts))]
        return cls(lat, symbols, counts, cart)

    def images_within(self, rmax):
        """All lattice translation vectors T (Cartesian) with |T| <= rmax
        plus a margin so that any |tau_B + T - tau_A| <= rmax is covered."""
        # bound on integer indices: |n_i| <= rmax * |b_i| / (2 pi) + 1
        blen = np.linalg.norm(self.rec_lattice, axis=1)
        nmax = np.ceil(rmax * blen / (2 * np.pi)).astype(int) + 1
        rng = [np.arange(-n, n + 1) for n in nmax]
        N = np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, 3)
        T = N @ self.lattice
        return N, T

    def distance_table(self, rmax):
        """List of (iA, iB, N(3 ints), d(3) cart, |d|) for all atom pairs and lattice
        images with |d| <= rmax, d = tau_B + N.A - tau_A (includes the on-site d=0 entries)."""
        N, T = self.images_within(rmax + 1e-9)
        out = []
        nat = self.natoms
        for a in range(nat):
            # (nat, nT, 3)
            d = self.cart[None, :, :] + T[:, None, :] - self.cart[a][None, None, :]
            d = d.transpose(1, 0, 2)
            dist = np.linalg.norm(d, axis=2)
            bb, tt = np.where(dist <= rmax)
            for b, t in zip(bb, tt):
                out.append((a, int(b), tuple(int(x) for x in N[t]), d[b, t], float(dist[b, t])))
        return out
