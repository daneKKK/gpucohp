import sys, time, math, numpy as np, torch
sys.path.insert(0, r"D:\work\lobster\gpucohp")
from gpucohp.wavecar import Wavecar
from gpucohp.potcar import read_potcar
from gpucohp.structure import Structure
from gpucohp.sto import read_lobster_basisfunctions
from gpucohp.basis import LocalBasis
from gpucohp.paw import PawSetup
from gpucohp.augment import Augmentation
from gpucohp.twocenter import RadialFTTable, TwoCenterOverlap
from gpucohp.engine import ProjectionEngine, _hermitian_power
from gpucohp.lobsterin import LobsterIn
from gpucohp.analysis import build_pairs
from gpucohp.kweights import read_kweights
from scipy.special import erf
REF = r"D:\work\lobster\ref_TiB2_sber"
w = Wavecar(REF + r"\WAVECAR"); st = Structure.from_poscar(REF + r"\CONTCAR"); sp = read_potcar(REF + r"\POTCAR")
paw = PawSetup(st, sp); rad = read_lobster_basisfunctions(REF + r"\basisFunctions.lobster")
basis = LocalBasis(st, rad, {"Ti": ["3s", "3p", "3d", "4s"], "B": ["2s", "2p"]})
aug = Augmentation(basis, st, paw)
table = RadialFTTable(basis.radials); tc = TwoCenterOverlap(basis, st, table, rmax_pair=14.0)
eng = ProjectionEngine(w, st, basis, paw, aug, tc, device="cpu", log=lambda *a: None)
kw = read_kweights(REF + r"\vasprun.xml", nk=w.nkpts)
t0 = time.time(); eng.run(kw, progress=False); print("projection", time.time() - t0)
li = LobsterIn.read(REF + r"\lobsterin"); pairs = build_pairs(st, li)
nk, Nb = w.nkpts, eng.Nb
sig = 0.2 / math.sqrt(2)
eps = w.eig[0, :, :Nb] - w.efermi
focc = 0.5 * (1 + erf(-eps / (sig * math.sqrt(2))))            # (nk, Nb) smeared occupation
C1 = np.stack([r["C1"] for r in eng.results[0]]); C2 = np.stack([r["C2"] for r in eng.results[0]])
S = np.stack([r["S"] for r in eng.results[0]]); H2 = np.stack([r["H2"] for r in eng.results[0]])
E = w.eig[0, :, :Nb]
Cn = np.stack([r["Cn"] for r in eng.results[0]])
Sh = np.stack([_hermitian_power(torch.tensor(x), 0.5).numpy() for x in S])
# Gram-Schmidt (Cholesky) re-orthonormalisation in band order instead of symmetric Loewdin
X = np.einsum("kab,kbn->kan", Sh, Cn)                                     # S^1/2 Cn : Gram-Schmidt = QR in band order
C2gs = np.stack([np.linalg.qr(X[k])[0] * np.sign(np.diag(np.linalg.qr(X[k])[1]).real) for k in range(nk)])
C1gs = np.einsum("kab,kbn->kan", np.linalg.inv(Sh), C2gs)
H2gs = np.einsum("kan,kn,kbn->kab", C2gs, E, np.conj(C2gs))
print("check GS orthonormal:", np.abs(np.einsum("kan,kab,kbm->knm", np.conj(C1gs), S, C1gs) - np.eye(Nb)).max())
Hn = np.einsum("kab,kbn,kn,kcn,kcd->kad", S, C1, E, np.conj(C1), S)      # non-orthogonal basis H = S C1 E C1^H S
kf = w.kpoints
def icohp(C, Hk, p, mode):
    slA, slB = basis.atom_slices[p.iA], basis.atom_slices[p.iB]
    ph = np.exp(2j * np.pi * (kf @ np.array(p.cell)))                       # e^{ik.N}
    if mode == "realspace":
        HT = np.einsum("k,k,kab->ab", kw, np.conj(ph), Hk[:, slA, slB])       # H_{A0,BN}
        v = np.einsum("kan,kbn,k,ab->kn", np.conj(C[:, slA]), C[:, slB], ph, HT)
    else:  # k-local
        v = np.einsum("kan,kbn,k,kab->kn", np.conj(C[:, slA]), C[:, slB], ph, Hk[:, slA, slB])
    return 2 * np.sum(kw[:, None] * focc * v.real)
for p in pairs:
    if p.index in (1, 2, 9, 34):
        vals = {"LSO sym": icohp(C2, H2, p, "realspace"), "GS both": icohp(C2gs, H2gs, p, "realspace"),
                "GS H only": icohp(C2, H2gs, p, "realspace"), "GS dens only": icohp(C2gs, H2, p, "realspace")}
        print(f"pair {p.index:2d} {st.symbol_of_atom[p.iA]}{p.iA+1}-{st.symbol_of_atom[p.iB]}{p.iB+1} {p.cell} d={p.dist:.3f}: " + "  ".join(f"{k}={v:9.5f}" for k, v in vals.items()))
print("ref: pair1 -0.27595, pair2 -0.48166, pair9 -1.58626, pair34 -6.66552")
