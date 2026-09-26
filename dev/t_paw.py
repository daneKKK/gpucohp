import sys, numpy as np
sys.path.insert(0, r"D:\work\lobster\gpucohp")
from gpucohp.wavecar import Wavecar
from gpucohp.potcar import read_potcar
from gpucohp.structure import Structure
from gpucohp.paw import PawSetup

REF = r"D:\work\lobster\ref_TiB2_sber"
w = Wavecar(REF + r"\WAVECAR")
st = Structure.from_poscar(REF + r"\POSCAR")
sp = read_potcar(REF + r"\POTCAR")
paw = PawSetup(st, sp)
print("nproj total", paw.nproj)
np.set_printoptions(precision=5, suppress=True, linewidth=180)
for ik in (0, 1, 777):
    G = w.gvectors(ik)
    kc = w.kpoints[ik] @ w.rec_lattice
    q = (G + w.kpoints[ik]) @ w.rec_lattice
    C = w.coeffs(0, ik)[:18]
    P = paw.projections(q, kc, C)
    Opw = C.conj() @ C.T
    Oaug = P.conj().T @ paw.Qfull @ P
    O = Opw + Oaug
    print(f"k{ik}: diag pw   {np.real(np.diag(Opw))[:8]}")
    print(f"k{ik}: diag aug  {np.real(np.diag(Oaug))[:8]}")
    print(f"k{ik}: diag total{np.real(np.diag(O))[:8]}")
    off = O - np.eye(len(O))
    print(f"k{ik}: max|O-1| = {np.abs(off).max():.2e}")
    # scale scan for the augmentation part
    best = None
    for s in (0.5, 1 / np.sqrt(2), 1.0, np.sqrt(2), 2.0, 4 * np.pi, 1 / (4 * np.pi), np.sqrt(4 * np.pi), 1 / np.sqrt(4 * np.pi)):
        dev = np.abs(Opw + s * s * Oaug - np.eye(len(O))).max()
        if best is None or dev < best[1]:
            best = (s, dev)
    print(f"k{ik}: best scale for projectors {best[0]:.4f} -> max dev {best[1]:.2e}")
