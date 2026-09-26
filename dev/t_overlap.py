import sys, time, numpy as np
sys.path.insert(0, r"D:\work\lobster\gpucohp")
sys.path.insert(0, r"D:\work\lobster\gpucohp\tests")
from gpucohp.sto import read_lobster_basisfunctions
from gpucohp.structure import Structure
from gpucohp.basis import LocalBasis
from gpucohp.twocenter import RadialFTTable, TwoCenterOverlap
from refmat import read_matrices

REF = r"D:\work\lobster\ref_TiB2_sber"
st = Structure.from_poscar(REF + r"\POSCAR")
rad = read_lobster_basisfunctions(REF + r"\basisFunctions.lobster")
basis = LocalBasis(st, rad, {"Ti": ["3s", "3p", "3d", "4s"], "B": ["2s", "2p"]})
print(basis.describe())
names = basis.names()
ref = read_matrices(REF + r"\overlapMatrices.lobster", maxk=3)
print("ref names:", ref[0]["names"])
assert ref[0]["names"] == names, "basis ordering differs from LOBSTER"

t0 = time.time()
table = RadialFTTable(basis.radials, qmax=400.0, nq=40001)
print("FT table", time.time() - t0, "s")
for rmax in (20.0,):
    t0 = time.time()
    tc = TwoCenterOverlap(basis, st, table, rmax_pair=rmax)
    print(f"rmax={rmax}: {len(tc.entries)} pair images, {time.time()-t0:.1f} s")
    for r in ref:
        Sk = tc.Sk(r["kpoint"])
        dev = np.abs(Sk - r["mat"])
        print(f"  k={r['kpoint']}  max|dS|={dev.max():.2e}  at {np.unravel_index(dev.argmax(), dev.shape)}  "
              f"diag ref {np.real(np.diag(r['mat']))[:3].round(6)} ours {np.real(np.diag(Sk))[:3].round(6)}")
Sk = tc.Sk(ref[0]["kpoint"])
np.set_printoptions(precision=5, suppress=True, linewidth=200)
print("ours Gamma block Ti:\n", Sk.real[:10, :10])
print("ref  Gamma block Ti:\n", ref[0]["mat"].real[:10, :10])
print("ours Ti-B block:\n", Sk.real[:10, 10:14])
print("ref  Ti-B block:\n", ref[0]["mat"].real[:10, 10:14])
