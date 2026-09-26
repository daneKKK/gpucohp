import sys, time, pickle, numpy as np
sys.path.insert(0, r"D:\work\lobster\gpucohp"); sys.path.insert(0, r"D:\work\lobster\gpucohp\tests")
from gpucohp.structure import Structure
from gpucohp.sto import read_lobster_basisfunctions
from gpucohp.basis import LocalBasis
from gpucohp.twocenter import RadialFTTable, TwoCenterOverlap
from refmat import read_matrices
REF = r"D:\work\lobster\ref_TiB2_sber"
st = Structure.from_poscar(REF + r"\POSCAR"); rad = read_lobster_basisfunctions(REF + r"\basisFunctions.lobster")
basis = LocalBasis(st, rad, {"Ti": ["3s", "3p", "3d", "4s"], "B": ["2s", "2p"]})
table = RadialFTTable(basis.radials, qmax=400.0, nq=40001)
t0 = time.time(); tc = TwoCenterOverlap(basis, st, table, rmax_pair=24.0); print("S(T) rmax24", time.time()-t0, flush=True)
pickle.dump(tc, open("tests/tc24.pkl", "wb"))
ref = read_matrices(REF + r"\overlapMatrices.lobster", maxk=4)
names = basis.names()
for rc in np.arange(8, 24.1, 1.0):
    sel = tc.dist <= rc
    res = []
    for r in ref:
        ph = np.exp(2j*np.pi*(tc.Nimg[sel] @ r["kpoint"]))
        Sk = np.einsum("t,tab->ab", ph, tc.S_T[sel])
        res.append(np.abs(Sk - r["mat"]).max())
    print(f"cutoff {rc:5.1f} A: max|dS| per k = {np.array(res).round(6)}   S44={Sk[1,1].real:.6f} (ref {ref[-1]['mat'][1,1].real:.6f}) S3p={Sk[2,2].real:.6f} (ref {ref[-1]['mat'][2,2].real:.6f})", flush=True)
