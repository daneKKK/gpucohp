import sys, time, numpy as np
sys.path.insert(0, r"D:\work\lobster\gpucohp")
sys.path.insert(0, r"D:\work\lobster\gpucohp\tests")
from gpucohp.wavecar import Wavecar
from gpucohp.potcar import read_potcar
from gpucohp.structure import Structure
from gpucohp.sto import read_lobster_basisfunctions
from gpucohp.basis import LocalBasis
from gpucohp.paw import PawSetup
from gpucohp.augment import Augmentation
from gpucohp.projection import transfer_matrix
from refmat import read_matrices

REF = r"D:\work\lobster\ref_TiB2_sber"
w = Wavecar(REF + r"\WAVECAR")
st = Structure.from_poscar(REF + r"\POSCAR")
sp = read_potcar(REF + r"\POTCAR")
paw = PawSetup(st, sp)
rad = read_lobster_basisfunctions(REF + r"\basisFunctions.lobster")
basis = LocalBasis(st, rad, {"Ti": ["3s", "3p", "3d", "4s"], "B": ["2s", "2p"]})
t0 = time.time()
aug = Augmentation(basis, st, paw, rmax_pair=float(sys.argv[1]) if len(sys.argv) > 1 else 10.0)
print(f"augmentation: {len(aug.blocks)} (mu-atom, sphere, T) blocks in {time.time()-t0:.1f}s")
ref = read_matrices(REF + r"\transferMatrices.lobster", maxk=3)
np.set_printoptions(precision=5, suppress=True, linewidth=200)
for r in ref:
    ik = r["ik"] - 1
    T, T_pw, P, A = transfer_matrix(basis, st, paw, aug, w, 0, ik, nbands=18)
    dev = np.abs(T - r["mat"])
    print(f"k{ik} {r['kpoint']}: max|dT| = {dev.max():.2e} at {np.unravel_index(dev.argmax(), dev.shape)}; "
          f"max|dT_pw-only| = {np.abs(T_pw - r['mat']).max():.2e}")
    if ik == 0:
        print("ours   T[:6,:6].real\n", T.real[:6, :6])
        print("ref    T[:6,:6].real\n", r["mat"].real[:6, :6])
        print("pw-only T[:6,:6].real\n", T_pw.real[:6, :6])
        print("ours Ti1_4s row:", T.real[1, :10])
        print("ref  Ti1_4s row:", r["mat"].real[1, :10])
        print("A(k=0) Ti block (basis x proj on Ti):\n", A.real[:10, :18])
