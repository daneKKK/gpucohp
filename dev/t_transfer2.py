import sys, time, pickle, numpy as np
sys.path.insert(0, r"D:\work\lobster\gpucohp"); sys.path.insert(0, r"D:\work\lobster\gpucohp\tests")
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
w = Wavecar(REF + r"\WAVECAR"); st = Structure.from_poscar(REF + r"\POSCAR"); sp = read_potcar(REF + r"\POTCAR")
paw = PawSetup(st, sp); rad = read_lobster_basisfunctions(REF + r"\basisFunctions.lobster")
basis = LocalBasis(st, rad, {"Ti": ["3s", "3p", "3d", "4s"], "B": ["2s", "2p"]})
import os
if os.path.exists("tests/aug6.pkl"):
    aug = pickle.load(open("tests/aug6.pkl","rb"))
else:
    t0=time.time(); aug = Augmentation(basis, st, paw, rmax_pair=6.0); print("aug", time.time()-t0); pickle.dump(aug, open("tests/aug6.pkl","wb"))
ref = read_matrices(REF + r"\transferMatrices.lobster", maxk=2)
np.set_printoptions(precision=5, suppress=True, linewidth=200)
full_blocks = [b.copy() for b in aug.blocks]
for mode in ("all", "onsite"):
    for i,(b, ia_mu, ia_a, N) in enumerate(zip(full_blocks, aug.mu_atom, aug.a_atom, aug.Nimg)):
        onsite = (ia_mu == ia_a) and tuple(N) == (0,0,0)
        aug.blocks[i] = b if (mode == "all" or onsite) else np.zeros_like(b)
    for r in ref:
        ik = r["ik"]-1
        T, T_pw, P, A = transfer_matrix(basis, st, paw, aug, w, 0, ik, nbands=18)
        dev = np.abs(T - r["mat"])
        print(f"{mode:7s} k{ik}: max|dT|={dev.max():.2e} at {np.unravel_index(dev.argmax(), dev.shape)}  4s row: {T.real[1,[0,4,8]]} ref {r['mat'].real[1,[0,4,8]]}  B2s row: {T.real[10,[0,4,8]]} ref {r['mat'].real[10,[0,4,8]]}")
