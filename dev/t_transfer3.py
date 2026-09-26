import sys, time, numpy as np
sys.path.insert(0, r"D:\work\lobster\gpucohp"); sys.path.insert(0, r"D:\work\lobster\gpucohp\tests")
from gpucohp.wavecar import Wavecar
from gpucohp.potcar import read_potcar
from gpucohp.structure import Structure
from gpucohp.sto import read_lobster_basisfunctions
from gpucohp.basis import LocalBasis
from gpucohp.paw import PawSetup
from gpucohp.augment import Augmentation
from gpucohp.projection import transfer_matrix
from gpucohp.twocenter import RadialFTTable, TwoCenterOverlap
from refmat import read_matrices
REF = r"D:\work\lobster\ref_TiB2_sber"
w = Wavecar(REF + r"\WAVECAR"); st = Structure.from_poscar(REF + r"\POSCAR"); sp = read_potcar(REF + r"\POTCAR")
paw = PawSetup(st, sp); rad = read_lobster_basisfunctions(REF + r"\basisFunctions.lobster")
basis = LocalBasis(st, rad, {"Ti": ["3s", "3p", "3d", "4s"], "B": ["2s", "2p"]})
t0=time.time(); aug = Augmentation(basis, st, paw); print("onsite aug", time.time()-t0, "s")
table = RadialFTTable(basis.radials, qmax=400.0, nq=40001)
import pickle, os
if os.path.exists("tests/tc12.pkl"):
    tc = pickle.load(open("tests/tc12.pkl", "rb"))
else:
    t0 = time.time(); tc = TwoCenterOverlap(basis, st, table, rmax_pair=12.0); print("S(T)", time.time()-t0); pickle.dump(tc, open("tests/tc12.pkl", "wb"))
refT = read_matrices(REF + r"\transferMatrices.lobster", maxk=6)
refC = read_matrices(REF + r"\coefficientMatrices.lobster", maxk=6)
refS = read_matrices(REF + r"\overlapMatrices.lobster", maxk=6)
def gauge_align(A, B):
    """multiply columns of A by phases so that they best match B"""
    ph = np.exp(-1j*np.angle(np.sum(np.conj(B)*A, axis=0)))
    return A*ph
for rT, rC, rS in zip(refT, refC, refS):
    ik = rT["ik"]-1
    T, T_pw, P, A = transfer_matrix(basis, st, paw, aug, w, 0, ik, nbands=18)
    S = rS["mat"]                    # LOBSTER's S to isolate the C = S^-1 T step
    C = np.linalg.solve(S, T)
    Cours = np.linalg.solve(tc.Sk(rT["kpoint"]), T)
    if ik == 0:
        print("ours imag row0:", T.imag[0, :5], " ref imag row0:", rT["mat"].imag[0, :5])
    print(f"k{ik} {rT['kpoint']}: max|dT| = {np.abs(T-rT['mat']).max():.2e}  max|conj(T)-Tref| = {np.abs(np.conj(T)-rT['mat']).max():.2e} (max|T| {np.abs(T).max():.2f})"
          f"   C=S_ref^-1 T vs ref: {np.abs(C - rC['mat']).max():.2e}   C=S_ours^-1 T vs ref: {np.abs(Cours - rC['mat']).max():.2e}  conj: {np.abs(np.conj(Cours) - rC['mat']).max():.2e}")
