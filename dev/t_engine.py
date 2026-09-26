import sys, time, pickle, numpy as np, torch
sys.path.insert(0, r"D:\work\lobster\gpucohp"); sys.path.insert(0, r"D:\work\lobster\gpucohp\tests")
from gpucohp.wavecar import Wavecar
from gpucohp.potcar import read_potcar
from gpucohp.structure import Structure
from gpucohp.sto import read_lobster_basisfunctions
from gpucohp.basis import LocalBasis
from gpucohp.paw import PawSetup
from gpucohp.augment import Augmentation
from gpucohp.engine import ProjectionEngine
from refmat import read_matrices
REF = r"D:\work\lobster\ref_TiB2_sber"
w = Wavecar(REF + r"\WAVECAR"); st = Structure.from_poscar(REF + r"\POSCAR"); sp = read_potcar(REF + r"\POTCAR")
paw = PawSetup(st, sp); rad = read_lobster_basisfunctions(REF + r"\basisFunctions.lobster")
basis = LocalBasis(st, rad, {"Ti": ["3s", "3p", "3d", "4s"], "B": ["2s", "2p"]})
aug = Augmentation(basis, st, paw)
tc = pickle.load(open("tests/tc12.pkl", "rb"))
eng = ProjectionEngine(w, st, basis, paw, aug, tc, device="cpu")
refT = read_matrices(REF + r"\transferMatrices.lobster", maxk=6)
refC = read_matrices(REF + r"\coefficientMatrices.lobster", maxk=6)
refS = read_matrices(REF + r"\overlapMatrices.lobster", maxk=6)
for rT, rC, rS in zip(refT, refC, refS):
    ik = rT["ik"] - 1
    t0 = time.time(); r = eng.project_k(0, ik); dt = time.time() - t0
    T = r["S"] @ r["C1"]   # not the raw T; recompute raw T for the check
    S = r["S"].numpy()
    Cn = np.linalg.solve(rS["mat"], rT["mat"])
    print(f"k{ik}: {dt*1000:.0f} ms  |S-Sref|max={np.abs(S-rS['mat']).max():.2e}  spill[:4]={r['spill'].numpy()[:4].round(5)} band_dev={r['band_dev']:.2e}  eig[:3]={r['eig'].numpy()[:3].round(3)}")
    # check C = S^-1 T from engine internals: rebuild T = S C (C before orthonormalisation not stored) -> compare C1 with ref C orthonormalised
    O = rC["mat"].conj().T @ rS["mat"] @ rC["mat"]
    wv, V = np.linalg.eigh(O); C1ref = rC["mat"] @ (V * wv**-0.5) @ V.conj().T
    print(f"     C1 vs ref-derived C1: {np.abs(r['C1'].numpy() - C1ref).max():.2e}")
# spilling over a subset of k
eng.kweights = np.full(w.nkpts, 1.0 / w.nkpts)
t0 = time.time()
for ik in range(0, 40):
    eng.results = getattr(eng, "results", [[None]*w.nkpts]); eng.results[0][ik] = {k:(v.numpy() if torch.is_tensor(v) else v) for k,v in eng.project_k(0, ik).items()}
print("40 k-points:", time.time()-t0, "s")
