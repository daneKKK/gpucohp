import sys, numpy as np
sys.path.insert(0, r"D:\work\lobster\gpucohp")
from gpucohp.wavecar import Wavecar
w = Wavecar(r"D:\work\lobster\ref_TiB2_sber\WAVECAR")
print(w.summary())
print("lattice\n", w.lattice)
print("k0", w.kpoints[0], "k1", w.kpoints[1], "nplane0", w.nplane[0, :3])
for ik in (0, 1, 100, 2924):
    G = w.gvectors(ik)
    C = w.coeffs(0, ik)
    norms = np.sum(np.abs(C) ** 2, axis=1)
    print(ik, len(G), "norms(first 3, min, max):", norms[:3].round(5), norms.min().round(5), norms.max().round(5))
print("eig k0 (eV):", w.eig[0, 0, :20].round(3))
print("occ k0:", w.occ[0, 0, :20].round(3))
