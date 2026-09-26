"""LOBSTER (cpu*) vs gpucohp (gpu_*) on the unchanged LOBSTER examples."""
import re, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from importlib import util
spec = util.spec_from_file_location("cb", os.path.join(os.path.dirname(__file__), "..", "maintz2013", "compare.py"))
src = open(spec.origin).read().split("print(f\"{'system'")[0]
exec(src)                                   # ilist, charges, spill, wall

def cohpcar(path):
    lines = open(path).read().splitlines()
    npair, nspin, nE = (int(x) for x in lines[1].split()[:3])
    data = np.array([[float(x) for x in l.split()] for l in lines[3 + npair:]])
    return data                                              # (nE-1, 1 + 2*(npair+1)*nspin)

def row(label, c, g):
    A, B = ilist(c + "/ICOHPLIST.lobster"), ilist(g + "/ICOHPLIST.lobster")
    com = [k for k in A if k in B]
    d = max(abs(A[k] - B[k]) for k in com)
    k = max(com, key=lambda k: abs(A[k] - B[k]))
    qa, qb = charges(c + "/CHARGE.lobster"), charges(g + "/CHARGE.lobster")
    dq = max(max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(qa, qb))
    cur = ""
    if os.path.exists(c + "/COHPCAR.lobster") and os.path.exists(g + "/COHPCAR.lobster"):
        X, Y = cohpcar(c + "/COHPCAR.lobster"), cohpcar(g + "/COHPCAR.lobster")
        # align rows on the energy column
        common = np.intersect1d(np.round(X[:, 0], 4), np.round(Y[:, 0], 4))
        X = X[np.isin(np.round(X[:, 0], 4), common)]; Y = Y[np.isin(np.round(Y[:, 0], 4), common)]
        colsA, colsB = X[:, 3::2], Y[:, 3::2]
        amp = np.abs(colsA).max()
        dcur = np.abs(colsA - colsB).max()
        # best one-grid-point shift, to detect curve placement conventions
        cur = f"curve max|d| {dcur:.4f} of {amp:.3f} ({len(X)} pts)"
    print(f"{label:28s} ICOHP {B[k]:9.4f} vs LOBSTER {A[k]:9.4f}  max|d| {d:.4f}   dq {dq:.2f}   "
          f"spill {spill(c + '/lobsterout')} / {spill(g + '/lobsterout')}   {cur}")

R = sys.argv[1]
for s, c, gs in [("diamond", "cpu", ["tet", "tetlin", "gauss"]), ("gaas", "cpu", ["tet", "tetlin", "gauss"]),
                 ("ti", "cpu", ["pbe_tet", "pbe_vaspocc"]), ("ti", "cpu_koga", ["koga_tet", "koga_vaspocc"]),
                 ("feni3", "cpu", ["pbe"]), ("feni3", "cpu_koga", ["koga", "koga_vaspocc"])]:
    for g in gs:
        if os.path.exists(f"{R}/{s}/gpu_{g}/ICOHPLIST.lobster"):
            row(f"{s} {g} vs LOBSTER {c}", f"{R}/{s}/{c}", f"{R}/{s}/gpu_{g}")
