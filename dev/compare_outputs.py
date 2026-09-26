"""Compare gpucohp output files with a LOBSTER reference directory.
usage: python compare_outputs.py <ours_dir> <ref_dir>"""
import sys, re, numpy as np

ours, ref = sys.argv[1], sys.argv[2]


def read_ilist(path):
    rows = {}
    for ln in open(path).readlines()[2:]:
        p = ln.split()
        if len(p) < 8:
            continue
        key = (p[1], p[2], int(p[4]), int(p[5]), int(p[6]))
        rows[key] = (float(p[3]), float(p[7]))
    return rows


def read_car(path):
    lines = open(path).read().splitlines()
    n = int(lines[1].split()[0])
    labels = lines[3:3 + n]
    data = np.array([[float(x) for x in ln.split()] for ln in lines[3 + n:] if ln.strip()])
    return labels, data


for kind in ("COHP", "COOP", "COBI"):
    try:
        a = read_ilist(f"{ours}/I{kind}LIST.lobster"); b = read_ilist(f"{ref}/I{kind}LIST.lobster")
    except FileNotFoundError as e:
        print(kind, "missing:", e); continue
    common = [k for k in a if k in b]
    atoms = [k for k in common if "_" not in k[0]]
    print(f"I{kind}LIST: {len(a)} ours / {len(b)} ref rows, {len(common)} common ({len(atoms)} atom pairs); "
          f"missing in ours: {[k for k in b if k not in a][:3]}")
    dv = np.array([a[k][1] - b[k][1] for k in common]); dd = np.array([a[k][0] - b[k][0] for k in common])
    va = np.array([a[k][1] for k in atoms]); vb = np.array([b[k][1] for k in atoms])
    print(f"   max|d dist| = {np.abs(dd).max():.1e}; atom pairs: max|dI| = {np.abs(va - vb).max():.5f} "
          f"(max|I_ref| {np.abs(vb).max():.4f}); all rows max|dI| = {np.abs(dv).max():.5f}")
    for k in atoms[:6]:
        print(f"     {k}: ours {a[k][1]:9.5f}  ref {b[k][1]:9.5f}")
    # a few orbital rows with the largest deviation
    worst = sorted(common, key=lambda k: -abs(a[k][1] - b[k][1]))[:4]
    for k in worst:
        print(f"     worst {k}: ours {a[k][1]:9.5f}  ref {b[k][1]:9.5f}")

for kind in ("COHP", "COOP", "COBI"):
    try:
        la, da = read_car(f"{ours}/{kind}CAR.lobster"); lb, db = read_car(f"{ref}/{kind}CAR.lobster")
    except FileNotFoundError as e:
        print(kind, "CAR missing:", e); continue
    print(f"{kind}CAR: labels equal: {la == lb}; shape ours {da.shape} ref {db.shape}")
    if da.shape == db.shape:
        print(f"   energy grid max diff {np.abs(da[:, 0] - db[:, 0]).max():.2e}; average col max|d| {np.abs(da[:, 1] - db[:, 1]).max():.5f} "
              f"(max {np.abs(db[:, 1]).max():.4f}); int avg max|d| {np.abs(da[:, 2] - db[:, 2]).max():.5f}; "
              f"all cols max|d| {np.abs(da[:, 1:] - db[:, 1:]).max():.5f}")
        i0 = np.argmin(np.abs(db[:, 0]))
        print(f"   at E=0: ours {da[i0, 1:7]} ref {db[i0, 1:7]}")

# DOSCAR
try:
    A = open(f"{ours}/DOSCAR.lobster").read().splitlines(); B = open(f"{ref}/DOSCAR.lobster").read().splitlines()
    n = int(A[5].split()[2])
    da = np.array([[float(x) for x in ln.split()] for ln in A[6:6 + n]]); db = np.array([[float(x) for x in ln.split()] for ln in B[6:6 + n]])
    print(f"DOSCAR total: max|dDOS| {np.abs(da[:, 1] - db[:, 1]).max():.5f} (max {db[:, 1].max():.3f}); int DOS at E=0: ours {da[np.argmin(np.abs(da[:, 0])), 2]:.4f} ref {db[np.argmin(np.abs(db[:, 0])), 2]:.4f}; final int {da[-1, 2]:.3f} / {db[-1, 2]:.3f}")
    print("   header ours:", A[5].strip(), "| ref:", B[5].strip())
    print("   atom1 header ours:", A[6 + n].strip()[:90]); print("   atom1 header ref: ", B[6 + n].strip()[:90])
    pa = np.array([[float(x) for x in ln.split()] for ln in A[7 + n:7 + 2 * n]]); pb = np.array([[float(x) for x in ln.split()] for ln in B[7 + n:7 + 2 * n]])
    print(f"   atom1 pDOS max|d| {np.abs(pa[:, 1:] - pb[:, 1:]).max():.5f} (max {pb[:, 1:].max():.3f})")
except Exception as e:
    print("DOSCAR compare failed:", e)
for f in ("CHARGE.lobster", "GROSSPOP.lobster"):
    print("==", f, "ours ||| ref")
    a = open(f"{ours}/{f}").read().splitlines(); b = open(f"{ref}/{f}").read().splitlines()
    for x, y in zip(a[:16], b[:16]):
        print(f"   {x[:60]:60s} ||| {y[:60]}")
