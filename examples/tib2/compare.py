#!/usr/bin/env python3
"""Compare two ICOHPLIST.lobster (or ICOOPLIST / ICOBILIST) files.

    python3 compare.py cpu/ICOHPLIST.lobster gpu/ICOHPLIST.lobster

Rows are matched on (atom A, atom B, translation), not on row order, because
the two codes may enumerate the pairs differently. Relative errors are only
quoted for bonds above a threshold: a pair whose ICOHP is 2e-5 eV turns a
5e-5 eV difference into a meaningless "250 %".
"""
import sys


def read(path):
    rows = {}
    for line in open(path).readlines()[2:]:
        f = line.split()
        if len(f) < 8:
            continue
        key = (f[1], f[2], int(f[4]), int(f[5]), int(f[6]))
        rows[key] = (float(f[3]), float(f[7]))          # distance, integral at E_F
    return rows


def main(pa, pb, thresh=0.5):
    a, b = read(pa), read(pb)
    common = sorted(set(a) & set(b))
    atoms = [k for k in common if "_" not in k[0] and "_" not in k[1]]
    print(f"{pa}: {len(a)} rows   {pb}: {len(b)} rows   matched: {len(common)}"
          f"  ({len(atoms)} atom-atom, rest orbital-resolved)")
    only = [k for k in a if k not in b] + [k for k in b if k not in a]
    if only:
        print(f"  WARNING: {len(only)} rows present in only one file, e.g. {only[:3]}")
    if not common:
        return 1
    dd = [abs(a[k][0] - b[k][0]) for k in common]
    dv = [abs(a[k][1] - b[k][1]) for k in common]
    big = [k for k in common if abs(b[k][1]) > thresh]
    print(f"  max |d distance| : {max(dd):.2e} A")
    print(f"  max |d value|    : {max(dv):.5f}   mean |d value| : {sum(dv)/len(dv):.6f}")
    if big:
        rel = [abs(a[k][1] - b[k][1]) / abs(b[k][1]) for k in big]
        print(f"  bonds with |value| > {thresh}: {len(big)}, max relative error {100*max(rel):.3f} %")
    worst = max(common, key=lambda k: abs(a[k][1] - b[k][1]))
    print(f"  worst row: {worst}  {pa} {a[worst][1]:9.5f}   {pb} {b[worst][1]:9.5f}")
    for k in atoms[:8]:
        print(f"    {k[0]:>9s} {k[1]:>9s} {k[2]:3d}{k[3]:3d}{k[4]:3d}  "
              f"{a[k][1]:9.5f}  {b[k][1]:9.5f}   d={a[k][1]-b[k][1]:+.5f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2],
                  float(sys.argv[3]) if len(sys.argv) > 3 else 0.5))
