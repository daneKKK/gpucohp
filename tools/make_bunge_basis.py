#!/usr/bin/env python3
"""Build gpucohp's "bunge" basis library from the published Bunge tables.

    python tools/make_bunge_basis.py [third_party/bunge1993/RHF.TABLES.txt] [gpucohp/basis_data/bunge.lobster]

Source: C. F. Bunge, J. A. Barrientos, A. V. Bunge, At. Data Nucl. Data Tables
53, 113 (1993) - Roothaan-Hartree-Fock ground states He-Xe in normalised STOs.
LOBSTER calls this set basisSet "bunge". Every occupied orbital is written as
one contracted STO in the writeBasisFunctions format read by gpucohp.
In the tables the p and d blocks of an atom are printed side by side, so a
line is split at every primitive label (1S, 2P, 3D, ...), not by column.
"""
import re
import sys

LABEL = re.compile(r"^\d[SPDF]$")
ORB = re.compile(r"^\d[spdf]$")
HEAD = re.compile(r"^([A-Z]+), Z=(\d+)\s+(.*)")

SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
           "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
           "Sn Sb Te I Xe").split()


def parse(path):
    atoms = []                      # (symbol, config, {orb: [(n, alpha, c)]})
    cur = orbs = None
    for line in open(path):
        m = HEAD.match(line.strip())
        if m:
            orbs = {}
            cur = (SYMBOLS[int(m.group(2)) - 1], m.group(3).strip(), orbs)
            atoms.append(cur)
            groups = {}
            continue
        if cur is None:
            continue
        tok = line.split()
        if tok and all(ORB.match(t) for t in tok):
            groups = {}             # orbital names of the block(s) that follow, by symmetry
            for t in tok:
                groups.setdefault(t[1].upper(), []).append(t)
            continue
        if not tok or not LABEL.match(tok[0]):
            continue
        # one or two primitive records on this line, e.g. "2P 30.96 c c c   3D 15.74 c"
        i = 0
        while i < len(tok):
            lab = tok[i]
            names = groups[lab[1]]
            alpha = float(tok[i + 1])
            coeffs = [float(x) for x in tok[i + 2:i + 2 + len(names)]]
            for name, c in zip(names, coeffs):
                orbs.setdefault(name, []).append((int(lab[0]), alpha, c))
            i += 2 + len(names)
    return atoms


def main(src="third_party/bunge1993/RHF.TABLES.txt", dst="gpucohp/basis_data/bunge.lobster"):
    out = ["This file lists Bunge-Barrientos-Bunge Roothaan-Hartree-Fock STO orbitals (basisSet bunge).",
           "q and coeff are dimensionless, but alpha is given in 1/a_0",
           "Source: At. Data Nucl. Data Tables 53, 113 (1993) [He-Xe]",
           ""]
    n = 0
    for el, config, orbs in parse(src):
        for name, prims in orbs.items():
            out.append(f"{el} {name:<14s} at atom 0")
            out.append("q        alpha       coeff")
            out += [f"{q}  {a:12.6f}  {c: .6f}" for q, a, c in prims]
            out.append("")
            n += 1
        print(f"  {el:2s}: {' '.join(orbs):32s} [{config}]")
    with open(dst, "w", newline="\n") as f:
        f.write("\n".join(out))
    print(f"wrote {n} radial functions to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
