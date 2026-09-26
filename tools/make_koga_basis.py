#!/usr/bin/env python3
"""Build gpucohp's open basis library from the published Koga-Thakkar tables.

    python tools/make_koga_basis.py [third_party/koga_thakkar] [gpucohp/basis_data/koga.lobster]

Source: analytical Roothaan-Hartree-Fock wave functions of the neutral
ground-state atoms,
  He-Xe  T. Koga, K. Kanayama, S. Watanabe, A. J. Thakkar,
         Int. J. Quantum Chem. 71, 491 (1999)
  Cs-Lr  T. Koga, K. Kanayama, T. Watanabe, T. Imai, A. J. Thakkar,
         Theor. Chem. Acc. 104, 411 (2000)
as distributed (CC0) in https://github.com/JFurness1/AtomicOrbitals,
literature_data/{k99l/neutral,k00heavy}. These are the same tables LOBSTER
calls basisSet "koga". Every occupied orbital of the atomic ground state is
written as one contracted STO:  R(r) = sum_a c_a N_a r^(n_a-1) exp(-alpha_a r),
alpha in 1/bohr, primitives normalised, exactly as tabulated.

Output format is the one LOBSTER writes with writeBasisFunctions, so the
gpucohp loader reads it unchanged; the file is selected by "basisSet koga".
"""
import os
import re
import sys

SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
           "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
           "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
           "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm "
           "Bk Cf Es Fm Md No Lr").split()
PRIM = re.compile(r"\s+(\d)([SPDF])\s+(\d*\.\d+)\s+(.*)")      # exponents < 1 are printed as .841438


def parse(path):
    """Koga-Thakkar table -> (header, [(label, [(n, alpha, c), ...]), ...])."""
    lines = open(path).read().splitlines()
    header = lines[0].strip()
    orbs, i = [], 0
    while i < len(lines):
        m = re.match(r"\s+([SPDF])\s{5,}(\d.*)", lines[i])
        if not m:
            i += 1
            continue
        names = m.group(2).split()
        i += 1
        rows = []
        while i < len(lines) and not (PRIM.match(lines[i]) or re.match(r"\s+[SPDF]\s{5,}\d", lines[i])):
            i += 1
        while i < len(lines) and PRIM.match(lines[i]):
            p = PRIM.match(lines[i])
            if p.group(2) != m.group(1):
                raise ValueError(f"{path}: symmetry mismatch in line {lines[i]!r}")
            rows.append((int(p.group(1)), float(p.group(3)), [float(x) for x in p.group(4).split()]))
            i += 1
        for j, name in enumerate(names):
            orbs.append((name.lower(), [(n, a, c[j]) for n, a, c in rows]))
    return header, orbs


def main(src="third_party/koga_thakkar", dst="gpucohp/basis_data/koga.lobster"):
    out = ["This file lists Koga-Thakkar analytical Hartree-Fock STO orbitals (basisSet koga).",
           "q and coeff are dimensionless, but alpha is given in 1/a_0",
           "Sources: Int. J. Quantum Chem. 71, 491 (1999) [He-Xe]; Theor. Chem. Acc. 104, 411 (2000) [Cs-Lr]",
           ""]
    count = 0
    for z, el in enumerate(SYMBOLS, start=1):
        path = None
        for sub in ("k99l/neutral", "k00heavy"):
            p = os.path.join(src, sub, el.lower())
            if os.path.exists(p):
                path = p
        if path is None:
            print(f"  {el:2s} (Z={z}): not tabulated, skipped")
            continue
        header, orbs = parse(path)
        for label, prims in orbs:
            out.append(f"{el} {label:<14s} at atom 0")
            out.append("q        alpha       coeff")
            out += [f"{n}  {a:12.6f}  {c: .12e}" for n, a, c in prims]
            out.append("")
            count += 1
        print(f"  {el:2s} (Z={z:3d}): {' '.join(l for l, _ in orbs):40s} [{header}]")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", newline="\n") as f:
        f.write("\n".join(out))
    print(f"wrote {count} radial functions to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
