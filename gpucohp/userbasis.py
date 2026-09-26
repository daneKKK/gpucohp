"""User basis library: basis sets that gpucohp may not redistribute.

LOBSTER's default basis, pbeVaspFit2015, is part of the LOBSTER program and
covered by its licence; its parameters are not published. A licensed LOBSTER
user can still let gpucohp use it: every LOBSTER run with the keyword
`writeBasisFunctions` writes the functions it used to basisFunctions.lobster.

    gpucohp-basis import path/to/run            # run dir with basisFunctions.lobster + lobsterout
    gpucohp-basis list

merges those functions into the user library ($GPUCOHP_BASIS_DIR, default
~/.gpucohp/basis), one file per basis set (<set>.lobster). The basis set of
every element is taken from the run's lobsterout ("Ti (pbeVaspFit2015) 4s ..."),
so functions of different sets never mix. gpucohp reads this library after
--basis-dir and before its own open sets (koga, bunge). Each run adds the
elements and orbitals it used; a library grows with the user's LOBSTER runs.
"""
import argparse
import os
import re
import sys

from .sto import read_lobster_basisfunctions


def library_dir():
    return os.path.expanduser(os.environ.get("GPUCOHP_BASIS_DIR", "~/.gpucohp/basis"))


def _sets_from_lobsterout(path):
    """{element: basis set name} from LOBSTER's 'El (set) orbitals...' lines."""
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\s*([A-Z][a-z]?)\s+\(([^)]+)\)\s+\d[spdf]", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _raw_blocks(path):
    """[(element, label, block text)] of a basisFunctions.lobster file, first occurrence only."""
    txt = open(path).read()
    blocks, seen = [], set()
    for b in re.split(r"\n(?=\S+\s+\d[spdf]\S*\s+at atom)", txt):
        m = re.match(r"\s*(\S+)\s+(\d[spdf])\S*\s+at atom", b)
        if not m or (m.group(1), m.group(2)) in seen:
            continue
        seen.add((m.group(1), m.group(2)))
        body = [ln for ln in b.strip().splitlines()[1:] if ln.strip()]
        blocks.append((m.group(1), m.group(2), "\n".join(body)))
    return blocks


def import_run(rundir, libdir=None):
    libdir = libdir or library_dir()
    bf, lo = os.path.join(rundir, "basisFunctions.lobster"), os.path.join(rundir, "lobsterout")
    if not os.path.exists(bf):
        sys.exit(f"{bf} not found - run LOBSTER with the keyword writeBasisFunctions")
    if not os.path.exists(lo):
        sys.exit(f"{lo} not found - needed to know which basis set each element used")
    sets = _sets_from_lobsterout(lo)
    os.makedirs(libdir, exist_ok=True)
    added = {}
    for el, lab, body in _raw_blocks(bf):
        bset = sets.get(el)
        if bset is None:
            print(f"  skipping {el} {lab}: basis set not stated in lobsterout")
            continue
        if bset.lower() in ("koga", "bunge"):
            continue                                  # shipped with gpucohp, built from the published tables
        target = os.path.join(libdir, f"{bset}.lobster")
        have = read_lobster_basisfunctions(target) if os.path.exists(target) else {}
        if lab in have.get(el, {}):
            continue
        new = not os.path.exists(target)
        with open(target, "a", newline="\n") as f:
            if new:
                f.write(f"Basis set {bset}, imported by gpucohp-basis from the user's own LOBSTER runs.\n"
                        "q and coeff are dimensionless, but alpha is given in 1/a_0\n"
                        "Covered by the LOBSTER licence of the user who ran LOBSTER: do not redistribute.\n\n")
            f.write(f"{el} {lab:<14s} at atom 0\n{body}\n\n")
        added.setdefault(bset, []).append(f"{el} {lab}")
    for bset, items in added.items():
        print(f"  {bset}: added {' '.join(items)}")
    if not added:
        print("  nothing new")
    print(f"library: {libdir}")


def list_library(libdir=None):
    libdir = libdir or library_dir()
    if not os.path.isdir(libdir):
        print(f"no user basis library at {libdir}")
        return
    for f in sorted(os.listdir(libdir)):
        if f.endswith(".lobster"):
            lib = read_lobster_basisfunctions(os.path.join(libdir, f))
            print(f"{f[:-8]}: " + ", ".join(f"{el} {' '.join(sorted(d))}" for el, d in sorted(lib.items())))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gpucohp-basis", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="add the functions of a LOBSTER run to the user library")
    p.add_argument("rundirs", nargs="+")
    sub.add_parser("list", help="show the user library")
    args = ap.parse_args(argv)
    if args.cmd == "import":
        for d in args.rundirs:
            import_run(d)
    else:
        list_library()


if __name__ == "__main__":
    main()
