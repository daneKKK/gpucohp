"""Command-line driver:  python -m gpucohp [workdir] [--device cuda] [--basis-dir DIR]

Reads lobsterin, POSCAR, POTCAR, WAVECAR, vasprun.xml from the working
directory and writes LOBSTER-compatible output files.
"""
import argparse
import glob
import os
import sys
import time
import numpy as np
import torch

from . import __version__
from .wavecar import Wavecar
from .potcar import read_potcar
from .structure import Structure
from .sto import read_lobster_basisfunctions, read_sto_file
from .basis import LocalBasis
from .paw import PawSetup
from .augment import Augmentation
from .twocenter import RadialFTTable, TwoCenterOverlap
from .engine import ProjectionEngine
from .kweights import read_kweights
from .lobsterin import LobsterIn
from .analysis import build_pairs, EnergyGrid, Analysis
from . import output as out


class Logger:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


KOGA_FILE = os.path.join(os.path.dirname(__file__), "basis_data", "koga.lobster")


class BasisLibrary(dict):
    """{element: {label: RadialSTO}} plus the elements that fell back to Koga."""
    fallback = ()


def load_basis_library(basisset, basis_dir, custom, elements=None, log=print):
    """{element: {label: RadialSTO}} for basis set `basisset`.

    Directories are searched in the order basis_dir (--basis-dir), the user
    library (gpucohp-basis import), the shipped open sets (koga, bunge; built
    from the published tables by tools/make_*_basis.py). Files are
    writeBasisFunctions dumps whose name contains the set name, and custom
    LOBSTER .sto files named <element>.sto. Elements the requested set does
    not provide fall back to Koga. pbeVaspFit2015, LOBSTER's default, is not
    redistributable; licensed LOBSTER users import it from their own runs."""
    from .userbasis import library_dir
    pkg = os.path.join(os.path.dirname(__file__), "basis_data")
    lib = {}
    for d in [basis_dir, library_dir(), pkg]:
        if d and os.path.isdir(d):
            for el, fns in _load_dir(basisset, d).items():
                for lab, f in fns.items():
                    lib.setdefault(el, {}).setdefault(lab, f)
    return _finish_library(lib, basisset, custom, elements, log)


def _load_dir(basisset, basis_dir):
    """Functions of basis set `basisset` found in one directory."""
    lib = {}
    for f in sorted(glob.glob(os.path.join(basis_dir, "*.lobster"))):
        if basisset.lower() in os.path.basename(f).lower() or basisset == "custom":
            for el, d in read_lobster_basisfunctions(f).items():
                lib.setdefault(el, {}).update(d)
    for f in sorted(glob.glob(os.path.join(basis_dir, "*.sto"))):
        el = os.path.basename(f).split(".")[0].split("_")[0]
        lib.setdefault(el, {}).update(read_sto_file(f, element=el))
    return lib


def _finish_library(lib, basisset, custom, elements, log):
    """Adds custom .sto files and the Koga fallback; records fallback elements in lib.fallback."""
    for el, f in custom.items():
        lib.setdefault(el, {}).update(read_sto_file(f, element=el))
    if basisset.lower() != "koga" and os.path.exists(KOGA_FILE):
        koga = read_lobster_basisfunctions(KOGA_FILE)
        fallback = [el for el in (elements or koga) if el in koga and el not in lib]
        if fallback:
            log(f"WARNING: basis set '{basisset}' not found for {' '.join(fallback)}; using the "
                f"Koga-Thakkar set for those. With a LOBSTER licence, import your own LOBSTER run "
                f"(keyword writeBasisFunctions) with 'gpucohp-basis import <dir>' to use '{basisset}'.")
        for el in fallback:
            lib[el] = koga[el]
    else:
        fallback = []
    lib = BasisLibrary(lib)
    lib.fallback = fallback
    return lib


def valence_orbitals(labels, z, zval):
    """Valence shells of a free atom as LOBSTER picks them from the POTCAR.

    The atom's occupied orbitals (the library entries) are filled with z
    electrons in Madelung order; shells are then taken from the outermost
    (highest n, then highest l) inwards until they hold zval electrons
    (e.g. C: 2s 2p; Ti_sv, zval 12: 3s 3p 3d 4s; Ga_d, zval 13: 3d 4s 4p)."""
    lval = {"s": 0, "p": 1, "d": 2, "f": 3}
    order = sorted(labels, key=lambda o: (int(o[0]) + lval[o[1]], int(o[0])))
    occ, left = {}, z
    for o in order:
        occ[o] = min(2 * (2 * lval[o[1]] + 1), max(left, 0))
        left -= occ[o]
    chosen, n = [], 0.0
    for o in sorted(labels, key=lambda o: (int(o[0]), lval[o[1]]), reverse=True):
        if n >= zval - 1e-6:
            break
        chosen.append(o)
        n += occ[o]
    return sorted(chosen, key=lambda o: (int(o[0]), lval[o[1]]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gpucohp")
    ap.add_argument("workdir", nargs="?", default=".")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: lobsterin 'device' or auto)")
    ap.add_argument("--basis-dir", default=None,
                    help="extra basis directory, searched before the user library and the shipped sets")
    ap.add_argument("--rmax-overlap", type=float, default=None, help="lattice-sum cut-off for overlaps (A)")
    ap.add_argument("--nbands", type=int, default=None)
    ap.add_argument("--single", action="store_true", help="complex64 arithmetic on the device")
    ap.add_argument("--integration", choices=("auto", "gaussian", "tetrahedron"), default="auto",
                    help="energy integration: tetrahedra when VASP used them (ISMEAR=-5), else Gaussian (default auto)")
    ap.add_argument("--tetra-scheme", choices=("lobster", "linear", "blochl"), default="lobster",
                    help="tetrahedron weights: LOBSTER's corner average (default), linear, or linear with Bloechl corrections")
    ap.add_argument("--occupations", choices=("lobster", "vasp"), default="vasp",
                    help="weights for charge spilling, populations and COBI: VASP's occupations (default) "
                         "or those of gpucohp's energy integration")
    ap.add_argument("--rank-deficient", choices=("complete", "drop"), default="complete",
                    help="bands that leave a basis direction uncovered: complete it (default) or leave it empty")
    args = ap.parse_args(argv)
    os.chdir(args.workdir)
    log = Logger("lobsterout")
    t_start = time.time()
    log(f"gpucohp {__version__} - GPU re-implementation of the LOBSTER projection scheme")
    log(f"starting in {os.getcwd()}")
    li = LobsterIn.read("lobsterin")
    if li.unsupported:
        log(f"WARNING: lobsterin keywords not supported by gpucohp, ignored: {' '.join(li.unsupported)}")
    device = args.device or li.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.complex64 if (args.single or li.precision == "single") else torch.complex128
    log(f"device: {device}  ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}), dtype {dtype}")

    # ---- input data
    st = Structure.from_poscar("CONTCAR" if os.path.exists("CONTCAR") else "POSCAR")   # LOBSTER uses VASP's (wrapped) positions
    species = read_potcar("POTCAR")
    wav = Wavecar("WAVECAR")
    log(wav.summary())
    kw = read_kweights("vasprun.xml", nk=wav.nkpts)
    lib = load_basis_library(li.basisset, args.basis_dir, li.custom_sto,
                             elements=sorted(set(st.symbols)), log=log)
    from .structure import ATOMIC_NUMBER
    req = {}
    for isp, el in enumerate(st.symbols):
        if el in li.basisfunctions:
            req[el] = li.basisfunctions[el]
        elif el in lib:
            # LOBSTER's rule: valence shells implied by the POTCAR, optionally
            # restricted to the l-channels of includeOrbitals
            zval = species[isp].zval
            req[el] = valence_orbitals(list(lib[el]), ATOMIC_NUMBER[el], zval)
            if li.include_orbitals:
                req[el] = [o for o in req[el] if o[1] in li.include_orbitals]
            log(f"basis functions for {el} (ZVAL {zval:g}): {' '.join(req[el])}")
        else:
            log(f"ERROR: no basis functions available for element {el}. Provide a LOBSTER basisFunctions.lobster "
                f"dump or a custom .sto file in {args.basis_dir}")
            sys.exit(1)
    try:
        basis = LocalBasis(st, lib, req)
    except ValueError as err:
        log(f"ERROR: {err}")
        sys.exit(1)
    log("setting up local basis functions...")
    for el in dict.fromkeys(st.symbols):
        bset = "koga" if el in lib.fallback else li.basisset
        log(f"{el} ({bset}) " + " ".join(f.name for f in basis.functions if f.element == el and f.iatom ==
                                         st.symbol_of_atom.index(el)))
    if wav.nbands < basis.nbasis:
        log(f"WARNING: only {wav.nbands} bands for {basis.nbasis} basis functions - pCOHP cannot be reconstructed")
    elif wav.nbands > basis.nbasis:
        log(f"INFO: PAW bands from {basis.nbasis + 1} upwards will be ignored")
    paw = PawSetup(st, species)
    aug = Augmentation(basis, st, paw, mode=li.augmentation, rmax_pair=8.0)

    # ---- overlaps
    rmax = args.rmax_overlap or max(r.rcut(1e-6) for r in basis.radials.values())
    log(f"calculating two-centre overlaps (lattice sums up to {rmax:.1f} A)...")
    t0 = time.time()
    table = RadialFTTable(basis.radials)
    ov = TwoCenterOverlap(basis, st, table, rmax_pair=rmax, log=log, device=device)
    log(f"  {len(ov.entries)} pair images, {time.time() - t0:.1f} s")

    # ---- projection
    eng = ProjectionEngine(wav, st, basis, paw, aug, ov, device=device, dtype=dtype, nbands_use=args.nbands, log=log,
                           rank_deficient=args.rank_deficient)
    log("projecting...")
    t0 = time.time()
    eng.run(kw)
    log(f"projection finished in {time.time() - t0:.1f} s")
    nbad = sum(1 for s in range(wav.nspin) for ik in range(wav.nkpts) if eng.results[s][ik]["band_dev"] > 1e-5)
    if nbad:
        log(f"WARNING: {nbad} of {wav.nkpts} k-points could not be orthonormalized with an accuracy of 1.0E-5.")
        out.write_bandoverlaps("bandOverlaps.lobster", eng)
    ndef = sum(1 for s in range(wav.nspin) for ik in range(wav.nkpts) if eng.results[s][ik]["min_sv"] < 1e-6)
    if ndef:
        log(f"WARNING: at {ndef} k-points the retained bands leave a basis direction uncovered (rank-deficient "
            f"projection); it was completed by the orthogonal complement, as LOBSTER does. Results there depend "
            f"on that completion; a basis that describes all retained bands (e.g. extra polarisation functions) avoids it.")

    # ---- analysis
    pairs = build_pairs(st, li)
    log(f"setting up CO interactions... found {len(pairs)} interactions.")
    eg = EnergyGrid(li.estart, li.eend, li.esteps, wav.efermi)
    an = Analysis(eng, st, basis, pairs, eg, li.sigma, device=device)
    from .tetra import read_tetrahedra
    tet = None if (args.integration == "gaussian" or li.integration == "gaussian") else read_tetrahedra("vasprun.xml")
    if tet is not None:
        an.use_tetrahedra(tet, scheme=args.tetra_scheme)
        integ = f"tetrahedron method ({len(tet[1])} tetrahedra, {args.tetra_scheme} weights)"
    else:
        if args.integration == "tetrahedron":
            log("ERROR: --integration tetrahedron needs a VASP run with ISMEAR = -5 (tetrahedra in vasprun.xml)")
            sys.exit(1)
        integ = f"Gaussian smearing (sigma={li.sigma}eV)"
    # occupations for charge spilling, populations and COBI: VASP's (default) or
    # those of the energy integration (tetrahedra / Gaussian)
    occ = None if args.occupations == "vasp" else [an.occupations(s) for s in range(wav.nspin)]
    an.occ_override = occ
    spill_c, spill_t = eng.spilling(occ=occ)
    if wav.nspin == 2:
        for s in range(2):
            sc, stt = eng.spilling(spins=[s], occ=occ)
            log(f"spillings for spin channel {s + 1}")
            log(f"abs. charge spilling: {100 * sc:6.2f}%")
            log(f"abs. total spilling:  {100 * stt:6.2f}%")
    else:
        log(f"abs. charge spilling: {100 * spill_c:6.2f}%")
        log(f"abs. total spilling:  {100 * spill_t:6.2f}%")
    zval_atoms = np.array([species[st.species_of_atom[ia]].zval for ia in range(st.natoms)])
    from .structure import ATOMIC_NUMBER
    znum_atoms = np.array([ATOMIC_NUMBER.get(st.symbol_of_atom[ia], 0) for ia in range(st.natoms)])
    if "dos" not in li.skip:
        log(f"calculating pDOS... using {integ}")
        dosr = [an.dos(s) for s in range(wav.nspin)]
        out.write_doscar("DOSCAR.lobster", st, basis, dosr, eg, wav.efermi, znum_atoms)
        log("writing DOSCAR.lobster...")
    for kind, skipkey, car, ilist in (("coop", "coop", "COOPCAR.lobster", "ICOOPLIST.lobster"),
                                      ("cohp", "cohp", "COHPCAR.lobster", "ICOHPLIST.lobster"),
                                      ("cobi", "cobi", "COBICAR.lobster", "ICOBILIST.lobster")):
        if skipkey in li.skip or not pairs:
            continue
        log(f"calculating p{kind.upper()}s...")
        t0 = time.time()
        res = [an.bonding(kind, s) for s in range(wav.nspin)]
        out.write_car(car, kind, st, basis, pairs, res, eg, wav.efermi, all_points=tet is not None)
        out.write_ilist(ilist, kind, st, basis, pairs, res)
        log(f"writing {car} and {ilist} ({time.time() - t0:.1f} s)")
    if "populationanalysis" not in li.skip:
        pops = [an.populations(s) for s in range(wav.nspin)]
        mull, loew = sum(p[0] for p in pops), sum(p[1] for p in pops)
        ma, la = an.atom_populations(mull, loew)
        out.write_charge("CHARGE.lobster", st, ma, la, zval_atoms)
        if "grosspopulation" not in li.skip:
            out.write_grosspop("GROSSPOP.lobster", st, basis, [p[0] for p in pops], [p[1] for p in pops])
        log(f"number of electrons recovered by projection: {ma.sum():.4f} of {zval_atoms.sum():g}")
        log("writing CHARGE.lobster and GROSSPOP.lobster...")
    if lib.fallback or li.unsupported:
        log("")
        if lib.fallback:
            log(f"NOTE: basis set {li.basisset} was not available for {' '.join(lib.fallback)}; the open Koga set "
                f"was used instead. Results can differ strongly for transition metals (see README, Basis sets).")
        if li.unsupported:
            log(f"NOTE: ignored lobsterin keywords: {' '.join(li.unsupported)}")
    dt = time.time() - t_start
    log(f"finished in {int(dt // 3600)} h {int(dt % 3600 // 60)} min {dt % 60:.1f} s of wall time")


if __name__ == "__main__":
    main()
