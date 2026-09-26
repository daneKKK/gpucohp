# gpucohp

Chemical-bonding analysis from a VASP `WAVECAR` on a GPU: the PAW
wavefunctions are projected onto atom-centred Slater-type orbitals, and
pCOHP, pCOOP, COBI, pDOS and Mulliken/Löwdin charges are written in LOBSTER's
file formats. gpucohp implements the projection scheme published by Maintz,
Deringer, Tchougréeff and Dronskowski (*J. Comput. Chem.* **34**, 2557 (2013);
**37**, 1030 (2016)) and reproduces LOBSTER 5.1.1 on the same input; it
contains no LOBSTER code. A 48-atom cell that takes LOBSTER 6.5 h on 12 cores
takes 1.5 min on one RTX 2080 Ti.

## Install

```bash
git clone https://github.com/TODO/gpucohp && cd gpucohp
conda env create -f environment.yml && conda activate gpucohp
```

or, into an existing environment with a CUDA build of PyTorch
([pytorch.org](https://pytorch.org/get-started/locally/) has the right command
for your GPU and driver):

```bash
pip install -e .
```

Tested: PyTorch 2.6 (CUDA 12.4) on RTX 2080 Ti, PyTorch 2.8 (CUDA 12.9) on
RTX PRO 4500 Blackwell. Without a GPU it runs on the CPU (`--device cpu`).
`pip install -e ".[test]" && pytest` runs the tests (CPU, about 1.5 min).

## Use

The inputs are LOBSTER's: in the VASP run directory, next to a `lobsterin`,

```bash
gpucohp .
```

reads `WAVECAR`, `POTCAR`, `CONTCAR` (or `POSCAR`), `vasprun.xml` and
`lobsterin`, and writes `ICOHPLIST.lobster`, `COHPCAR.lobster`, the COOP and
COBI files, `DOSCAR.lobster`, `CHARGE.lobster`, `GROSSPOP.lobster` and
`lobsterout`, in LOBSTER's formats. The VASP run needs what LOBSTER needs:
`ISYM = -1`, `LWAVE = .TRUE.`, `NBANDS` at least the number of basis functions,
and the standard (not gamma-only) executable. OUTCAR and KPOINTS are not needed.

`lobsterin` keywords: `COHPstartEnergy`, `COHPendEnergy`, `COHPSteps`,
`gaussianSmearingWidth`, `basisSet`, `basisFunctions`, `includeOrbitals`,
`customSTOforAtom`, `cohpBetween`, `cohpGenerator` (with `orbitalWise`), and
the `skip...` keywords for the outputs above. Any other keyword is listed as
ignored at the start and the end of the run.

If VASP used the tetrahedron method (`ISMEAR = -5`), gpucohp integrates with
the same tetrahedra, as LOBSTER does; otherwise with Gaussian smearing.

`examples/tib2` is a complete run (VASP, gpucohp, optionally LOBSTER) with
reference outputs. `gpucohp --help` lists the options.

## Basis sets

gpucohp ships two open basis sets, rebuilt from the published tables by the
scripts in `tools/` (sources in `third_party/`):

* **Koga** (default), all elements H–Lr: T. Koga, K. Kanayama, S. Watanabe,
  A. J. Thakkar, *Int. J. Quantum Chem.* **71**, 491 (1999); T. Koga,
  K. Kanayama, T. Watanabe, T. Imai, A. J. Thakkar, *Theor. Chem. Acc.* **104**,
  411 (2000).
* **Bunge**, He–Xe (`basisSet bunge`): C. F. Bunge, J. A. Barrientos,
  A. V. Bunge, *At. Data Nucl. Data Tables* **53**, 113 (1993). Identical to
  LOBSTER's Bunge set.

LOBSTER's default, **pbeVaspFit2015**, adds fitted polarisation functions whose
parameters belong to LOBSTER and are not published. With a LOBSTER licence
you can use it: run LOBSTER once with `writeBasisFunctions` in `lobsterin`, then

```bash
gpucohp-basis import path/to/that/lobster/run
```

adds its functions to your personal library (`~/.gpucohp/basis`), which gpucohp
then uses. Without it, a `lobsterin` asking for pbeVaspFit2015 runs with Koga,
and the run says so at its start and end.

For compounds whose bonding the occupied atomic shells describe (the
diborides, C, GaAs below), Koga and pbeVaspFit2015 give ICOHP within 0.6 %.
For transition metals that need 4p functions they do not: in hcp Ti the
Ti–Ti ICOHP is −1.79 eV with pbeVaspFit2015 and −0.71 eV with Koga, in both
codes (`benchmarks/figures/ti_basis_koga_vs_pbevaspfit2015.png`).

## Agreement with LOBSTER 5.1.1

Same WAVECAR, same `lobsterin`, same basis set; pairs matched by atoms and
cell translation.

| system | bonds | max \|ΔICOHP\| | charges, spilling | LOBSTER | gpucohp (RTX 2080 Ti) |
|---|---|---|---|---|---|
| HEB2 grain boundary, 48 atoms, 5 metals | 801 | 0.005 eV (0.3 %) | identical | 6 h 26 min, 12 cores | 1 min 41 s |
| TiB2, 2700 k-points | 48 | 0.05 eV (0.7 %, B–B σ), else < 0.003 eV | identical | 4 min, 24 cores | 48 s |
| C60 in a box | 90 | 0.003 eV | identical | 9 min, 24 cores | 10 s |
| carbon nanotube, 32 atoms | 48 | 0.005 eV | identical | 97 s, 24 cores | 11 s |
| diamond, GaAs (tetrahedra) | 1 each | < 0.001 eV; curves to 5e-4 | identical | 2–3 s | 7–9 s |
| hcp Ti, FeNi3 (spin-polarised) | 1 each | 2–9 % | ≤ 0.02 e | 4–6 s | 7–9 s |

The larger differences (the TiB2 B–B bond, Ti, FeNi3) sit on k-points where
the retained bands leave one basis direction uncovered; LOBSTER flags them as
"could not be orthonormalized", gpucohp as "rank-deficient projection". The
Hamiltonian there depends on how that direction is completed, in either code;
at every other k-point the two codes' Hamiltonians agree to 2e-7 eV.
Small cells are dominated by gpucohp's start-up (about 6 s).
`benchmarks/` holds the inputs, both codes' outputs, comparison scripts and
figures (`maintz2013/`: the examples of the 2013 paper; `lobster_examples_shipped/`:
the examples shipped with LOBSTER, unchanged).

## Citing

Please cite gpucohp (`CITATION.cff`), the LOBSTER method papers above, and the
basis-set tables you used.

## Licence

TODO
