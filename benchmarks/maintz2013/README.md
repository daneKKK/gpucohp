# Benchmark: the test systems of the original LOBSTER papers

Diamond, GaAs, hcp Ti, a (C) nanotube and C60 are the examples of
S. Maintz, V. L. Deringer, A. L. Tchougréeff, R. Dronskowski,
*J. Comput. Chem.* **34**, 2557 (2013); FeNi3 (spin-polarised) is the sixth
example shipped with LOBSTER 5.1.1. The inputs here are the ones LOBSTER ships
in `VASP/<system>/`, with these changes:

* `ISMEAR = 0`, `SIGMA = 0.05` everywhere (the shipped diamond/GaAs/Ti inputs
  use the tetrahedron method, which LOBSTER then also uses; gpucohp integrates
  with Gaussians only, so both codes are compared with Gaussian integration);
  `NPAR = 4`, `LWAVE = .TRUE.`.
* POTCARs are not included (VASP licence); `POTCAR.spec` names the PBE
  datasets used (C, Ga_d, As, Ti_pv, Fe, Ni).
* `lobsterin` is the input given to **both** codes. For diamond, GaAs, the
  nanotube and C60 it is the shipped one (basis set Bunge). For Ti and FeNi3
  the shipped input asks for pbeVaspFit2015 4p functions, which are not
  published; there both codes use Koga (`Ti 3p 3d 4s`, `Fe/Ni 3d 4s`). The
  shipped inputs are kept as `lobsterin.shipped`.

VASP 6.6.0 and LOBSTER 5.1.1 ran on one 24-core node of lkdm2 (`run_cpu.sh`),
gpucohp on one RTX 2080 Ti of Zhores (`run_gpu.sh`, second of two runs timed so
CUDA start-up is not counted twice). `python compare.py diamond gaas ti cnt c60
feni3` rebuilds the table below from the `reference_*` outputs.

| system | bonds | max \|ΔICOHP\| (eV) | max \|ΔICOOP\| | charge spilling LOBSTER / gpucohp (%) | Δq max | LOBSTER (s) | gpucohp (s) |
|---|---|---|---|---|---|---|---|
| diamond | 1 | 0.0002 | 0.00006 | 0.98 / 0.98 | 0.00 | 2.4 | 7.2 |
| GaAs | 1 | 0.0010 | 0.00024 | 0.89 / 0.88 | 0.00 | 3.2 | 9.2 |
| hcp Ti | 1 | 0.020 (2.4 %) | 0.030 | 3.04 / 3.00 | 0.00 | 5.2 | 7.2 |
| nanotube, 32 C | 48 | 0.0049 | 0.00014 | 1.09 / 1.09 | 0.01 | 97 | 11.3 |
| C60 | 90 | 0.0033 | 0.00039 | 1.16 / 1.16 | 0.00 | 536 | 10.1 |
| FeNi3, spin | 1 × 2 spins | 0.018 | 0.006 | 2.90+5.06 / 2.89+5.12 | 0.01 | 6.2 | 9.1 |

gpucohp wall times include Python and CUDA start-up (about 6 s), which is why
the three smallest cells are slower than LOBSTER; for C60 the GPU is 53 times
faster, for the nanotube 9 times.

The Bunge basis that gpucohp builds from the published tables
(`tools/make_bunge_basis.py`) is identical, to every printed digit, to the
basis LOBSTER dumps with `writeBasisFunctions` for C, Ga and As.

**Against the 2013 paper.** Both codes reproduce the paper's qualitative
results; the numbers differ from the paper's because its prototype neglected
off-site overlaps and used an older basis and VASP 5.2. Example: C60, the
1.40 Å bonds are stronger than the 1.45 Å ones in the paper (−6.4 vs −5.9 eV)
and in both codes now (−9.72 vs −8.72 eV, identical in LOBSTER and gpucohp).

**Ti and FeNi3 are ill-conditioned.** In both metals the retained bands
(as many as there are basis functions) leave one local-orbital direction
uncovered at some k-points (Ti: 19 of 147 in the kz = 0 plane; FeNi3: 50 of
250 per spin). The band overlap matrix is then singular, and the Hamiltonian
at those k depends on how that direction is completed. gpucohp completes it
with the orthogonal complement (`--rank-deficient complete`, default) or
leaves it empty (`drop`); LOBSTER flags these k-points in FeNi3 ("could not
be orthonormalized") and silently completes them in Ti. The Ti–Ti and Fe–Ni
ICOHP differences above (2-13 %) come only from these k-points: the S, T and
C matrices of both codes agree to 1e-3 or better and the Hamiltonians to
2e-7 eV at every regular k-point. The shipped FeNi3 input with 4p functions
is affected even more (LOBSTER: 186 of 250 k-points). The same mechanism
explains the 1.6 % → 0.7 % B–B difference in primitive TiB2.
