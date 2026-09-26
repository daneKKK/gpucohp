# Benchmark: LOBSTER's own examples, unchanged

The six VASP examples shipped with LOBSTER 5.1.1 (`VASP/<system>/`), with
their INCAR, KPOINTS, POSCAR and lobsterin exactly as distributed; only the
POTCARs had to be chosen (`POTCAR.spec`: C, Ga_d/As, Ti, Fe/Ni). Diamond,
GaAs and Ti use `ISMEAR = -5`, so both codes integrate with tetrahedra; the
nanotube, C60 and FeNi3 use Gaussian integration. VASP 6.6.0 and LOBSTER
5.1.1 on one lkdm2 node (`run_cpu.sh`), gpucohp on an RTX 2080 Ti
(`run_gpu.sh`). `python compare.py .` rebuilds the lines below.

## Tetrahedron integration

| system | gpucohp variant | ICOHP gpucohp / LOBSTER (eV) | COHP curve max \|Δ\| |
|---|---|---|---|
| diamond | `--tetra-scheme lobster` (default) | −9.5986 / −9.5986 | 0.0005 of 2.34 |
| diamond | `--tetra-scheme linear` | −9.5986 / −9.5986 | 0.042 |
| diamond | `--integration gaussian` | −9.5974 / −9.5986 | (different method) |
| GaAs | default | −4.3288 / −4.3288 | 0.0004 of 2.05 |
| GaAs | `linear` | −4.3288 / −4.3288 | 0.061 |

LOBSTER multiplies the occupied fraction of each tetrahedron by the *mean*
of its four corner values, rather than interpolating the quantity linearly
inside it; with that scheme gpucohp reproduces LOBSTER's curves to 5e-4 and
its integrals exactly. Both codes give the same value at E_F either way.

## pbeVaspFit2015: with and without

Ti and FeNi3 ask for pbeVaspFit2015 including 4p functions. gpucohp does not
ship that set; a licensed LOBSTER user imports it from their own runs
(`gpucohp-basis import <dir>`, see the main README). Without it gpucohp uses
the open Koga set, which has no 4p for these atoms. LOBSTER was run both ways
too, so the effect of the basis is separated from the difference between codes:

| system | basis | Ti–Ti or Fe–Ni ICOHP (eV) LOBSTER / gpucohp | charge spilling (%) LOBSTER / gpucohp | Mulliken q(Fe) |
|---|---|---|---|---|
| hcp Ti | pbeVaspFit2015 4s 3d 4p | −1.794 / −1.835 | 3.21 / 3.13 | – |
| hcp Ti | Koga 4s 3d | −0.706 / −0.714 | 7.92 / 8.34 | – |
| FeNi3 ↑ / ↓ | pbeVaspFit2015 3d 4s 4p | −0.98, −1.04 / −0.90, −1.13 | 1.45, 1.74 / 1.42, 1.89 | −0.28 / −0.27 |
| FeNi3 ↑ / ↓ | Koga 3d 4s | −0.14, −0.32 / −0.16, −0.35 | 2.96, 4.71 / 2.88, 5.12 | +0.09 / +0.08 |

For these metals the basis matters far more than the code: without the 4p
polarisation functions the Ti–Ti ICOHP falls to 40 % of its value, spilling
more than doubles, and the Mulliken charge of Fe changes sign. The Koga set is
adequate where the semicore and valence shells describe the bonding (the
diborides, diamond, GaAs, C60: see `../maintz2013` and the main README), not
for transition metals that need 4p.

The remaining code-to-code differences in this table (2-9 %) sit on
rank-deficient k-points, which LOBSTER itself reports: Ti 61 of 147, FeNi3 186
of 250 with pbeVaspFit2015 ("could not be orthonormalized"). There the
Hamiltonian depends on how the uncovered basis direction is completed, in
either code (see the main README). Charge spilling in the metals differs by
up to 0.4 % because LOBSTER weights it with occupations we have not
reproduced exactly; `--occupations lobster` (weights of gpucohp's own
integration) changes it only slightly and is not the default.
