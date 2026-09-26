# Example: TiB2

Primitive TiB2 (3 atoms, 15×15×12 k-mesh without symmetry, 2700 k-points).

```bash
sbatch 1_vasp.sh          # VASP static -> WAVECAR
sbatch 2_gpucohp.sh       # gpucohp on one GPU, then compares with the reference
sbatch 3_lobster_optional.sh   # the same with LOBSTER, if you have it
```

Adapt the environment and POTCAR lines of `1_vasp.sh` to your cluster; the
POTCARs are PBE `Ti_sv` and `B` (`POTCAR.spec`). The `lobsterin` uses the open
Koga basis, so both codes run it as it is.

`reference/` holds LOBSTER 5.1.1's output for this input and VASP setup.
`compare.py` matches bonds by atoms and cell translation and prints the
largest difference; expect about 0.05 eV on the short B–B bond (it sits on
k-points where the projection is rank-deficient, see the main README) and
1e-3 eV or less elsewhere, identical charges and charge spilling (1.39 %).
Your VASP build and machine will shift the numbers slightly: the same LOBSTER
on two different machines differs by up to 0.008 eV here.
