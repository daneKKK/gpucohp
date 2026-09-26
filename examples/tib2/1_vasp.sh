#!/bin/bash
#SBATCH --job-name=tib2_vasp
#SBATCH --ntasks=24
#SBATCH --output=%x-%j.out
# Step 1: VASP static whose WAVECAR the projection reads (about 700 MB).
# Adapt the two site-specific lines (environment, POTCAR) to your cluster.
# The INCAR has what both LOBSTER and gpucohp need: ISYM = -1, LWAVE = .TRUE.,
# NBANDS = 40 >= 18 basis functions (Ti 3s 3p 3d 4s = 10, B 2s 2p = 4 each).

module load vasp                                       # site-specific
[ -f POTCAR ] || cat $POTCAR_DIR/Ti_sv/POTCAR $POTCAR_DIR/B/POTCAR > POTCAR   # PBE Ti_sv, B (see POTCAR.spec)

srun vasp_std > vasp.out
