#!/bin/bash
#SBATCH --job-name=tib2_lobster
#SBATCH --cpus-per-task=24
#SBATCH --output=%x-%j.out
# Optional: the same analysis with LOBSTER 5.1.1 (needs your own LOBSTER licence,
# and OUTCAR and KPOINTS next to the other files). Runs in lobster/ so the two
# sets of outputs stay apart.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p lobster && cd lobster
for f in WAVECAR OUTCAR POTCAR POSCAR CONTCAR KPOINTS vasprun.xml lobsterin; do [ -e $f ] || ln -s ../$f .; done
lobster
python ../compare.py ICOHPLIST.lobster ../ICOHPLIST.lobster
