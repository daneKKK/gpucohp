#!/bin/bash
#SBATCH --job-name=tib2_gpucohp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
# Step 2: the bonding analysis on one GPU (about 1 min on an RTX 2080 Ti).
# Needs WAVECAR, POTCAR, CONTCAR (or POSCAR), vasprun.xml and lobsterin here.

gpucohp . --device cuda
python compare.py reference/lobster_ICOHPLIST.lobster ICOHPLIST.lobster
