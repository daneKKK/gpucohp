#!/bin/sh
#SBATCH -p mpi
#SBATCH -J lobship
#SBATCH -t 0-04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=0
#SBATCH --array=0-5%3
#SBATCH -o slurm-%A_%a.out

# ── RUN META (dashboard) ─────────────────────────────────────────────────────
# meta_id:   2026-09-26_lkdm_gpucohp_shipped-examples-cpu
# project:   GPU LOBSTER (проекция WAVECAR на атомные орбитали)
# what:      Примеры дистрибутива LOBSTER 5.1.1 БЕЗ изменений (ISMEAR=-5 где задан,
#            исходные lobsterin, pbeVaspFit2015 с 4p для Ti/FeNi3, POTCAR Ti обычный):
#            VASP 6.6.0 + LOBSTER 5.1.1 (+writeBasisFunctions) — эталон для
#            тетраэдрного интегрирования в gpucohp и для прогона gpucohp с
#            пользовательским дампом pbeVaspFit2015.
# prompt:    «Can we add tetrahedron interation? Can we use pbeVaspFit2015 in any shape
#            or form as dependency?»
# eta:       2026-09-26 вечер (~30 мин, C60 самый долгий)
# eta_basis: те же системы с ISMEAR=0 сегодня: VASP 0.2-6 мин, LOBSTER 2 с - 9 мин
# check:     ssh lkdm2_claude 'grep -h "spilling\|finished" /beegfs/home/dalekseev/lobster_gpu/shipped/*/cpu/lobsterout'
# record:    D:\work\dashboard\runs\2026-09-26_lkdm_gpucohp_shipped-examples-cpu.json
# ─────────────────────────────────────────────────────────────────────────────
module purge
module load compilers/intel-2020
module load mpi/impi-5.0.3
ulimit -s unlimited
SYS=(diamond gaas ti cnt c60 feni3); s=${SYS[$SLURM_ARRAY_TASK_ID]}
cd /beegfs/home/dalekseev/lobster_gpu/shipped/$s
/usr/bin/time -v -o vasp_time.txt srun --mpi=pmi2 -n 24 /home/pmaslov/vasp.6.6.0/bin/vasp_std > vasp.out 2>&1
grep "F=" OSZICAR | tail -1
mkdir -p cpu; cd cpu
for f in WAVECAR OUTCAR POTCAR POSCAR CONTCAR KPOINTS IBZKPT vasprun.xml; do [ -e $f ] || ln -s ../$f .; done
cp ../lobsterin .; echo writeBasisFunctions >> lobsterin
export OMP_NUM_THREADS=24
/usr/bin/time -v -o time.txt /beegfs/home/dalekseev/lobster_gpu/bin/lobster-5.1.1 > lobster.out 2>&1
grep -E "spilling|finished|tetrahedron|orthonormalized" lobsterout
