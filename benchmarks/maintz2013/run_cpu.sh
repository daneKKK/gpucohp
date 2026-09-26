#!/bin/sh
#SBATCH -p mpi
#SBATCH -J lobbench
#SBATCH -t 0-06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=0
#SBATCH --array=0-5%2
#SBATCH -o slurm-%A_%a.out

# ── RUN META (dashboard) ─────────────────────────────────────────────────────
# meta_id:   2026-09-26_lkdm_gpucohp_paper-benchmark-cpu
# project:   GPU LOBSTER (проекция WAVECAR на атомные орбитали)
# what:      Эталонные системы статьи Maintz 2013 / примеры дистрибутива LOBSTER
#            (алмаз, GaAs, hcp Ti, нанотрубка, C60, FeNi3 со спином): VASP 6.6.0
#            (24 ранга, ISMEAR=0) + LOBSTER 5.1.1 (24 потока) с тем же lobsterin,
#            что пойдёт в gpucohp (Bunge/Koga), плюс для Ti и FeNi3 — исходный
#            lobsterin дистрибутива (pbeVaspFit2015 с 4p). Время и память через time -v.
# prompt:    «Fix 1-2 and run and benchmark the systems on CPU and GPU port»
# eta:       2026-09-26 вечер (C60 самый долгий, ~30-60 мин)
# eta_basis: TiB2 на этих узлах: VASP 35 мин (2700 k), LOBSTER 11 мин
# check:     ssh lkdm2_claude 'grep -h "spilling\|finished" /beegfs/home/dalekseev/lobster_gpu/bench/*/cpu/lobsterout'
# record:    D:\work\dashboard\runs\2026-09-26_lkdm_gpucohp_paper-benchmark-cpu.json
# ─────────────────────────────────────────────────────────────────────────────
module purge
module load compilers/intel-2020
module load mpi/impi-5.0.3
ulimit -s unlimited
SYS=(diamond gaas ti cnt c60 feni3); s=${SYS[$SLURM_ARRAY_TASK_ID]}
T=/beegfs/home/dalekseev/lobster_gpu/bench/$s; cd $T
LOB=/beegfs/home/dalekseev/lobster_gpu/bin/lobster-5.1.1
echo "VASP start $(date)"
/usr/bin/time -v -o vasp_time.txt srun --mpi=pmi2 -n 24 /home/pmaslov/vasp.6.6.0/bin/vasp_std > vasp.out 2>&1
echo "VASP end $(date)"; grep "F=" OSZICAR | tail -1
export OMP_NUM_THREADS=24
run_lob () {   # $1 = subdir, $2 = lobsterin source
  mkdir -p $1; cd $1
  for f in WAVECAR OUTCAR POTCAR POSCAR CONTCAR KPOINTS IBZKPT vasprun.xml; do [ -e $f ] || ln -s ../$f .; done
  cp ../$2 lobsterin; echo writeBasisFunctions >> lobsterin
  /usr/bin/time -v -o time.txt $LOB > lobster.out 2>&1
  grep -E "spilling|finished" lobsterout; cd ..
}
run_lob cpu lobsterin
case $s in ti|feni3) run_lob cpu_shipped lobsterin.shipped;; esac
ls -la WAVECAR
