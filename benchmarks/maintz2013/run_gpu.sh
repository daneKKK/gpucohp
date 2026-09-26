#!/bin/sh
#SBATCH --partition=gpu_devel
#SBATCH --job-name=gpucohp_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24gb
#SBATCH --time=01:00:00
#SBATCH -o slurm-%j.out

# ── RUN META (dashboard) ─────────────────────────────────────────────────────
# meta_id:   2026-09-26_zhores1_gpucohp_paper-benchmark-gpu
# project:   GPU LOBSTER (проекция WAVECAR на атомные орбитали)
# what:      gpucohp на WAVECAR'ах эталонных систем (алмаз, GaAs, hcp Ti, нанотрубка,
#            C60, FeNi3), посчитанных на lkdm2; тот же lobsterin, что у LOBSTER 5.1.1.
#            Каждая система запускается дважды: первый прогон прогревает CUDA,
#            во втором меряется время.
# prompt:    «Fix 1-2 and run and benchmark the systems on CPU and GPU port»
# eta:       2026-09-26 (~10 мин после старта)
# eta_basis: TiB2 (2925 k, 18 ф-ций) 45 с на 2080 Ti
# check:     ssh zhores1 'grep -h "spilling\|finished" ~/lobster_gpu/bench/*/gpu/lobsterout'
# record:    D:\work\dashboard\runs\2026-09-26_zhores1_gpucohp_paper-benchmark-gpu.json
# ─────────────────────────────────────────────────────────────────────────────
T=~/lobster_gpu/bench
export PYTHONPATH=$T/src
nvidia-smi -L
for s in diamond gaas ti cnt c60 feni3; do
  [ -f $T/$s/WAVECAR ] || continue
  mkdir -p $T/$s/gpu; cd $T/$s/gpu
  for f in WAVECAR POTCAR POSCAR CONTCAR vasprun.xml lobsterin; do [ -e $f ] || ln -s ../$f .; done
  for rep in 1 2; do
    /usr/bin/time -v -o time$rep.txt python -m gpucohp . --device cuda > run$rep.log 2>&1
  done
  echo "== $s"; grep -E "spilling|projection finished|finished in" lobsterout
done
