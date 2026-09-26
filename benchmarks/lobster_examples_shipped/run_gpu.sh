#!/bin/sh
#SBATCH --partition=gpu_devel
#SBATCH --job-name=gpucohp_ship
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=01:00:00
#SBATCH -o slurm-%j.out
# ── RUN META (dashboard) ─────────────────────────────────────────────────────
# meta_id:   2026-09-26_zhores1_gpucohp_shipped-examples-gpu
# project:   GPU LOBSTER (проекция WAVECAR на атомные орбитали)
# what:      gpucohp на неизменённых примерах LOBSTER: тетраэдры (с/без поправок
#            Блёхля) для алмаза, GaAs, Ti; Ti и FeNi3 с pbeVaspFit2015 из
#            пользовательской библиотеки (импорт дампов своих прогонов LOBSTER)
#            и с открытым Koga.
# prompt:    «Can we add tetrahedron interation? Can we use pbeVaspFit2015 ...»
# eta:       2026-09-26 (~5 мин)
# eta_basis: те же системы сегодня: 7-10 с на запуск
# check:     ssh zhores1 "cat ~/lobster_gpu/shipped/slurm-*.out"
# record:    D:\work\dashboard\runs\2026-09-26_zhores1_gpucohp_shipped-examples-gpu.json
# ─────────────────────────────────────────────────────────────────────────────
T=$HOME/lobster_gpu/shipped
export PYTHONPATH=$T/src GPUCOHP_BASIS_DIR=$T/userlib
run () {  # sys variant lobsterin-sed extra-args
  d=$T/$1/gpu_$2; mkdir -p $d; cd $d
  for f in WAVECAR POTCAR POSCAR CONTCAR vasprun.xml; do ln -sf ../$f .; done
  sed "$3" ../lobsterin > lobsterin
  python -m gpucohp . --device cuda $4 > run1.log 2>&1
  /usr/bin/time -v -o time2.txt python -m gpucohp . --device cuda $4 > run2.log 2>&1
  echo "== $1 $2"; grep -E "spill|pDOS|recovered|WARNING: at|ERROR" lobsterout
}
for s in diamond gaas; do
  run $s tet "s/x/x/" ""
  run $s tetlin "s/x/x/" "--tetra-scheme linear"
  run $s gauss "s/x/x/" "--integration gaussian"
done
run ti pbe_tet "s/x/x/" ""
run ti pbe_vaspocc "s/x/x/" "--occupations vasp"
run ti koga_tet "s/pbeVaspFit2015/koga/I; s/ 4p//g" ""
run feni3 pbe "s/x/x/" ""
run feni3 koga "s/pbeVaspFit2015/koga/I; s/ 4p//g" ""
run feni3 koga_vaspocc "s/pbeVaspFit2015/koga/I; s/ 4p//g" "--occupations vasp"
run ti koga_vaspocc "s/pbeVaspFit2015/koga/I; s/ 4p//g" "--occupations vasp"
