#!/usr/bin/env bash
# Full grid: 2 datasets x 3 Dirichlet concentrations x 4 aggregators x 3 seeds.
# Partitions depend only on (dataset, alpha); seeds and partitions are shared by all aggregators.
set -euo pipefail
PY=${PY:-python}
WORK=${WORK:-$HOME/fedids_work}
cd "$(dirname "$0")"
for ds in ${DATASETS:-cicids2017 ciciot2023}; do
  for seed in ${SEEDS:-0 1 2}; do
    for alpha in ${ALPHAS:-0.1 0.5 1.0}; do
      for agg in ${AGGS:-fedavg fedprox scaffold fedadam}; do
        out="$WORK/scores/$ds/a$alpha/$agg/s$seed.npz"
        [ -f "$out" ] && continue
        echo "$(date +%T) $ds alpha=$alpha agg=$agg seed=$seed"
        [ -f "$WORK/runs/$ds/a$alpha/$agg/s$seed/utility.json" ] || $PY run_experiment.py --work "$WORK" --dataset "$ds" --alpha "$alpha" --agg "$agg" --seed "$seed" \
          > "$WORK/logs/${ds}_a${alpha}_${agg}_s${seed}.log" 2>&1
        $PY attacks.py --work "$WORK" --dataset "$ds" --alpha "$alpha" --agg "$agg" --seed "$seed"
      done
    done
  done
done
