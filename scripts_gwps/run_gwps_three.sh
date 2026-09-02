#!/bin/bash
# Runs CBO (true parents) + PARENT_SCALE (dr2) + RANDOM_SCALE on the gwps graph,
# 3 seeds. gwps analogue of run_erdos_50_100_dr2.sh.
# NOTE: gwps_three_script.py does os.chdir("..") so this MUST be launched from
# the scripts_gwps/ directory.
set -euo pipefail

SEEDS=(71 72 73)

N_OBS=200
N_TRIALS=30
N_INT=1
WS=3
MAXN=60
TOPK=8

run_num=0
for seed in "${SEEDS[@]}"; do
    run_num=$((run_num + 1))
    echo "=== gwps seed=${seed} run_num=${run_num} ==="
    python3 gwps_three_script.py \
        --seeds_replicate "${seed}" \
        --n_observational "${N_OBS}" \
        --n_trials "${N_TRIALS}" \
        --n_int "${N_INT}" \
        --weight_scale "${WS}" \
        --max_nodes "${MAXN}" \
        --top_k_parents "${TOPK}" \
        --run_num "${run_num}" \
        --noiseless
done