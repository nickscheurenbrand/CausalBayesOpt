#!/bin/bash
# Runs CBO_algorithm + PARENT_SCALE + RANDOM_SCALE (parent_method "three")
# for Erdos20, 3 seeds each, both linear and nonlinear. Saves to results/Erdos20
# (nonlinear pickles carry the _nonlinear suffix).
# NOTE: large_graph_script.py does os.chdir("..") so this MUST be launched
# from the scripts_erdos/ directory.
set -euo pipefail

SEEDS=(71 11 89)

N_OBS=200
N_TRIALS=30
N_ANCHOR=35

run_num=0
for seed in "${SEEDS[@]}"; do
    run_num=$((run_num + 1))
    echo "=== Erdos20 linear seed=${seed} run_num=${run_num} ==="
    python3 large_graph_script.py \
        --seeds_replicate "${seed}" \
        --n_observational "${N_OBS}" \
        --n_trials "${N_TRIALS}" \
        --n_anchor_points "${N_ANCHOR}" \
        --run_num "${run_num}" \
        --noiseless \
        --graph_type "Erdos20" \
        --parent_method "three"

    echo "=== Erdos20 nonlinear seed=${seed} run_num=${run_num} ==="
    python3 large_graph_script.py \
        --seeds_replicate "${seed}" \
        --n_observational "${N_OBS}" \
        --n_trials "${N_TRIALS}" \
        --n_anchor_points "${N_ANCHOR}" \
        --run_num "${run_num}" \
        --noiseless \
        --graph_type "Erdos20" \
        --parent_method "three" \
        --nonlinear
done