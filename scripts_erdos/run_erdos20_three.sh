#!/bin/bash
# Runs CBO_algorithm + PARENT_SCALE + RANDOM_SCALE (parent_method "three")
# for Erdos20, 3 seeds each, both linear and nonlinear. Saves to results/Erdos20
# (nonlinear pickles carry the _nonlinear suffix).
# Resumable: each (run_num, kind) invocation is skipped if its three output
# pickles already exist, so a restart only computes what is missing.
# NOTE: large_graph_script.py does os.chdir("..") so this MUST be launched
# from the scripts_erdos/ directory.
set -euo pipefail

SEEDS=(71 11 89)

N_OBS=200
N_TRIALS=30
N_ANCHOR=35

RESULTS="$HOME/causal_bayes_opt/results/Erdos20"

# all three "three"-method pickles present for this run_num + suffix?
already_done () {
    local n="$1" ns="$2"
    [[ -f "$RESULTS/run${n}_cbo_results_200_2${ns}.pickle" \
       && -f "$RESULTS/run${n}_cbo_unknown_dr2_results_200_2${ns}.pickle" \
       && -f "$RESULTS/run${n}_cbo_results_random_200_2${ns}.pickle" ]]
}

run_num=0
for seed in "${SEEDS[@]}"; do
    run_num=$((run_num + 1))

    if already_done "$run_num" ""; then
        echo "=== SKIP Erdos20 linear seed=${seed} run_num=${run_num} (already done) ==="
    else
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
    fi

    if already_done "$run_num" "_nonlinear"; then
        echo "=== SKIP Erdos20 nonlinear seed=${seed} run_num=${run_num} (already done) ==="
    else
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
    fi
done
