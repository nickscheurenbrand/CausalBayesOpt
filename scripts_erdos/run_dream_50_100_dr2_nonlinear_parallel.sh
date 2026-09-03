#!/bin/bash
# Parallel runner for the SAME experiment as run_erdos_50_100_dr2_nonlinear_parallel.sh
# (CBO + PARENT_SCALE + RANDOM_SCALE, parent_method "three", NONLINEAR, DR2)
# but on the DREAM4 graphs Dream50 (InSilicoSize50-Ecoli1) and Dream100
# (InSilicoSize100-Ecoli1). Runs the 6 (seed x graph) tasks across NGPUS GPUs,
# one task per GPU, dispatching the next task to whichever GPU frees up first.
#
# NOTE: large_graph_script.py does os.chdir("..") so this MUST be launched
# from the scripts_erdos/ directory.
set -euo pipefail

GRAPHS=("Dream50" "Dream100")
SEEDS=(71 11 89)

N_OBS=200
N_TRIALS=30
N_ANCHOR=35

# GPUs to spread the work over, as space-separated physical device indices.
# Override via env, e.g. to use only GPU 0:
#   GPUS="0" bash run_dream_50_100_dr2_nonlinear_parallel.sh
# Default uses both GPUs.
read -r -a GPUS <<< "${GPUS:-0 1}"

# Build the task list: each entry is "run_num seed graph".
tasks=()
run_num=0
for seed in "${SEEDS[@]}"; do
    run_num=$((run_num + 1))
    for graph in "${GRAPHS[@]}"; do
        tasks+=("${run_num} ${seed} ${graph}")
    done
done

run_task() {
    local gpu="$1" run_num="$2" seed="$3" graph="$4"
    local log="../results/${graph}/run${run_num}_${graph}_nonlinear_gpu${gpu}.log"
    echo "[GPU ${gpu}] START graph=${graph} seed=${seed} run_num=${run_num} -> ${log}"
    # cdt requires a parseable CUDA_VISIBLE_DEVICES (it ast.literal_eval's it),
    # so a single integer device index is exactly what it wants.
    CUDA_VISIBLE_DEVICES="${gpu}" python3 large_graph_script.py \
        --seeds_replicate "${seed}" \
        --n_observational "${N_OBS}" \
        --n_trials "${N_TRIALS}" \
        --n_anchor_points "${N_ANCHOR}" \
        --run_num "${run_num}" \
        --noiseless \
        --graph_type "${graph}" \
        --parent_method "three" \
        --nonlinear \
        > "${log}" 2>&1
    echo "[GPU ${gpu}] DONE  graph=${graph} seed=${seed} run_num=${run_num}"
}

declare -A slot_pid
for g in "${GPUS[@]}"; do slot_pid[$g]=""; done

for task in "${tasks[@]}"; do
    launched=""
    while [[ -z "${launched}" ]]; do
        for g in "${GPUS[@]}"; do
            if [[ -z "${slot_pid[$g]}" ]] || ! kill -0 "${slot_pid[$g]}" 2>/dev/null; then
                read -r rn sd gr <<< "${task}"
                run_task "${g}" "${rn}" "${sd}" "${gr}" &
                slot_pid[$g]=$!
                launched="yes"
                break
            fi
        done
        [[ -z "${launched}" ]] && sleep 2
    done
done

wait
echo "All tasks complete."
