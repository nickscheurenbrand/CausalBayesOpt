#!/bin/bash
# Runs the THREE-algorithm GWPS comparison (PARENT_SCALE dr2 + CBO true-parents
# + RANDOM_SCALE) via gwps_three_script.py at size 40 (max_nodes=40), for 3
# replicate seeds. Spreads the runs across the GPUs listed in GPUS (one task per
# GPU, dispatching the next task to whichever GPU frees up first).
#
# GWPS is linear-Gaussian, so no --nonlinear flag is passed.
#
# NOTE: gwps_three_script.py does os.chdir("..") so this MUST be launched from
# the scripts_gwps/ directory.
#
# Usage:
#   bash run_gwps_40_three_parallel.sh          # both GPUs (0 and 1)
#   GPUS="0" bash run_gwps_40_three_parallel.sh # only GPU 0
set -euo pipefail

MAX_NODES=40
SEEDS=(71 11 89)

N_OBS=200
N_TRIALS=30
N_INT=1

# GPUs to spread the work over, as space-separated physical device indices.
read -r -a GPUS <<< "${GPUS:-0 1}"

# Build the task list: each entry is "run_num seed".
tasks=()
run_num=0
for seed in "${SEEDS[@]}"; do
    run_num=$((run_num + 1))
    tasks+=("${run_num} ${seed}")
done

run_task() {
    local gpu="$1" run_num="$2" seed="$3"
    local log="../results/Gwps${MAX_NODES}/run${run_num}_gwps${MAX_NODES}_gpu${gpu}.log"
    mkdir -p "../results/Gwps${MAX_NODES}"
    echo "[GPU ${gpu}] START gwps n=${MAX_NODES} seed=${seed} run_num=${run_num} -> ${log}"
    # cdt requires a parseable CUDA_VISIBLE_DEVICES (it ast.literal_eval's it),
    # so a single integer device index is exactly what it wants.
    CUDA_VISIBLE_DEVICES="${gpu}" python3 gwps_three_script.py \
        --max_nodes "${MAX_NODES}" \
        --seeds_replicate "${seed}" \
        --n_observational "${N_OBS}" \
        --n_trials "${N_TRIALS}" \
        --n_int "${N_INT}" \
        --run_num "${run_num}" \
        --noiseless \
        > "${log}" 2>&1
    echo "[GPU ${gpu}] DONE  gwps n=${MAX_NODES} seed=${seed} run_num=${run_num}"
}

declare -A slot_pid
for g in "${GPUS[@]}"; do slot_pid[$g]=""; done

for task in "${tasks[@]}"; do
    launched=""
    while [[ -z "${launched}" ]]; do
        for g in "${GPUS[@]}"; do
            if [[ -z "${slot_pid[$g]}" ]] || ! kill -0 "${slot_pid[$g]}" 2>/dev/null; then
                read -r rn sd <<< "${task}"
                run_task "${g}" "${rn}" "${sd}" &
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
