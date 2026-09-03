#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=16:00:00
#PBS -N gwps_cbou_reseeded

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_gwps

# Reseeded CBO-U (PARENT_SCALE dr2) baseline for gwps40, aligned run-for-run with
# the gwps40 CBO-U-Geo runs. Unlike the erdos case this is NOT a reseeding-bug
# fix -- the gwps SEM is already reseeded correctly via GwpsGraph(seed=...) -- it
# just re-runs at the geometry-matching config. The existing results/Gwps40
# baseline used seeds 71,11,89 and n_int=1, but the gwps40 geometry runs use
# seeds 71,72,73 and n_int=2 (results/boundary_tracking_gwps_geometry/gwps_n40_ws3/
# run{1,2,3}_..._200_2.pickle), so both axes are realigned here.
#
# gwps60 is deliberately NOT re-run: results/gwps_three already holds the 3-run
# CBO-U baseline at seeds 71,72,73 / n_int=1, matching the gwps60 geometry
# (gwps_n60_ws3/..._200_1). Only gwps40 was off.
#
# Output goes to results/Gwps40_reseeded/ (a NEW dir, notebook naming convention
# via --out_suffix) so the existing results/Gwps40 pickles are untouched.
# CPU-only: PARENT_SCALE is numpy/GPy and gwps_cbo_script.py now sets a parseable
# CUDA_VISIBLE_DEVICES for the cdt import when no GPU is present.

SEEDS=(71 72 73)          # run r uses seed 71+r-1, == geometry base_seed default
MAX_NODES=40
WS=3
TOPK=8
N_OBS=200
N_TRIALS=30
N_INT=2                   # matches gwps40 geometry (..._200_2)

LOG=$HOME/causal_bayes_opt/results/gwps_cbou_reseeded_pbs.log
mkdir -p $HOME/causal_bayes_opt/results

{
run_num=0
for seed in "${SEEDS[@]}"; do
    run_num=$((run_num + 1))
    echo "===== gwps n=${MAX_NODES} seed=${seed} run_num=${run_num} ====="
    python3 gwps_cbo_script.py \
        --max_nodes "${MAX_NODES}" \
        --top_k_parents "${TOPK}" \
        --weight_scale "${WS}" \
        --seeds_replicate "${seed}" \
        --n_observational "${N_OBS}" \
        --n_trials "${N_TRIALS}" \
        --n_int "${N_INT}" \
        --run_num "${run_num}" \
        --out_suffix _reseeded \
        --noiseless
done
} > "$LOG" 2>&1
