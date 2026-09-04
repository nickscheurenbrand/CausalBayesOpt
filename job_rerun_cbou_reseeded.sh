#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -N rerun_cbou_reseeded

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_erdos

# Rerun CBO-U (PARENT_SCALE, doubly-robust dr2) on Erdos20/50/100, linear +
# nonlinear, with the SAME per-replicate reseeding that CBO-U-Geo uses, so the
# runs line up seed-for-seed with the geometry runs. base_seed defaults to 71
# (== geometry_boundary_bash.py's --base_seed), so run r uses seed 71+r-1;
# 3 runs -> seeds 71,72,73, identical to the geometry Erdos runs. Output lands in
# results/<graph>_reseeded{,_nonlinear}/ -- a new dir, so existing pickles are
# untouched. ngpus=1 is kept because the deferred diffcbed/cdt import needs a
# parseable CUDA_VISIBLE_DEVICES (see job_erdos_50_100.sh); the CBO-U work itself
# is numpy/GPy on CPU. Erdos100 is the long pole; walltime padded.
nvidia-smi

LOG=$HOME/causal_bayes_opt/results/rerun_cbou_reseeded_pbs.log
mkdir -p $HOME/causal_bayes_opt/results

python3 rerun_cbou_reseeded.py \
    --graphs Erdos100 \
    --runs 3 \
    --variants linear,nonlinear \
    --n_observational 200 \
    --n_trials 30 \
    --n_int 2 \
    > "$LOG" 2>&1
