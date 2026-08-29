#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -N geometry_boundary_erdos

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_geometry

mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_erdos_geometry

# GEOMETRY_SCALE (appendix E) on Erdos50 + Erdos100, baseline only, 5 replicate
# seeds each -- 10 runs. Baseline means the parent set is inferred by the
# doubly-robust bootstrap rather than injected, so each run also pays that
# parent-ID step. Unlike the other boundary jobs, ngpus=1 is
# load-bearing rather than a cdt-import workaround: the TabPFN prior mean is
# evaluated on every acquisition evaluation, so this is the one loop in the
# repo that is genuinely GPU-bound. Budget generously; 48h is deliberate.
nvidia-smi

LOG=$HOME/causal_bayes_opt/results/boundary_tracking_erdos_geometry/geometry_pbs.log

python geometry_boundary_bash.py \
    --graphs Erdos50,Erdos100 \
    --runs 5 \
    --variants baseline \
    --prior tabpfn \
    --acquisition EI \
    --n_observational 200 \
    --n_trials 30 \
    --n_int 2 \
    --device cuda \
    > "$LOG" 2>&1
