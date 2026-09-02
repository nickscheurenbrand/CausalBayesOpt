#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=32:00:00
#PBS -N geometry_boundary_erdos_nonlinear

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_geometry

# GEOMETRY_SCALE (appendix E) on NONLINEAR Erdos50 + Erdos100, baseline only,
# 3 replicate seeds each -- 3 runs per graph. Same as job_geometry_boundary.sh
# but with --nonlinear, which builds the Erdos SEMs nonlinear and saves into
# results/boundary_tracking_erdos_geometry/{Erdos50_nonlinear,Erdos100_nonlinear}/
# (pickles carry the _nonlinear suffix). ngpus=1 is load-bearing here: the
# TabPFN prior mean is evaluated on every acquisition evaluation, so this is the
# one loop in the repo that is genuinely GPU-bound.
nvidia-smi

LOG=$HOME/causal_bayes_opt/results/boundary_tracking_erdos_geometry/geometry_nonlinear_pbs.log
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_erdos_geometry

python geometry_boundary_bash.py \
    --graphs Erdos50,Erdos100 \
    --runs 3 \
    --variants baseline \
    --prior tabpfn \
    --acquisition EI \
    --n_observational 200 \
    --n_trials 30 \
    --n_int 2 \
    --device cuda \
    --nonlinear \
    > "$LOG" 2>&1
