#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=16:00:00
#PBS -N geometry_boundary_erdos20

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_geometry

# GEOMETRY_SCALE (appendix E, CBO-U-Geo) on Erdos20, baseline only, 3 replicate
# seeds each -- both linear and nonlinear in one job. The driver is called twice:
# once plain (linear) and once with --nonlinear, which builds the Erdos SEM
# nonlinear and saves to a <graph>_nonlinear tag. Results land in
# results/boundary_tracking_erdos_geometry/{Erdos20,Erdos20_nonlinear}/.
# ngpus=1 is load-bearing here: the TabPFN prior mean is evaluated on every
# acquisition evaluation, so this is the one loop in the repo that is genuinely
# GPU-bound.
nvidia-smi

LOG=$HOME/causal_bayes_opt/results/boundary_tracking_erdos_geometry/geometry_erdos20_pbs.log
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_erdos_geometry

{
python geometry_boundary_bash.py \
    --graphs Erdos20 \
    --runs 3 \
    --variants baseline \
    --prior tabpfn \
    --acquisition EI \
    --n_observational 200 \
    --n_trials 30 \
    --n_int 2 \
    --device cuda

python geometry_boundary_bash.py \
    --graphs Erdos20 \
    --runs 3 \
    --variants baseline \
    --prior tabpfn \
    --acquisition EI \
    --n_observational 200 \
    --n_trials 30 \
    --n_int 2 \
    --device cuda \
    --nonlinear
} > "$LOG" 2>&1