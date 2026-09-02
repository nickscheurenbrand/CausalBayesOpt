#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=32:00:00
#PBS -N geometry_boundary_dream

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_geometry

# GEOMETRY_SCALE (appendix E) on the DREAM Size50/Size100 E. coli nets, baseline
# only, 3 replicate seeds each -- 3 runs per graph. Same as
# job_geometry_boundary.sh but with the dream graphs instead of Erdos; the DREAM
# SEM is an inherently nonlinear JAX model, so the geometry script builds it
# nonlinear automatically (no --nonlinear flag). Custom targets: Size50 -> 19,
# Size100 -> 41. Because the two graphs use different targets, the driver is
# called once per graph (--target applies to all --graphs in one call). With a
# custom --target the dream tag gets a _t<target> suffix, so results land in
# results/boundary_tracking_dream_geometry/{Size50-Ecoli1_t19,Size100-Ecoli1_t41}/.
# ngpus=1 is load-bearing here: the TabPFN prior mean is evaluated on every
# acquisition evaluation, so this is the one loop in the repo that is genuinely
# GPU-bound. Size100 is the long pole; walltime padded.
nvidia-smi

LOG=$HOME/causal_bayes_opt/results/boundary_tracking_dream_geometry/geometry_dream_pbs.log
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_dream_geometry

{
python geometry_boundary_bash.py \
    --graphs Size50-Ecoli1 \
    --target 19 \
    --runs 3 \
    --variants baseline \
    --prior tabpfn \
    --acquisition EI \
    --n_observational 200 \
    --n_trials 30 \
    --n_int 2 \
    --device cuda

python geometry_boundary_bash.py \
    --graphs Size100-Ecoli1 \
    --target 41 \
    --runs 3 \
    --variants baseline \
    --prior tabpfn \
    --acquisition EI \
    --n_observational 200 \
    --n_trials 30 \
    --n_int 2 \
    --device cuda
} > "$LOG" 2>&1
