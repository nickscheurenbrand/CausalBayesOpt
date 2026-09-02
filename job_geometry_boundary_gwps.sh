#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=32:00:00
#PBS -N geometry_boundary_gwps

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_geometry

# GEOMETRY_SCALE (appendix E) on the GWPS gene network (60-node DAG subgraph,
# linear G_hat SEM at weight_scale=3 -- the diagnostic's stable/usable scale),
# baseline only, 3 replicate seeds each -- 3 runs. n_int=1 to match the existing
# boundary_tracking_gwps runs. The gwps knobs (max_nodes=60, weight_scale=3,
# top_k=8) use the geometry script defaults, which already match, so the output
# tag is gwps_n60_ws3, saved into
# results/boundary_tracking_gwps_geometry/gwps_n60_ws3/. Assumes
# data/gwps_direct_edges.csv exists. ngpus=1 is load-bearing: the TabPFN prior
# mean is evaluated on every acquisition evaluation, so this loop is genuinely
# GPU-bound.
nvidia-smi

LOG=$HOME/causal_bayes_opt/results/boundary_tracking_gwps_geometry/geometry_gwps_pbs.log
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_gwps_geometry

python geometry_boundary_bash.py \
    --graphs gwps \
    --runs 3 \
    --variants baseline \
    --prior tabpfn \
    --acquisition EI \
    --n_observational 200 \
    --n_trials 30 \
    --n_int 1 \
    --device cuda \
    > "$LOG" 2>&1
