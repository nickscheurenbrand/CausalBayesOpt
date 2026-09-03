#!/bin/bash
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=02:00:00
#PBS -N precompute_cbou_boundary

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_erdos

# Reconstructs the CBO-U intervention ranges (rebuild Erdos graph + resample D_O
# per run seed) and computes the boundary fraction per iteration, writing
# results/<graph>/cbou_boundary_200_2<ns>.pickle. CPU-only, lightweight, but uses
# the graph stack so it must run on a compute node, not the login node.
LOG=$HOME/causal_bayes_opt/results/Erdos50/cbou_boundary_precompute.log
python precompute_cbou_boundary.py > "$LOG" 2>&1
