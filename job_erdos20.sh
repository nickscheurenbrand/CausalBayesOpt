#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=16:00:00
#PBS -N erdos20

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/Erdos20

# CBO_algorithm + PARENT_SCALE + RANDOM_SCALE (parent_method "three") on Erdos20,
# 3 replicate seeds each, both linear and nonlinear -> 6 iterations. PARENT_SCALE
# runs the doubly-robust bootstrap parent-ID step, which is the expensive part.
# ngpus=1 is kept because cdt fails to import without a parseable
# CUDA_VISIBLE_DEVICES (see job_oracle_boundary.sh).
nvidia-smi

LOG=$HOME/causal_bayes_opt/results/Erdos20/erdos20_pbs.log

bash run_erdos20_three.sh > "$LOG" 2>&1