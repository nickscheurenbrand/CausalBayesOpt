#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N erdos_50_100

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/Erdos50
mkdir -p $HOME/causal_bayes_opt/results/Erdos100

# CBO_algorithm + PARENT_SCALE + RANDOM_SCALE (parent_method "three") on
# Erdos50 + Erdos100, 3 replicate seeds each -> 6 iterations. PARENT_SCALE runs
# the doubly-robust bootstrap parent-ID step, which is the expensive part;
# walltime is budgeted generously to cover both graphs. ngpus=1 is kept because
# cdt fails to import without a parseable CUDA_VISIBLE_DEVICES (see
# job_oracle_boundary.sh).
nvidia-smi

LOG=$HOME/causal_bayes_opt/results/Erdos50/erdos_50_100_pbs.log

bash run_erdos_50_100_dr2.sh > "$LOG" 2>&1
