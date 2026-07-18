#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=14:00:00
#PBS -N check_parent_recovery_erdos100

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/cold_start_check

# Checks, for Erdos100, whether the initial doubly-robust bootstrap draws
# include the TRUE parents (85, 98) "in more than one case": across 20 bootstrap
# draws per seed AND across two data seeds (40 draws total). The marginal
# summary at the end reports, per true parent, in how many draws it appears.
#
# Cost: the individual selector runs a per-variable doubly-robust + t-test over
# all 99 candidates per bootstrap (~13-17 min/bootstrap for Erdos100), so 40
# draws ~= 9-11 h. Reduce --num_bootstraps or drop a seed for a faster check;
# a single seed with --num_bootstraps 10 (~2.5-3 h) already answers "does 85/98
# appear in more than one draw". ngpus=1 is kept because cdt fails to import
# without a parseable CUDA_VISIBLE_DEVICES.
python cold_start_bootstrap_dump.py \
    --graph_type Erdos100 \
    --num_bootstraps 20 \
    --seeds 71,72 \
    > $HOME/causal_bayes_opt/results/cold_start_check/parent_recovery_erdos100_pbs.log 2>&1