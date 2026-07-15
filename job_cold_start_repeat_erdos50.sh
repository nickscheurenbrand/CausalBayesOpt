#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=08:00:00
#PBS -N cold_start_repeat_erdos50

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/cold_start_check

# Repeats the cold-start candidate-parent-set construction (bootstrap +
# initial-D_I Bayesian update, i.e. everything before iteration 0 of the CBO
# loop) 5x against the same data, to see how much posterior_history[0] varies
# purely from the doubly-robust bootstrap's unseeded RNG. Each repeat costs
# ~76 min for Erdos50 (10 bootstraps @ ~7.5 min each); 5 repeats ~= 6.3h,
# walltime padded to 8h for variance. Lower --num_repeats if you want a
# faster turnaround.
python cold_start_repeat_check.py \
    --graph_type Erdos50 \
    --num_repeats 5 \
    > $HOME/causal_bayes_opt/results/cold_start_check/cold_start_repeat_erdos50_pbs.log 2>&1
