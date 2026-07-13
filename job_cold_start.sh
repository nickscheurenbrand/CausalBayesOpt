#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=02:00:00
#PBS -N cold_start_bootstrap_dump

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/cold_start_check

# Dumps the raw per-bootstrap parent-set estimates behind the iteration-0
# candidate posterior (determine_initial_probabilities), for Erdos50/100,
# at the default num_bootstraps=10 and a much larger 200 to check whether
# raising it ever surfaces the true parents as a candidate.
# Only touches observational data (no GP fitting, no CBO trial loop), so
# this needs far less compute than boundary_tracking_bash.py / job.sh -- no
# GPU requested here; add ":ngpus=1" to the #PBS -l select line above if the
# venv's torch/jax build errors without a visible device.
LOG=$HOME/causal_bayes_opt/results/cold_start_check/cold_start_pbs.log

{
    for graph_type in Erdos50 Erdos100; do
        for num_bootstraps in 10 200; do
            echo "=== graph_type=$graph_type num_bootstraps=$num_bootstraps ==="
            python cold_start_bootstrap_dump.py \
                --graph_type "$graph_type" \
                --num_bootstraps "$num_bootstraps"
        done
    done
} > "$LOG" 2>&1
