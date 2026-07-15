#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=16:00:00
#PBS -N cold_start_repeat_erdos100

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/cold_start_check

# Same as job_cold_start_repeat_erdos50.sh but for Erdos100. Individual mode
# loops a per-variable test over every candidate parent (99 here vs 49 for
# Erdos50), so each bootstrap costs roughly double: ~13-17 min/bootstrap,
# ~2.5-3h per repeat, 5 repeats ~= 13-15h. Walltime padded to 16h for
# variance. Lower --num_repeats if you want a faster turnaround, or submit
# this alongside job_cold_start_repeat_erdos50.sh as separate jobs so they
# run in parallel instead of sequentially.
python cold_start_repeat_check.py \
    --graph_type Erdos100 \
    --num_repeats 5 \
    > $HOME/causal_bayes_opt/results/cold_start_check/cold_start_repeat_erdos100_pbs.log 2>&1
