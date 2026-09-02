#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=20:00:00
#PBS -N oracle_boundary_tracking

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_oracle

# Confound-closing experiment: forces the true parent set (skips the
# doubly-robust bootstrap), so Erdos50/Erdos100 intervene on their real
# parents, then measures boundary bias. Skipping the bootstrap makes this far
# cheaper than the original boundary run (no ~76 min/graph parent-ID step);
# the cost is just the 30-trial CBO loop. ngpus=1 is kept because cdt fails to
# import without a parseable CUDA_VISIBLE_DEVICES (see job_cold_start.sh).
python oracle_boundary_tracking_bash.py \
    > $HOME/causal_bayes_opt/results/boundary_tracking_oracle/oracle_pbs.log 2>&1
