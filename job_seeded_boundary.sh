#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N seeded_boundary_tracking

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_seeded

# Injects the true parents into the initial candidate posterior at small mass
# (alongside the REAL bootstrap, so they compete with realistic wrong
# candidates), then runs the full 30-trial CBO and tracks whether that mass is
# promoted or pruned. Because it runs the real bootstrap, it incurs the same
# ~76 min (Erdos50) / ~2.5-3 h (Erdos100) parent-ID cost as the original
# boundary run, plus the CBO loop. ngpus=1 kept for the cdt import.
python seeded_boundary_tracking_bash.py \
    > $HOME/causal_bayes_opt/results/boundary_tracking_seeded/seeded_pbs.log 2>&1
