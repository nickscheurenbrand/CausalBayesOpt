#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=16:00:00
#PBS -N gwps_sweep

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_gwps

mkdir -p $HOME/causal_bayes_opt/results/gwps

# CBO-U weight-scale sweep on the GWPS graph (scales 1, 3, 5) + summary/plot.
# Shows how signal strength affects the CBO objective and parent recovery:
# scale 1 (raw G_hat) is too weak, scale 3 is the diagnostic's sweet spot, scale 5
# is stronger-still-stable. Assumes data/gwps_direct_edges.csv exists.
python gwps_sweep_bash.py \
    > $HOME/causal_bayes_opt/results/gwps/gwps_sweep_pbs.log 2>&1
