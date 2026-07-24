#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=32:00:00
#PBS -N confounder_boundary

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_confounded

# Latent-confounder boundary experiment on Erdos (linear) + DREAM (nonlinear),
# both x_kind configurations. DREAM uses the NumPy nonlinear SEM; ngpus=1 kept
# for the cdt import + any jax use. Runs 4 graphs x 2 x_kinds = 8 experiments,
# each ~ a boundary run; walltime padded for the Size100 nonlinear ones.
python confounder_boundary_bash.py \
    > $HOME/causal_bayes_opt/results/boundary_tracking_confounded/confounder_pbs.log 2>&1
