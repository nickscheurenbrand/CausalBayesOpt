#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N boundary_tracking_dream

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_dream

mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_dream

# Baseline boundary tracking on the DREAM Size50/Size100 E. coli nets, then the
# boundary-bias analysis. The DREAM SEM is a JAX nonlinear-Gaussian model, so a
# GPU is requested (ngpus=1 also keeps CUDA_VISIBLE_DEVICES parseable for the
# cdt import). Size100 nonlinear + 30 trials is the long pole; walltime padded.
python boundary_tracking_dream_bash.py \
    > $HOME/causal_bayes_opt/results/boundary_tracking_dream/dream_pbs.log 2>&1
