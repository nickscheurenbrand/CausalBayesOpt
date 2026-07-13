#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=20:00:00
#PBS -N boundary_analysis_acqisition

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_erdos

mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking

# runs CBO-U with boundary + parent-posterior tracking on Erdos20/50/100,
# then builds the combined summary plot
python boundary_tracking_bash.py > $HOME/causal_bayes_opt/results/boundary_tracking/pipeline_pbs.log 2>&1
