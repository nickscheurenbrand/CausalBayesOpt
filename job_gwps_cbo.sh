#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=08:00:00
#PBS -N gwps_cbo

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_gwps

mkdir -p $HOME/causal_bayes_opt/results/gwps

# One end-to-end CBO-U (PARENT_SCALE) run on a 60-node DAG subgraph of the GWPS
# gene network with a linear-Gaussian SEM using the real G_hat weights. Assumes
# data/gwps_direct_edges.csv already exists (run data/extract_gwps_direct_edges.py
# once if not). ngpus=1 kept for the cdt import.
#
# --weight_scale 3: the raw G_hat effects are too weak for CBO to detect
# (gwps_diagnose.py: best-lever SNR 0.18 at scale 1). Scale 3 gives SNR ~1 with a
# few detectable levers while staying numerically stable (scale 10 explodes).
python gwps_cbo_script.py \
    --max_nodes 60 --top_k_parents 8 --weight_scale 3 \
    --n_observational 200 --n_trials 30 --noiseless \
    > $HOME/causal_bayes_opt/results/gwps/gwps_pbs.log 2>&1
