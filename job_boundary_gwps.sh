#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N boundary_gwps_ei_ucb

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_gwps
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_gwps
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_gwps_oracle

# Boundary tracking on the GWPS gene network (60-node DAG subgraph, linear G_hat
# SEM at weight_scale=3 -- the diagnostic's stable/usable scale), baseline +
# oracle, once with EI and once with UCB, then the boundary-bias analysis for
# each. Assumes data/gwps_direct_edges.csv exists.
SEED=71; NOBS=200; NTRIALS=30; RUN=1; WS=3; MAXN=60; TOPK=8
TAG=gwps_n${MAXN}_ws${WS}
LOG=$HOME/causal_bayes_opt/results/boundary_tracking_gwps/gwps_ei_ucb_pbs.log

{
for ACQ in EI UCB; do
    echo "===== baseline $ACQ ====="
    python gwps_boundary_script.py --acquisition $ACQ --weight_scale $WS \
        --max_nodes $MAXN --top_k_parents $TOPK --seeds_replicate $SEED \
        --n_observational $NOBS --n_trials $NTRIALS --n_int 1 --run_num $RUN --noiseless
    echo "===== oracle $ACQ ====="
    python gwps_boundary_script.py --acquisition $ACQ --weight_scale $WS --oracle \
        --max_nodes $MAXN --top_k_parents $TOPK --seeds_replicate $SEED \
        --n_observational $NOBS --n_trials $NTRIALS --n_int 1 --run_num $RUN --noiseless

    echo "===== ANALYSIS baseline $ACQ ====="
    python ../results_erdos/boundary_bias_analysis.py \
        --run_num $RUN --n_obs $NOBS --n_int 1 \
        --results_subdir boundary_tracking_gwps --graph_types $TAG \
        --base_name_prefix cbo_unknown_dr2_boundary_$ACQ
    echo "===== ANALYSIS oracle $ACQ ====="
    python ../results_erdos/boundary_bias_analysis.py \
        --run_num $RUN --n_obs $NOBS --n_int 1 \
        --results_subdir boundary_tracking_gwps_oracle --graph_types $TAG \
        --base_name_prefix cbo_unknown_dr2_boundary_$ACQ
done
} > "$LOG" 2>&1
