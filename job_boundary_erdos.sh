#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=32:00:00
#PBS -N boundary_erdos_ei_ucb

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_erdos
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_oracle

# Erdos50 + Erdos100 boundary tracking, baseline + oracle, once with EI and once
# with UCB acquisition, then the boundary-bias analysis for each.
SEED=71; NOBS=200; NTRIALS=30; NANCHOR=35; RUN=1
GRAPHS="Erdos50 Erdos100"
LOG=$HOME/causal_bayes_opt/results/boundary_tracking/erdos_ei_ucb_pbs.log

{
for ACQ in EI UCB; do
    for G in $GRAPHS; do
        echo "===== baseline $G $ACQ ====="
        python boundary_tracking_script.py --graph_type "$G" --acquisition $ACQ \
            --seeds_replicate $SEED --n_observational $NOBS --n_trials $NTRIALS \
            --n_anchor_points $NANCHOR --run_num $RUN --noiseless
        echo "===== oracle $G $ACQ ====="
        python oracle_boundary_tracking_script.py --graph_type "$G" --acquisition $ACQ \
            --seeds_replicate $SEED --n_observational $NOBS --n_trials $NTRIALS \
            --n_anchor_points $NANCHOR --run_num $RUN --noiseless
    done
    echo "===== ANALYSIS baseline $ACQ ====="
    python ../results_erdos/boundary_bias_analysis.py \
        --run_num $RUN --n_obs $NOBS --n_int 2 \
        --results_subdir boundary_tracking --graph_types Erdos50,Erdos100 \
        --base_name_prefix cbo_unknown_dr2_boundary_$ACQ
    echo "===== ANALYSIS oracle $ACQ ====="
    python ../results_erdos/boundary_bias_analysis.py \
        --run_num $RUN --n_obs $NOBS --n_int 2 \
        --results_subdir boundary_tracking_oracle --graph_types Erdos50,Erdos100 \
        --base_name_prefix cbo_unknown_dr2_boundary_$ACQ
done
} > "$LOG" 2>&1
