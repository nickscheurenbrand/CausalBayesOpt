#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N oracle_gwps_spherical_ei

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_gwps
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_gwps_oracle_spherical

# ORACLE GWPS gene network (60-node DAG, linear G_hat SEM at weight_scale=3)
# boundary tracking (intervene only on the true parents) with the spherical-
# LINEAR surrogate kernel, EI acquisition only, then the boundary-bias analysis.
# Assumes data/gwps_direct_edges.csv exists.
KERNEL=spherical_linear; SUBDIR=boundary_tracking_gwps_oracle_spherical
SEED=71; NOBS=200; NTRIALS=30; RUN=1; WS=3; MAXN=60; TOPK=8; ACQ=EI
TAG=gwps_n${MAXN}_ws${WS}
LOG=$HOME/causal_bayes_opt/results/$SUBDIR/gwps_ei_pbs.log

{
echo "===== oracle $ACQ ($KERNEL) ====="
python gwps_boundary_script.py --kernel $KERNEL --acquisition $ACQ --weight_scale $WS \
    --oracle --max_nodes $MAXN --top_k_parents $TOPK --seeds_replicate $SEED \
    --n_observational $NOBS --n_trials $NTRIALS --n_int 1 --run_num $RUN --noiseless
echo "===== ANALYSIS oracle $ACQ ($KERNEL) ====="
python ../results_erdos/boundary_bias_analysis.py \
    --run_num $RUN --n_obs $NOBS --n_int 1 \
    --results_subdir $SUBDIR --graph_types $TAG \
    --base_name_prefix cbo_unknown_dr2_boundary_$ACQ
} > "$LOG" 2>&1
