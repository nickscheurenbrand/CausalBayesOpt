#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=20:00:00
#PBS -N oracle_erdos_spherical_ei

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_erdos
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_oracle_spherical

# ORACLE Erdos50 + Erdos100 boundary tracking (intervene only on the true
# parents) with the spherical-LINEAR surrogate kernel, EI acquisition only,
# then the boundary-bias analysis.
KERNEL=spherical_linear; SUBDIR=boundary_tracking_oracle_spherical
SEED=71; NOBS=200; NTRIALS=30; NANCHOR=35; RUN=1; ACQ=EI
GRAPHS="Erdos50 Erdos100"
LOG=$HOME/causal_bayes_opt/results/$SUBDIR/erdos_ei_pbs.log

{
for G in $GRAPHS; do
    echo "===== oracle $G $ACQ ($KERNEL) ====="
    python oracle_boundary_tracking_script.py --kernel $KERNEL \
        --graph_type "$G" --acquisition $ACQ \
        --seeds_replicate $SEED --n_observational $NOBS --n_trials $NTRIALS \
        --n_anchor_points $NANCHOR --run_num $RUN --noiseless
done
echo "===== ANALYSIS oracle $ACQ ($KERNEL) ====="
python ../results_erdos/boundary_bias_analysis.py \
    --run_num $RUN --n_obs $NOBS --n_int 2 \
    --results_subdir $SUBDIR --graph_types Erdos50,Erdos100 \
    --base_name_prefix cbo_unknown_dr2_boundary_$ACQ
} > "$LOG" 2>&1
