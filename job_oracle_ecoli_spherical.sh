#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=32:00:00
#PBS -N oracle_ecoli_spherical_ei

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt/scripts_dream
mkdir -p $HOME/causal_bayes_opt/results/boundary_tracking_dream_oracle_spherical

# ORACLE Ecoli Size50 + Size100 (DREAM, nonlinear SEM) boundary tracking
# (intervene only on the true parents) with the spherical-LINEAR surrogate
# kernel, EI acquisition only, then the boundary-bias analysis.
KERNEL=spherical_linear; SUBDIR=boundary_tracking_dream_oracle_spherical
SEED=71; NOBS=200; NTRIALS=30; RUN=1; ACQ=EI
GRAPHS="Size50-Ecoli1 Size100-Ecoli1"
LOG=$HOME/causal_bayes_opt/results/$SUBDIR/ecoli_ei_pbs.log

{
for G in $GRAPHS; do
    echo "===== oracle $G $ACQ ($KERNEL) ====="
    python oracle_boundary_tracking_dream_script.py --kernel $KERNEL \
        --graph_type "$G" --acquisition $ACQ \
        --seeds_replicate $SEED --n_observational $NOBS --n_trials $NTRIALS \
        --n_int 1 --run_num $RUN --noiseless --nonlinear
done
echo "===== ANALYSIS oracle $ACQ ($KERNEL) ====="
python ../results_erdos/boundary_bias_analysis.py \
    --run_num $RUN --n_obs $NOBS --n_int 1 --nonlinear \
    --results_subdir $SUBDIR --graph_types Size50-Ecoli1,Size100-Ecoli1 \
    --base_name_prefix cbo_unknown_dr2_boundary_$ACQ
} > "$LOG" 2>&1
