#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N mediation_boundary

source $HOME/causal_bayes_opt/.venv/bin/activate
cd $HOME/causal_bayes_opt

# Mediation experiment: three cardinality-matched arms per graph -- true PARENTS
# (direct effect), non-parent ANCESTORS (mediated effect) and NON-ANCESTORS (no
# causal effect at all) -- so "does a mediated effect still drive the boundary
# behaviour?" can be answered within a single graph.
#
# The default graph setups cannot host all three arms: Erdos100's default target
# sits in a 5-node component and the DREAM targets have an ancestor set equal to
# their parent set (empty mediated pool), while the GWPS carve is the target's
# ancestor closure (empty null pool). run_mediation.py therefore re-targets the
# Erdos/DREAM graphs and builds GWPS with --max_nodes 80 --n_non_ancestors 20.
# Results go to tagged directories, so nothing from the earlier oracle/random
# runs is overwritten. ngpus=1 is kept because cdt fails to import without a
# parseable CUDA_VISIBLE_DEVICES (see job_cold_start.sh).
#
# 25 runs total: 5 jobs x 5 seeds x 1 arm, at 30 trials each. Each job runs only
# its default arm (erdos/gwps -> ancestor, dream -> non_ancestor), which is what
# the mkdir list below reflects; `python run_mediation.py --list` prints the
# mapping. Each arm also does a ground-truth search for the true optimum of
# E[Y|do(.)]. Split across submissions if 24h is tight, e.g.
#   qsub -v JOBS="erdos50 erdos100" job_mediation.sh
# and add arms explicitly with, e.g., JOBS="gwps --arms parents".
# Pick what runs with -v at submit time:
#   qsub job_mediation.sh                        # every dataset (25 runs)
#   qsub -v DATASET=gwps job_mediation.sh        # one dataset
#   qsub -v DATASET=erdos,dream job_mediation.sh # several
#   qsub -v JOBS="erdos50 gwps" job_mediation.sh # individual jobs
#   qsub -v JOBS="gwps --arms parents" job_mediation.sh   # a non-default arm
# DATASET takes erdos, dream (alias ecoli) or gwps. JOBS wins if both are set.
DATASET="${DATASET:-}"
JOBS="${JOBS:-}"

if   [ -n "$JOBS" ];    then SELECT="$JOBS"
elif [ -n "$DATASET" ]; then SELECT="--dataset $DATASET"
else                         SELECT="--all"
fi

echo "##### run_mediation.py $SELECT #####"

# Create only the directories this selection needs -- the runner reports them,
# so the list cannot drift out of step with the job table.
mkdir -p $(python run_mediation.py $SELECT --print-dirs)

python run_mediation.py $SELECT
