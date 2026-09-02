#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=16:00:00
#PBS -N gwps_three

source $HOME/causal_bayes_opt/.venv/bin/activate

cd $HOME/causal_bayes_opt/scripts_gwps

mkdir -p $HOME/causal_bayes_opt/results/gwps_three

# CBO (true parents) + PARENT_SCALE (dr2) + RANDOM_SCALE on the gwps gene network
# (60-node DAG subgraph, linear G_hat SEM at weight_scale=3), 3 replicate seeds
# -> 3 runs. The gwps analogue of job_erdos_50_100.sh. Parameters (n_int=1,
# weight_scale=3, max_nodes=60, top_k=8) match the gwps geometry run so the
# results are directly comparable. Output filenames follow the Erdos "three"
# convention (cbo_results / cbo_unknown_dr2_results / cbo_results_random) so the
# results_erdos notebooks work. ngpus=1 is kept because cdt fails to import
# without a parseable CUDA_VISIBLE_DEVICES (see job_oracle_boundary.sh).
nvidia-smi

LOG=$HOME/causal_bayes_opt/results/gwps_three/gwps_three_pbs.log

bash run_gwps_three.sh > "$LOG" 2>&1