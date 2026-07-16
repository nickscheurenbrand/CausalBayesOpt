import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Fixed parameters -- matched to boundary_tracking_bash.py so the oracle runs
# are directly comparable to the non-oracle pickles (same seed, n_obs, trials).
n_observational = 200
# Erdos100 is the confound-critical one; Erdos50 is a positive control that
# should stay boundary-biased (it already intervened on a true parent).
graphs = ["Erdos50", "Erdos100"]
n_trials = 30
n_anchor_points = 35
noiseless = "--noiseless"
seed = 71
run_num = 1

for graph_type in graphs:
    command = (
        f"{sys.executable} oracle_boundary_tracking_script.py --seeds_replicate {seed} "
        f"--n_observational {n_observational} --n_trials {n_trials} "
        f"--n_anchor_points {n_anchor_points} --run_num {run_num} {noiseless} "
        f'--graph_type "{graph_type}"'
    )
    os.system(command)
    print(f"Executed: {command}")

# boundary-bias report on the oracle output
command = (
    f"{sys.executable} ../results_erdos/boundary_bias_analysis.py "
    f"--run_num {run_num} --n_obs {n_observational} "
    f"--results_subdir boundary_tracking_oracle"
)
os.system(command)
print(f"Executed: {command}")
