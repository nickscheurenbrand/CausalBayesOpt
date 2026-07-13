import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Define the fixed parameters
n_observational = 200
graphs = ["Erdos20", "Erdos50", "Erdos100"]
n_trials = 30
n_anchor_points = 35
noiseless = "--noiseless"
seed = 71
run_num = 1

for graph_type in graphs:
    command = (
        f"{sys.executable} boundary_tracking_script.py --seeds_replicate {seed} "
        f"--n_observational {n_observational} --n_trials {n_trials} "
        f"--n_anchor_points {n_anchor_points} --run_num {run_num} {noiseless} "
        f'--graph_type "{graph_type}"'
    )
    os.system(command)
    print(f"Executed: {command}")

# combined summary plot across graph sizes
command = (
    f"{sys.executable} ../results_erdos/plot_boundary_tracking.py "
    f"--run_num {run_num} --n_obs {n_observational}"
)
os.system(command)
print(f"Executed: {command}")
