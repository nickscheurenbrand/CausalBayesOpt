import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Baseline boundary tracking on the larger DREAM (dream4) nets. Matches the
# Erdos boundary sweep's fixed params (n_obs=200, 30 trials, seed 71) so the
# results are directly comparable via boundary_bias_analysis.py.
n_observational = 200
graphs = ["Size50-Ecoli1", "Size100-Ecoli1"]
n_trials = 30
n_int = 1
noiseless = "--noiseless"
seed = 71
run_num = 1

for graph_type in graphs:
    command = (
        f"{sys.executable} boundary_tracking_dream_script.py "
        f"--seeds_replicate {seed} --n_observational {n_observational} "
        f"--n_trials {n_trials} --n_int {n_int} --run_num {run_num} {noiseless} "
        f"--nonlinear --graph_type \"{graph_type}\""
    )
    os.system(command)
    print(f"Executed: {command}")

# boundary-bias report on the DREAM output (nonlinear => --nonlinear so the
# analysis loader reconstructs the same pickle base name)
command = (
    f"{sys.executable} ../results_erdos/boundary_bias_analysis.py "
    f"--run_num {run_num} --n_obs {n_observational} --n_int {n_int} --nonlinear "
    f"--results_subdir boundary_tracking_dream "
    f"--graph_types {','.join(graphs)}"
)
os.system(command)
print(f"Executed: {command}")
