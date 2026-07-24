import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Latent-confounder boundary experiment on Erdos (linear) + DREAM (nonlinear),
# both x_kind configurations (confounded non-parent = false positive; confounded
# true parent = effect corruption).
n_observational = 200
n_trials = 30
n_int = 1
seed = 71
run_num = 1

# (graph_type, nonlinear-suffix-for-analysis)
graphs = ["Erdos50", "Erdos100", "Size50-Ecoli1", "Size100-Ecoli1"]
x_kinds = ["non_parent", "true_parent"]

for graph_type in graphs:
    for x_kind in x_kinds:
        command = (
            f"{sys.executable} confounder_boundary_script.py "
            f"--graph_type \"{graph_type}\" --x_kind {x_kind} "
            f"--seeds_replicate {seed} --n_observational {n_observational} "
            f"--n_trials {n_trials} --n_int {n_int} --run_num {run_num} --noiseless"
        )
        os.system(command)
        print(f"Executed: {command}")

# analysis per graph_type_x_kind subdir (dream ones are nonlinear)
for graph_type in graphs:
    ns = "--nonlinear" if graph_type.startswith("Size") else ""
    for x_kind in x_kinds:
        command = (
            f"{sys.executable} ../results_erdos/boundary_bias_analysis.py "
            f"--run_num {run_num} --n_obs {n_observational} --n_int {n_int} {ns} "
            f"--results_subdir boundary_tracking_confounded "
            f"--graph_types {graph_type}_{x_kind} "
            f"--base_name_prefix cbo_confounded_dr2_boundary"
        )
        os.system(command)
        print(f"Executed: {command}")
