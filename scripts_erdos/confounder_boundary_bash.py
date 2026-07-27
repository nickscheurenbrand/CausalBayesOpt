import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Latent-confounder boundary experiment on Erdos (linear) + DREAM (nonlinear).
# X is oracle-FORCED as the sole candidate so it is intervened every trial
# (guarantees observability; the earlier merely-seeded runs had X pruned by the
# nonlinear collapse on DREAM). For each (graph, x_kind) we run:
#   conf  -> hidden confounder Z->X, Z->Y  (confounded observational data)
#   ctrl  -> control, no confounder, same X (clean observational data)
# The conf-vs-ctrl contrast on the SAME forced X isolates the confounding's
# effect on X's boundary behaviour.
n_observational = 200
n_trials = 30
n_int = 1
seed = 71
run_num = 1

graphs = ["Erdos50", "Erdos100", "Size50-Ecoli1", "Size100-Ecoli1"]
x_kinds = ["non_parent", "true_parent"]
conditions = [("conf", ""), ("ctrl", "--no_confounder")]

for graph_type in graphs:
    for x_kind in x_kinds:
        for tag, extra in conditions:
            command = (
                f"{sys.executable} confounder_boundary_script.py "
                f"--graph_type \"{graph_type}\" --x_kind {x_kind} --force_x {extra} "
                f"--tag {tag} --seeds_replicate {seed} "
                f"--n_observational {n_observational} --n_trials {n_trials} "
                f"--n_int {n_int} --run_num {run_num} --noiseless"
            )
            os.system(command)
            print(f"Executed: {command}")

# analysis per subdir (dream => --nonlinear)
for graph_type in graphs:
    ns = "--nonlinear" if graph_type.startswith("Size") else ""
    for x_kind in x_kinds:
        for tag, _ in conditions:
            command = (
                f"{sys.executable} ../results_erdos/boundary_bias_analysis.py "
                f"--run_num {run_num} --n_obs {n_observational} --n_int {n_int} {ns} "
                f"--results_subdir boundary_tracking_confounded "
                f"--graph_types {graph_type}_{x_kind}_{tag} "
                f"--base_name_prefix cbo_confounded_dr2_boundary"
            )
            os.system(command)
            print(f"Executed: {command}")
