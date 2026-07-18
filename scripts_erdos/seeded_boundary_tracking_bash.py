import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Seed-and-track: inject the true parents into the initial candidate posterior
# at a small mass alongside the real bootstrap, then run the full 30-trial CBO
# and watch whether that mass is promoted or pruned. Erdos100 is the question of
# interest; Erdos50 is a cheaper comparison.
n_observational = 200
graphs = ["Erdos50", "Erdos100"]
n_trials = 30
seed = 71
run_num = 1
inject_prob = 0.1

for graph_type in graphs:
    command = (
        f"{sys.executable} seeded_boundary_tracking_script.py "
        f"--seeds_replicate {seed} --n_observational {n_observational} "
        f"--n_trials {n_trials} --run_num {run_num} --noiseless "
        f"--inject_prob {inject_prob} --graph_type \"{graph_type}\""
    )
    os.system(command)
    print(f"Executed: {command}")
