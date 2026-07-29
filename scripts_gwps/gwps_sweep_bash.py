import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Weight-scale sweep: run CBO-U on the GWPS graph at several --weight_scale values
# so the effect of signal strength on the CBO objective (Best_Y) and on parent
# recovery is visible side by side. The raw G_hat effects are weak (diagnostic:
# best-lever SNR 0.18 at scale 1); this sweep spans too-weak -> good -> strong.
n_observational = 200
n_trials = 30
n_int = 1
max_nodes = 60
top_k_parents = 8
run_num = 1
scales = [1, 3, 5]

for ws in scales:
    command = (
        f"{sys.executable} gwps_cbo_script.py "
        f"--max_nodes {max_nodes} --top_k_parents {top_k_parents} "
        f"--weight_scale {ws} --n_observational {n_observational} "
        f"--n_trials {n_trials} --n_int {n_int} --run_num {run_num} --noiseless"
    )
    os.system(command)
    print(f"Executed: {command}")

# summary + plot across scales
command = (
    f"{sys.executable} gwps_sweep_summary.py "
    f"--run_num {run_num} --n_obs {n_observational} --n_int {n_int} "
    f"--max_nodes {max_nodes} --scales {','.join(str(s) for s in scales)}"
)
os.system(command)
print(f"Executed: {command}")
