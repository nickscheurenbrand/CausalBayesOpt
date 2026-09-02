"""
Combined summary plot for the CBO-U boundary-tracking experiments.

Loads the boundary-tracking pickles for Erdos20/50/100 and produces a
2-panel figure (intervention boundary percentage and P(true parents) per
iteration, one curve per graph size) saved to
results/boundary_tracking/boundary_and_posterior_summary.png.
"""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_TYPES = ["Erdos20", "Erdos50", "Erdos100"]
COLORS = {"Erdos20": "tab:blue", "Erdos50": "tab:orange", "Erdos100": "tab:green"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--n_obs", type=int, default=200)
    parser.add_argument("--n_int", type=int, default=2)
    parser.add_argument("--nonlinear", action="store_true")
    return parser.parse_args()


def load_results(run_num: int, n_obs: int, n_int: int, nonlinear: bool):
    nonlinear_string = "_nonlinear" if nonlinear else ""
    results = {}
    for graph_type in GRAPH_TYPES:
        filename = (
            f"{REPO_ROOT}/results/boundary_tracking/{graph_type}/"
            f"run{run_num}_cbo_unknown_dr2_boundary_{n_obs}_{n_int}"
            f"{nonlinear_string}.pickle"
        )
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping {graph_type}")
            continue
        with open(filename, "rb") as file:
            results[graph_type] = pickle.load(file)
    return results


def main():
    args = parse_args()
    results = load_results(args.run_num, args.n_obs, args.n_int, args.nonlinear)
    if not results:
        print("No results found, nothing to plot")
        return

    fig, (ax_boundary, ax_posterior) = plt.subplots(1, 2, figsize=(14, 5))

    for graph_type, result in results.items():
        color = COLORS[graph_type]

        boundary = np.array(result["Boundary_Percentage"])
        iterations = np.arange(1, len(boundary) + 1)
        running_mean = np.cumsum(boundary) / iterations
        ax_boundary.scatter(iterations, boundary, alpha=0.25, color=color)
        ax_boundary.plot(iterations, running_mean, color=color, label=graph_type)

        p_true = np.array(result["P_True_Parents"])
        ax_posterior.plot(
            np.arange(len(p_true)), p_true, marker="o", color=color, label=graph_type
        )

    ax_boundary.set_xlabel("Iteration")
    ax_boundary.set_ylabel("Boundary percentage")
    ax_boundary.set_ylim(-0.05, 1.05)
    ax_boundary.set_title("Intervention boundary percentage\n(running mean + raw)")
    ax_boundary.legend()

    ax_posterior.set_xlabel("Iteration")
    ax_posterior.set_ylabel("P(true parents)")
    ax_posterior.set_ylim(-0.05, 1.05)
    ax_posterior.set_title("Posterior probability of true parent set")
    ax_posterior.legend()

    filename = (
        f"{REPO_ROOT}/results/boundary_tracking/boundary_and_posterior_summary.png"
    )
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined summary plot to {filename}")


if __name__ == "__main__":
    main()
