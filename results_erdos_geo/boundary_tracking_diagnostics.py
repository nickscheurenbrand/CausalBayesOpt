"""
Diagnostics for the CBO-U boundary-tracking experiments, computed directly
from the saved pickles (no plots).

For each graph size, decomposes the two headline symptoms:
  - "interventions aren't concentrating on the boundary" -> is this because
    the implied exploration set (union of variables appearing in surviving
    parent-set hypotheses) stays large, spreading the trial budget thin?
  - "the posterior isn't accurate" -> is that because per-edge (marginal)
    recovery is fine but exact-parent-set posterior mass collapses, or
    because per-edge recovery itself is bad?

Writes one CSV per graph_type/run under
results/boundary_tracking/{graph_type}/{base_name}_diagnostics.csv
and prints a short text summary (first/last iteration only) to stdout.
"""

import argparse
import csv
import os
import pickle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_TYPES = ["Erdos20", "Erdos50", "Erdos100"]


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
        base_name = f"run{run_num}_cbo_unknown_dr2_boundary_{n_obs}_{n_int}{nonlinear_string}"
        filename = (
            f"{REPO_ROOT}/results/boundary_tracking/{graph_type}/{base_name}.pickle"
        )
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping {graph_type}")
            continue
        with open(filename, "rb") as file:
            results[graph_type] = (pickle.load(file), base_name)
    return results


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def precision_recall(estimate, truth):
    est, tru = set(estimate), set(truth)
    tp = len(est & tru)
    precision = tp / len(est) if est else (1.0 if not tru else 0.0)
    recall = tp / len(tru) if tru else (1.0 if not est else 0.0)
    return precision, recall


def implied_exploration_set(posterior_snapshot):
    """Union of variables appearing in any surviving parent-set hypothesis."""
    variables = set()
    for parents in posterior_snapshot:
        variables.update(parents)
    return variables


def analyze_run(result: dict):
    posterior_history = result["Posterior_History"]
    true_parents = result["True_Parents"]
    intervention_set = result["Intervention_Set"]
    boundary_pct = result["Boundary_Percentage"]

    rows = []
    seen_variables = set()
    for i, posterior_snapshot in enumerate(posterior_history):
        n_candidate_parent_sets = len(posterior_snapshot)
        expl_set = implied_exploration_set(posterior_snapshot)
        n_expl_vars = len(expl_set)

        # MAP parent set and its overlap with the ground truth
        map_parents, map_prob = max(posterior_snapshot.items(), key=lambda kv: kv[1])
        map_precision, map_recall = precision_recall(map_parents, true_parents)
        map_jaccard = jaccard(map_parents, true_parents)

        # posterior-weighted expected overlap with the ground truth
        expected_jaccard = sum(
            prob * jaccard(parents, true_parents)
            for parents, prob in posterior_snapshot.items()
        )
        p_true_exact = posterior_snapshot.get(tuple(true_parents), 0.0)

        # iteration 0 is the pre-trial snapshot; no intervention/boundary yet
        if i == 0:
            var_intervened = None
            boundary_this_iter = None
        else:
            var_intervened = intervention_set[i - 1]
            seen_variables.update(var_intervened)
            boundary_this_iter = boundary_pct[i - 1]

        rows.append(
            {
                "iteration": i,
                "n_candidate_parent_sets": n_candidate_parent_sets,
                "n_exploration_vars": n_expl_vars,
                "map_parents": map_parents,
                "map_prob": map_prob,
                "map_precision": map_precision,
                "map_recall": map_recall,
                "map_jaccard": map_jaccard,
                "expected_jaccard": expected_jaccard,
                "p_true_exact": p_true_exact,
                "variable_intervened": var_intervened,
                "cumulative_distinct_vars_intervened": len(seen_variables),
                "boundary_pct_this_iter": boundary_this_iter,
            }
        )
    return rows


def write_csv(rows, filename):
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cold_start_report(result: dict) -> str:
    """
    Human-readable dump of the iteration-0 candidate parent-sets (i.e. what
    the initial observational-only bootstrap fit proposed before any CBO
    trial ran), plus whether each true-parent variable was EVER a candidate
    at any point in the run. Since the candidate set can only shrink after
    iteration 0 (parent-sets are only ever deleted, never added back), a
    true-parent variable missing from this union can never be recovered.
    """
    posterior_history = result["Posterior_History"]
    true_parents = tuple(result["True_Parents"])
    iter0 = posterior_history[0]

    lines = []
    lines.append("--- iteration 0 candidate parent-sets (raw bootstrap output) ---")
    for parents, prob in sorted(iter0.items(), key=lambda kv: -kv[1]):
        overlap = set(parents) & set(true_parents)
        flag = f"  <- overlaps true parents at {sorted(overlap)}" if overlap else ""
        lines.append(f"  {parents}: {prob:.4f}{flag}")

    ever_candidate_vars = set()
    for snapshot in posterior_history:
        ever_candidate_vars.update(implied_exploration_set(snapshot))

    lines.append("")
    lines.append(
        "--- was each true-parent variable EVER a candidate (any iteration)? ---"
    )
    for var in true_parents:
        status = "YES" if var in ever_candidate_vars else "NO (locked out from iteration 0 onward)"
        lines.append(f"  {var}: {status}")

    return "\n".join(lines)


def print_summary(graph_type, rows, true_parents):
    first, last = rows[0], rows[-1]
    n_trials = last["iteration"]
    print(f"\n=== {graph_type} (true parents: {true_parents}) ===")
    print(
        f"iter 0:  {first['n_candidate_parent_sets']} candidate parent-sets, "
        f"{first['n_exploration_vars']} exploration vars, "
        f"expected_jaccard={first['expected_jaccard']:.3f}, "
        f"p_true_exact={first['p_true_exact']:.4f}"
    )
    print(
        f"iter {n_trials}: {last['n_candidate_parent_sets']} candidate parent-sets, "
        f"{last['n_exploration_vars']} exploration vars, "
        f"expected_jaccard={last['expected_jaccard']:.3f}, "
        f"p_true_exact={last['p_true_exact']:.4f}, "
        f"MAP set={last['map_parents']} (precision={last['map_precision']:.2f}, "
        f"recall={last['map_recall']:.2f}), "
        f"distinct vars intervened so far={last['cumulative_distinct_vars_intervened']}/{n_trials}"
    )


def main():
    args = parse_args()
    results = load_results(args.run_num, args.n_obs, args.n_int, args.nonlinear)
    if not results:
        print("No results found, nothing to analyze")
        return

    for graph_type, (result, base_name) in results.items():
        rows = analyze_run(result)
        out_dir = f"{REPO_ROOT}/results/boundary_tracking/{graph_type}"
        out_path = f"{out_dir}/{base_name}_diagnostics.csv"
        write_csv(rows, out_path)
        print(f"Saved diagnostics to {out_path}")
        print_summary(graph_type, rows, result["True_Parents"])

        report = cold_start_report(result)
        report_path = f"{out_dir}/{base_name}_cold_start_report.txt"
        with open(report_path, "w") as f:
            f.write(f"=== {graph_type} (true parents: {result['True_Parents']}) ===\n")
            f.write(report + "\n")
        print(report)
        print(f"Saved cold-start report to {report_path}")


if __name__ == "__main__":
    main()
