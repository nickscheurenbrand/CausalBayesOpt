#!/usr/bin/env python3
"""Local runner for the MEDIATION boundary-tracking jobs: cardinality-matched parents/ancestor/non_ancestor arms isolating direct vs. mediated vs. null causal effect.
Run with --list to see jobs/arms/step counts, or --help for all flags."""

import argparse
import os
import subprocess
import sys

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "causal_bayes_opt")
PY = sys.executable

NOBS = "200"
NTRIALS = "30"
ACQ = "EI"
KERNEL = "rbf"

SEEDS = ["71", "72", "73", "74", "75"]
REPLICATES = [(seed, str(i + 1)) for i, seed in enumerate(SEEDS)]

ARMS = ("parents", "ancestor", "non_ancestor")

# --dataset selects a whole family; "ecoli" and "dream" mean the same thing
DATASET_ALIASES = {"erdos": "erdos", "dream": "dream", "ecoli": "dream",
                   "gwps": "gwps"}

RANDOM_CWD = os.path.join(REPO, "scripts_random")
RANDOM_SCRIPT = "random_boundary_script.py"


def _random_arm(graph_type, category, seed, run, n_int, extra):
    """One random-arm command: stratified draw + ground-truth optimum."""
    return (
        RANDOM_CWD,
        [PY, RANDOM_SCRIPT, "--graph_type", graph_type,
         "--node_category", category, "--category_subdir",
         "--kernel", KERNEL, "--acquisition", ACQ,
         "--seeds_replicate", seed, "--n_observational", NOBS,
         "--n_trials", NTRIALS, "--n_int", n_int, "--run_num", run,
         "--noiseless", *extra],
        f"===== {category} {graph_type} {ACQ} seed={seed} run={run} =====",
    )


def _erdos(graph_type, target, subdir, log, default_arms=ARMS):
    n_int, extra = "2", ["--target", target]
    steps = []
    for seed, run in REPLICATES:
        for arm in ARMS:
            if arm == "parents":
                steps.append((
                    os.path.join(REPO, "scripts_erdos"),
                    [PY, "oracle_boundary_tracking_script.py",
                     "--graph_type", graph_type, "--target", target,
                     "--kernel", KERNEL, "--acquisition", ACQ,
                     "--seeds_replicate", seed, "--n_observational", NOBS,
                     "--n_trials", NTRIALS, "--run_num", run, "--noiseless"],
                    f"===== parents {graph_type} {ACQ} seed={seed} run={run} =====",
                ))
            else:
                steps.append(_random_arm(graph_type, arm, seed, run, n_int, extra))
    return {"subdir": subdir, "log": log, "steps": steps,
            "default_arms": tuple(default_arms), "family": "erdos"}


def _ecoli(graph_type, target, subdir, log, default_arms=ARMS):
    n_int, extra = "1", ["--target", target]
    steps = []
    for seed, run in REPLICATES:
        for arm in ARMS:
            if arm == "parents":
                steps.append((
                    os.path.join(REPO, "scripts_dream"),
                    [PY, "oracle_boundary_tracking_dream_script.py",
                     "--graph_type", graph_type, "--target", target,
                     "--kernel", KERNEL, "--acquisition", ACQ,
                     "--seeds_replicate", seed, "--n_observational", NOBS,
                     "--n_trials", NTRIALS, "--n_int", n_int, "--run_num", run,
                     "--noiseless", "--nonlinear"],
                    f"===== parents {graph_type} {ACQ} seed={seed} run={run} =====",
                ))
            else:
                steps.append(_random_arm(graph_type, arm, seed, run, n_int, extra))
    return {"subdir": subdir, "log": log, "steps": steps,
            "default_arms": tuple(default_arms), "family": "dream"}


def _gwps(subdir, log, default_arms=ARMS):
    # max_nodes 80 + 20 reserved non-ancestors keeps k=8 and the full mediated
    # pool while creating a null pool; every arm must pass the same carve flags
    ws, maxn, topk, nonanc = "3", "80", "8", "20"
    carve = ["--weight_scale", ws, "--max_nodes", maxn,
             "--top_k_parents", topk, "--n_non_ancestors", nonanc]
    steps = []
    for seed, run in REPLICATES:
        for arm in ARMS:
            if arm == "parents":
                steps.append((
                    os.path.join(REPO, "scripts_gwps"),
                    [PY, "gwps_boundary_script.py", "--oracle",
                     "--kernel", KERNEL, "--acquisition", ACQ,
                     "--seeds_replicate", seed, "--n_observational", NOBS,
                     "--n_trials", NTRIALS, "--n_int", "1", "--run_num", run,
                     "--noiseless", *carve],
                    f"===== parents gwps {ACQ} seed={seed} run={run} =====",
                ))
            else:
                steps.append(_random_arm("gwps", arm, seed, run, "1", carve))
    return {"subdir": subdir, "log": log, "steps": steps,
            "default_arms": tuple(default_arms), "family": "gwps"}


# Targets below were picked by scanning every node of the fixed topologies for
# one with BOTH a mediated-ancestor pool and a non-ancestor pool of at least k
# (see results_erdos/mediation_feasible_targets.py).
JOBS = {
    # graph        target  k  mediated pool  non-ancestor pool
    # Erdos50        20    3        6              40
    "erdos50":  _erdos("Erdos50", "20",
                       "boundary_tracking_erdos_random_ancestor",
                       "erdos50_mediation_ei.log",
                       default_arms=("ancestor",)),
    # Erdos100       10    3        9              87
    "erdos100": _erdos("Erdos100", "10",
                       "boundary_tracking_erdos_random_ancestor",
                       "erdos100_mediation_ei.log",
                       default_arms=("ancestor",)),
    # Size50-Ecoli1  28    2        4              43
    "ecoli50":  _ecoli("Size50-Ecoli1", "28",
                       "boundary_tracking_dream_random_non_ancestor",
                       "ecoli50_mediation_ei.log",
                       default_arms=("non_ancestor",)),
    # Size100-Ecoli1 66    2        6              91
    "ecoli100": _ecoli("Size100-Ecoli1", "66",
                       "boundary_tracking_dream_random_non_ancestor",
                       "ecoli100_mediation_ei.log",
                       default_arms=("non_ancestor",)),
    # gwps (n80, 20 reserved)  k=8  mediated 50    non-ancestor 21
    "gwps":     _gwps("boundary_tracking_gwps_random_ancestor",
                      "gwps_mediation_ei.log",
                      default_arms=("ancestor",)),
}


def jobs_for_datasets(datasets):
    """Job names belonging to any of `datasets`, in a stable order."""
    families = {DATASET_ALIASES[d] for d in datasets}
    return [n for n in sorted(JOBS) if JOBS[n]["family"] in families]


def filter_arms(job, arms=None):
    """Keep only the steps for `arms` (defaults to the job's own arms; the
    other arms are still built, so --arms can pull them in without edits)."""
    wanted = arms or job["default_arms"]
    keep = [s for s in job["steps"] if s[2].split()[1] in wanted]
    return {**job, "steps": keep, "selected_arms": tuple(wanted)}


def run_job(name, job, dry_run=False, no_log=False):
    results_dir = os.path.join(REPO, "results", job["subdir"])
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, job["log"])

    if dry_run:
        print(f"##### {name} ({len(job['steps'])} steps, log: {log_path}) #####")
        for cwd, cmd, banner in job["steps"]:
            print(banner)
            print(f"  (cd {cwd})")
            print("  " + " ".join(cmd))
        print()
        return 0

    log_file = None if no_log else open(log_path, "w")

    def emit(text):
        if log_file:
            log_file.write(text + "\n")
            log_file.flush()
        else:
            print(text, flush=True)

    print(f"##### running {name} ({len(job['steps'])} steps) #####", flush=True)
    try:
        for cwd, cmd, banner in job["steps"]:
            emit(banner)
            result = subprocess.run(
                cmd, cwd=cwd,
                stdout=log_file if log_file else None,
                stderr=subprocess.STDOUT if log_file else None,
            )
            if result.returncode != 0:
                emit(f"[command failed with exit code {result.returncode}]")
                print(f"##### {name} FAILED (exit {result.returncode}) #####",
                      flush=True)
                return result.returncode
    finally:
        if log_file:
            log_file.close()

    if log_file is not None:
        print(f"##### {name} done -> {log_path} #####", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", nargs="*", help="Job names (see --list)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Comma-separated dataset(s) to run: %s"
                             % ",".join(sorted(DATASET_ALIASES)))
    parser.add_argument("--print-dirs", action="store_true",
                        help="Print results dirs and exit")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--arms", type=str, default=None,
                        help="Comma-separated subset of %s" % ",".join(ARMS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-log", action="store_true", help="Stream to stdout")
    args = parser.parse_args()

    if args.list:
        for name in sorted(JOBS):
            j = filter_arms(JOBS[name])
            print(f"{name:10s} [{j['family']:5s}] {len(j['steps']):3d} steps  "
                  f"arms={','.join(j['default_arms']):24s} -> "
                  f"results/{j['subdir']}/{j['log']}")
        return 0

    arms = None
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            parser.error("unknown arm(s): %s (choose from %s)"
                         % (", ".join(unknown), ", ".join(ARMS)))

    datasets = None
    if args.dataset:
        datasets = [d.strip().lower() for d in args.dataset.split(",") if d.strip()]
        unknown = [d for d in datasets if d not in DATASET_ALIASES]
        if unknown:
            parser.error("unknown dataset(s): %s (choose from %s)"
                         % (", ".join(unknown), ", ".join(sorted(DATASET_ALIASES))))

    if args.all:
        selected = sorted(JOBS)
    elif datasets:
        selected = jobs_for_datasets(datasets)
        if args.jobs:   # --dataset narrows, explicit names still win
            selected = [j for j in selected if j in args.jobs] or args.jobs
    elif args.jobs:
        unknown = [j for j in args.jobs if j not in JOBS]
        if unknown:
            parser.error("unknown job(s): %s (see --list)" % ", ".join(unknown))
        selected = args.jobs
    else:
        parser.error("specify job name(s), --dataset, --all, or --list")

    if args.print_dirs:
        seen = []
        for name in selected:
            d = os.path.join(REPO, "results", JOBS[name]["subdir"])
            if d not in seen:
                seen.append(d)
        print("\n".join(seen))
        return 0

    for name in selected:
        rc = run_job(name, filter_arms(JOBS[name], arms),
                     dry_run=args.dry_run, no_log=args.no_log)
        if rc != 0 and not args.dry_run:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
