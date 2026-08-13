#!/usr/bin/env python3
"""Local runner for the RANDOM NON-PARENT boundary-tracking jobs.

The counterfactual to run_oracle.py: identical protocol, but each run forces a
randomly drawn set of NON-parents (same cardinality as the true parent set)
instead of the true parents, intervened on jointly.

    ecoli -> Size50-Ecoli1, Size100-Ecoli1
    erdos -> Erdos50, Erdos100
    gwps  -> gwps_n60_ws3

Each job is repeated over 5 replicate seeds (71..75); the seed both draws the
random set and is encoded in the pickle name as the run number, so every
dataset/graph yields run1..run5 with five DIFFERENT random sets.

Results land in results/boundary_tracking_{erdos,dream,gwps}_random/.

Usage:
    python run_random.py --all
    python run_random.py erdos
    python run_random.py --list
    python run_random.py --all --dry-run
    python run_random.py erdos --no-log
"""

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

SCRIPT = "random_boundary_script.py"
CWD = os.path.join(REPO, "scripts_random")


def _job(graphs, n_int, subdir, log, extra=()):
    steps = []
    for seed, run in REPLICATES:
        for g in graphs:
            steps.append((
                CWD,
                [PY, SCRIPT, "--graph_type", g, "--kernel", KERNEL,
                 "--acquisition", ACQ, "--seeds_replicate", seed,
                 "--n_observational", NOBS, "--n_trials", NTRIALS,
                 "--n_int", n_int, "--run_num", run, "--noiseless", *extra],
                f"===== random non-parents {g} {ACQ} seed={seed} run={run} =====",
            ))
    return {"subdir": subdir, "log": log, "steps": steps}


JOBS = {
    "erdos": _job(["Erdos50", "Erdos100"], "2",
                  "boundary_tracking_erdos_random", "erdos_random_ei.log"),
    "ecoli": _job(["Size50-Ecoli1", "Size100-Ecoli1"], "1",
                  "boundary_tracking_dream_random", "ecoli_random_ei.log"),
    "gwps":  _job(["gwps"], "1",
                  "boundary_tracking_gwps_random", "gwps_random_ei.log",
                  extra=("--weight_scale", "3", "--max_nodes", "60",
                         "--top_k_parents", "8")),
}


def run_job(name, job, dry_run=False, no_log=False):
    results_dir = os.path.join(REPO, "results", job["subdir"])
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, job["log"])

    if dry_run:
        print(f"##### {name} (log: {log_path}) #####")
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

    print(f"##### running {name} #####", flush=True)
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
                print(f"##### {name} FAILED (exit {result.returncode}) #####", flush=True)
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
    parser.add_argument("jobs", nargs="*", help="Job names to run (see --list).")
    parser.add_argument("--all", action="store_true", help="Run every job.")
    parser.add_argument("--list", action="store_true", help="List job names and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--no-log", action="store_true", help="Stream to stdout.")
    args = parser.parse_args()

    if args.list:
        for name in sorted(JOBS):
            print(f"{name:8s} -> results/{JOBS[name]['subdir']}/{JOBS[name]['log']}")
        return 0

    if args.all:
        selected = sorted(JOBS)
    elif args.jobs:
        unknown = [j for j in args.jobs if j not in JOBS]
        if unknown:
            parser.error("unknown job(s): %s (see --list)" % ", ".join(unknown))
        selected = args.jobs
    else:
        parser.error("specify job name(s), --all, or --list")

    for name in selected:
        rc = run_job(name, JOBS[name], dry_run=args.dry_run, no_log=args.no_log)
        if rc != 0 and not args.dry_run:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
