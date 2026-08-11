#!/usr/bin/env python3
"""Local runner for job_oracle_ecoli_spherical.sh.

Replicates the PBS job locally: runs oracle boundary tracking on the DREAM
Ecoli graphs (Size50 + Size100) with the spherical-linear surrogate kernel and
EI acquisition, then runs the boundary-bias analysis. Output is teed to a log
file, mirroring the shell script.

Usage:
    python run_oracle_ecoli_spherical.py            # run everything
    python run_oracle_ecoli_spherical.py --dry-run  # print commands only
"""

import argparse
import os
import subprocess
import sys

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "causal_bayes_opt")

# Match the shell script's configuration.
KERNEL = "spherical_linear"
SUBDIR = "boundary_tracking_dream_oracle_spherical"
SEED = "71"
NOBS = "200"
NTRIALS = "30"
RUN = "1"
ACQ = "EI"
GRAPHS = ["Size50-Ecoli1", "Size100-Ecoli1"]


def build_commands():
    tracking = []
    for graph in GRAPHS:
        tracking.append(
            (
                os.path.join(REPO, "scripts_dream"),
                [
                    sys.executable,
                    "oracle_boundary_tracking_dream_script.py",
                    "--kernel", KERNEL,
                    "--graph_type", graph,
                    "--acquisition", ACQ,
                    "--seeds_replicate", SEED,
                    "--n_observational", NOBS,
                    "--n_trials", NTRIALS,
                    "--n_int", "1",
                    "--run_num", RUN,
                    "--noiseless",
                    "--nonlinear",
                ],
                f"===== oracle {graph} {ACQ} ({KERNEL}) =====",
            )
        )

    analysis = (
        os.path.join(REPO, "scripts_dream"),
        [
            sys.executable,
            os.path.join(REPO, "results_erdos", "boundary_bias_analysis.py"),
            "--run_num", RUN,
            "--n_obs", NOBS,
            "--n_int", "1",
            "--nonlinear",
            "--results_subdir", SUBDIR,
            "--graph_types", ",".join(GRAPHS),
            "--base_name_prefix", f"cbo_unknown_dr2_boundary_{ACQ}",
        ],
        f"===== ANALYSIS oracle {ACQ} ({KERNEL}) =====",
    )
    return tracking + [analysis]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running."
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Stream to stdout instead of writing to the log file.",
    )
    args = parser.parse_args()

    results_dir = os.path.join(REPO, "results", SUBDIR)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "ecoli_ei_pbs.log")

    commands = build_commands()

    if args.dry_run:
        for cwd, cmd, banner in commands:
            print(banner)
            print(f"  (cd {cwd})")
            print("  " + " ".join(cmd))
        print(f"\nLog would be written to: {log_path}")
        return 0

    log_file = None if args.no_log else open(log_path, "w")

    def emit(text):
        if log_file:
            log_file.write(text + "\n")
            log_file.flush()
        else:
            print(text, flush=True)

    try:
        for cwd, cmd, banner in commands:
            emit(banner)
            result = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=log_file if log_file else None,
                stderr=subprocess.STDOUT if log_file else None,
            )
            if result.returncode != 0:
                emit(f"[command failed with exit code {result.returncode}]")
                return result.returncode
    finally:
        if log_file:
            log_file.close()

    if log_file is not None:
        print(f"Done. Output written to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
