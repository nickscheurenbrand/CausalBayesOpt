#!/usr/bin/env python3
"""Local runner replicating the six job_oracle_*_spherical[_rbf].sh PBS jobs (ecoli/erdos/gwps x linear/rbf), without PBS.
Use --list for job names, --help for all flags."""

import argparse
import os
import subprocess
import sys

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "causal_bayes_opt")
PY = sys.executable

# Shared config (identical across all six shell scripts).
SEED = "71"
NOBS = "200"
NTRIALS = "30"
RUN = "1"
ACQ = "EI"

ANALYSIS = os.path.join(REPO, "results_erdos", "boundary_bias_analysis.py")


def _ecoli(kernel, subdir):
    cwd = os.path.join(REPO, "scripts_dream")
    graphs = ["Size50-Ecoli1", "Size100-Ecoli1"]
    steps = []
    for g in graphs:
        steps.append((
            cwd,
            [PY, "oracle_boundary_tracking_dream_script.py",
             "--kernel", kernel, "--graph_type", g, "--acquisition", ACQ,
             "--seeds_replicate", SEED, "--n_observational", NOBS,
             "--n_trials", NTRIALS, "--n_int", "1", "--run_num", RUN,
             "--noiseless", "--nonlinear"],
            f"===== oracle {g} {ACQ} ({kernel}) =====",
        ))
    steps.append((
        cwd,
        [PY, ANALYSIS, "--run_num", RUN, "--n_obs", NOBS, "--n_int", "1",
         "--nonlinear", "--results_subdir", subdir,
         "--graph_types", ",".join(graphs),
         "--base_name_prefix", f"cbo_unknown_dr2_boundary_{ACQ}"],
        f"===== ANALYSIS oracle {ACQ} ({kernel}) =====",
    ))
    return {"subdir": subdir, "log": "ecoli_ei_pbs.log", "steps": steps}


def _erdos(kernel, subdir):
    cwd = os.path.join(REPO, "scripts_erdos")
    graphs = ["Erdos50", "Erdos100"]
    nanchor = "35"
    steps = []
    for g in graphs:
        steps.append((
            cwd,
            [PY, "oracle_boundary_tracking_script.py",
             "--kernel", kernel, "--graph_type", g, "--acquisition", ACQ,
             "--seeds_replicate", SEED, "--n_observational", NOBS,
             "--n_trials", NTRIALS, "--n_anchor_points", nanchor,
             "--run_num", RUN, "--noiseless"],
            f"===== oracle {g} {ACQ} ({kernel}) =====",
        ))
    steps.append((
        cwd,
        [PY, ANALYSIS, "--run_num", RUN, "--n_obs", NOBS, "--n_int", "2",
         "--results_subdir", subdir, "--graph_types", ",".join(graphs),
         "--base_name_prefix", f"cbo_unknown_dr2_boundary_{ACQ}"],
        f"===== ANALYSIS oracle {ACQ} ({kernel}) =====",
    ))
    return {"subdir": subdir, "log": "erdos_ei_pbs.log", "steps": steps}


def _gwps(kernel, subdir):
    cwd = os.path.join(REPO, "scripts_gwps")
    ws, maxn, topk = "3", "60", "8"
    tag = f"gwps_n{maxn}_ws{ws}"
    steps = [
        (
            cwd,
            [PY, "gwps_boundary_script.py",
             "--kernel", kernel, "--acquisition", ACQ, "--weight_scale", ws,
             "--oracle", "--max_nodes", maxn, "--top_k_parents", topk,
             "--seeds_replicate", SEED, "--n_observational", NOBS,
             "--n_trials", NTRIALS, "--n_int", "1", "--run_num", RUN,
             "--noiseless"],
            f"===== oracle {ACQ} ({kernel}) =====",
        ),
        (
            cwd,
            [PY, ANALYSIS, "--run_num", RUN, "--n_obs", NOBS, "--n_int", "1",
             "--results_subdir", subdir, "--graph_types", tag,
             "--base_name_prefix", f"cbo_unknown_dr2_boundary_{ACQ}"],
            f"===== ANALYSIS oracle {ACQ} ({kernel}) =====",
        ),
    ]
    return {"subdir": subdir, "log": "gwps_ei_pbs.log", "steps": steps}


# Map job name -> job definition (mirrors the six shell scripts).
JOBS = {
    "ecoli_linear": _ecoli("spherical_linear", "boundary_tracking_dream_oracle_spherical"),
    "ecoli_rbf":    _ecoli("spherical_rbf",    "boundary_tracking_dream_oracle_spherical_rbf"),
    "erdos_linear": _erdos("spherical_linear", "boundary_tracking_oracle_spherical"),
    "erdos_rbf":    _erdos("spherical_rbf",    "boundary_tracking_oracle_spherical_rbf"),
    "gwps_linear":  _gwps("spherical_linear",  "boundary_tracking_gwps_oracle_spherical"),
    "gwps_rbf":     _gwps("spherical_rbf",     "boundary_tracking_gwps_oracle_spherical_rbf"),
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
    parser.add_argument("jobs", nargs="*",
                        help="Job names (see --list)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-log", action="store_true",
                        help="Stream to stdout")
    args = parser.parse_args()

    if args.list:
        for name in sorted(JOBS):
            print(f"{name:14s} -> results/{JOBS[name]['subdir']}/{JOBS[name]['log']}")
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
