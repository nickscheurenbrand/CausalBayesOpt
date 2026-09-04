"""Driver: GEOMETRY_SCALE boundary tracking (appendix-E adaptive geometry + TabPFN prior mean + spherical BLR) on
Erdos50/100 over N replicate seeds; needs GPU. Delegates to geometry_boundary_script.py; --variants oracle adds forced-true-parent runs."""

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

GRAPHS = ["Erdos50", "Erdos100"]
SCRIPT = "geometry_boundary_script.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", type=str, default=",".join(GRAPHS))
    p.add_argument("--runs", type=int, default=5,
                   help="number of replicates")
    p.add_argument("--base_seed", type=int, default=71)
    p.add_argument("--variants", type=str, default="baseline",
                   help="comma-separated subset of baseline, oracle")
    p.add_argument("--n_observational", type=int, default=200)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_int", type=int, default=2)
    p.add_argument("--acquisition", type=str, default="EI", choices=["EI", "UCB"])
    p.add_argument("--prior", type=str, default="tabpfn",
                   choices=["tabpfn", "pfn", "zero", "constant"])
    p.add_argument("--prior_mean", type=str, default="pfn",
                   choices=["pfn", "pfn+do", "do", "zero"])
    p.add_argument("--no_adapt_geometry", action="store_true")
    p.add_argument("--nonlinear", action="store_true",
                   help="nonlinear SEM (erdos only)")
    p.add_argument("--target", type=str, default=None,
                   help="override target node (dream/gwps)")
    # gwps-only knobs; forwarded only for --graphs gwps, ignored by the others
    p.add_argument("--max_nodes", type=int, default=60)
    p.add_argument("--top_k_parents", type=int, default=8)
    p.add_argument("--weight_scale", type=float, default=3.0)
    p.add_argument("--noise_sigma", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--allow_cpu_fallback", action="store_true",
                   help="fall back to CPU")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def resolve_device(args) -> str:
    """Fail loudly on a missing GPU rather than silently running for days."""
    if not args.device.startswith("cuda"):
        return args.device
    try:
        import torch
    except ImportError:
        raise SystemExit("--device cuda but torch is not installed")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.device_count()} visible)", flush=True)
        return args.device
    msg = (f"--device {args.device} requested but torch.cuda.is_available() is "
           f"False (CUDA_VISIBLE_DEVICES="
           f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r})")
    if not args.allow_cpu_fallback:
        raise SystemExit(msg + "\nPass --allow_cpu_fallback to run on CPU anyway.")
    print(f"WARNING: {msg}\nWARNING: falling back to CPU.", flush=True)
    return "cpu"


def main():
    args = parse_args()
    device = resolve_device(args)

    if args.prior in ("tabpfn", "pfn"):
        try:
            import tabpfn  # noqa: F401
        except ImportError:
            raise SystemExit(
                "prior=tabpfn requires the tabpfn package (`pip install tabpfn`); "
                "or pass --prior zero for the E.4 ablation"
            )

    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - {"baseline", "oracle"}
    if unknown:
        raise SystemExit(f"unknown variant(s) {sorted(unknown)}")

    jobs = [(v, g, r) for v in variants for g in graphs
            for r in range(1, args.runs + 1)]
    print(f"{len(jobs)} run(s): {variants} x {graphs} x runs 1..{args.runs}, "
          f"prior={args.prior} device={device}", flush=True)

    failures = []
    for i, (variant, graph, run) in enumerate(jobs, 1):
        cmd = [
            sys.executable, SCRIPT,
            "--graph_type", graph,
            "--seeds_replicate", str(args.base_seed + run - 1),
            "--n_observational", str(args.n_observational),
            "--n_trials", str(args.n_trials),
            "--n_int", str(args.n_int),
            "--run_num", str(run),
            "--acquisition", args.acquisition,
            "--prior", args.prior,
            "--prior_mean", args.prior_mean,
            "--device", device,
            "--noiseless",
        ]
        if variant == "oracle":
            cmd.append("--oracle")
        if args.no_adapt_geometry:
            cmd.append("--no_adapt_geometry")
        if args.nonlinear:
            cmd.append("--nonlinear")
        if args.target is not None:
            cmd += ["--target", args.target]
        if graph == "gwps":
            cmd += [
                "--max_nodes", str(args.max_nodes),
                "--top_k_parents", str(args.top_k_parents),
                "--weight_scale", str(args.weight_scale),
                "--noise_sigma", str(args.noise_sigma),
            ]

        print(f"\n===== [{i}/{len(jobs)}] {variant} {graph} run{run} =====",
              flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd)
        print(f"----- {variant} {graph} run{run}: rc={rc} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if rc != 0:
            failures.append((variant, graph, run, rc))

    if failures:
        print(f"\n{len(failures)}/{len(jobs)} run(s) FAILED:", flush=True)
        for variant, graph, run, rc in failures:
            print(f"  {variant} {graph} run{run} (rc={rc})", flush=True)
        sys.exit(1)
    print(f"\nAll {len(jobs)} run(s) completed.", flush=True)


if __name__ == "__main__":
    main()
