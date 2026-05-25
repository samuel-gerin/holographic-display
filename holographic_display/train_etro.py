"""
train_etro.py

Runs one or more training experiments sequentially on a GPU server.
This script lives inside the repo — no cloning needed.

Usage
-----
    python train_etro.py --data_path /path/to/data_5k
    python train_etro.py --data_path /path/to/data_5k --runs 1 2
    python train_etro.py --data_path /path/to/data_5k --runs all --out_root /scratch/checkpoints
    python train_etro.py --runs all   # uses DATA_PATH constant below
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── Change this if you prefer not to pass --data_path every time ──────────────
DATA_PATH_DEFAULT = "/path/to/data_5k"   # <- edit this or always pass --data_path
# ──────────────────────────────────────────────────────────────────────────────


# ── Experiment definitions ─────────────────────────────────────────────────────
EXPERIMENTS = {
    1: {
        "name": "ssim05_noph",
        "description": "SSIM=0.5, no phase loss (baseline)",
        "args": {
            "--epochs":           "150",
            "--batch_size":       "32",
            "--num_workers":      "4",
            "--lr":               "1e-3",
            "--lambda_src":       "2.0",
            "--lambda_ph":        "0.0",
            "--ssim_weight":      "0.5",
            "--src_weight_alpha": "1.0",
        },
    },
    2: {
        "name": "mse_only_noph",
        "description": "MSE only, no SSIM, no phase loss",
        "args": {
            "--epochs":           "150",
            "--batch_size":       "32",
            "--num_workers":      "4",
            "--lr":               "1e-3",
            "--lambda_src":       "2.0",
            "--lambda_ph":        "0.0",
            "--ssim_weight":      "0.0",
            "--src_weight_alpha": "1.0",
        },
    },
    3: {
        "name": "ssim05_ph01",
        "description": "SSIM=0.5, phase loss=0.1",
        "args": {
            "--epochs":           "150",
            "--batch_size":       "32",
            "--num_workers":      "4",
            "--lr":               "1e-3",
            "--lambda_src":       "2.0",
            "--lambda_ph":        "0.1",
            "--ssim_weight":      "0.5",
            "--src_weight_alpha": "1.0",
        },
    },
}

# Timestamp used once at startup so all runs in a session share the same folder
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# Root of the repo (directory that contains this script)
REPO_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, streaming output directly to the terminal."""
    print(f"\n>>> {cmd}\n", flush=True)
    result = subprocess.run(cmd, shell=True, check=check)
    return result


def verify_data(data_path: str) -> None:
    """Quick sanity-check that the data folder looks correct."""
    root = Path(data_path)
    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.exists():
            sys.exit(f"ERROR: expected {split_dir} to exist but it doesn't.")
        count = sum(1 for _ in split_dir.iterdir())
        print(f"  {split}: {count} samples")


def build_out_dir(out_root: str, exp_name: str) -> str:
    """
    Build a unique output directory so nothing is ever overwritten.

    Structure:
        <out_root>/
            session_<YYYYMMDD_HHMMSS>/
                run_1_ssim05_noph/
                run_2_mse_only_noph/
                run_3_ssim05_ph01/
    """
    path = Path(out_root) / f"session_{SESSION_TS}" / exp_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def train(experiment_id: int, data_path: str, out_root: str) -> None:
    """Run a single training experiment."""
    exp  = EXPERIMENTS[experiment_id]
    name = f"run_{experiment_id}_{exp['name']}"
    out_dir = build_out_dir(out_root, name)

    print(f"\n{'='*60}")
    print(f"  Experiment {experiment_id}: {exp['description']}")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*60}\n", flush=True)

    # Build CLI argument string (one flag=value per line for readability in logs)
    extra_args = " \\\n  ".join(f"{k} {v}" for k, v in exp["args"].items())

    cmd = (
        f"poetry run python holographic_display/train.py \\\n"
        f"  --data_root {data_path} \\\n"
        f"  --out_dir   {out_dir} \\\n"
        f"  {extra_args}"
    )

    run_cmd(cmd)
    print(f"\nExperiment {experiment_id} done. Output: {out_dir}", flush=True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run holographic training experiments on a GPU server."
    )
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH_DEFAULT,
        help="Path to the data_5k folder (default: DATA_PATH_DEFAULT constant)"
    )
    parser.add_argument(
        "--runs", nargs="+", default=["all"],
        help="Which experiments to run: 1 2 3  or  all  (default: all)"
    )
    parser.add_argument(
        "--out_root", type=str, default="checkpoints",
        help="Root directory for all checkpoint outputs (default: ./checkpoints)"
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    # Resolve experiment IDs
    if args.runs == ["all"]:
        run_ids = list(EXPERIMENTS.keys())
    else:
        try:
            run_ids = [int(r) for r in args.runs]
        except ValueError:
            sys.exit("ERROR: --runs expects integers or 'all', e.g. --runs 1 2 3")

    for rid in run_ids:
        if rid not in EXPERIMENTS:
            sys.exit(f"ERROR: experiment {rid} is not defined. Choose from {list(EXPERIMENTS.keys())}")

    print(f"\nSession timestamp : {SESSION_TS}")
    print(f"Data path         : {args.data_path}")
    print(f"Output root       : {args.out_root}/session_{SESSION_TS}/")
    print(f"Experiments queued: {run_ids}")
    for rid in run_ids:
        print(f"  [{rid}] {EXPERIMENTS[rid]['description']}")

    print(f"\nVerifying data at {args.data_path}...")
    verify_data(args.data_path)

    # Run experiments one by one
    failed = []
    for rid in run_ids:
        try:
            train(rid, args.data_path, args.out_root)
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: experiment {rid} failed (exit code {e.returncode}). Continuing...", flush=True)
            failed.append(rid)

    # Final summary
    session_dir = Path(args.out_root) / f"session_{SESSION_TS}"
    print(f"\n{'='*60}")
    print("Session complete.")
    print(f"All outputs in: {session_dir}")
    if failed:
        print(f"Failed experiments: {failed}")
    else:
        print("All experiments succeeded.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()