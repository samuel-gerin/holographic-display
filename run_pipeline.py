#!/usr/bin/env python3
"""
run_pipeline.py

Regenerate a dataset and then train a model sequentially.
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from datetime import datetime


def _run(cmd: list[str]) -> None:
    print("\nRunning:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_data_root(root: str, force: bool) -> None:
    if not os.path.isdir(root):
        return
    if not force and os.listdir(root):
        raise RuntimeError(
            f"Data root '{root}' exists and is not empty. Use --force or pass a new --data_root."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate dataset then train")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--n_total", type=int, default=200)
    parser.add_argument("--n_train", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="structured", choices=["random", "structured"])
    parser.add_argument("--sim_camera_size", type=int, default=1024)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing data_root")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_src", type=float, default=1.0)
    parser.add_argument("--lambda_ph", type=float, default=0.2)
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    parser.add_argument("--phase_loss", type=str, default="l1", choices=["l1", "mse", "angular"])
    parser.add_argument("--src_weight_alpha", type=float, default=0.0)
    parser.add_argument("--ph_weight_alpha", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)

    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))

    if args.data_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.data_root = os.path.join(
            repo_root,
            "holographic_display",
            f"data_structured_{args.camera_size}_{stamp}",
        )

    _ensure_data_root(args.data_root, args.force)

    gen_cmd = [
        sys.executable,
        os.path.join(repo_root, "holographic_display", "generate_dataset.py"),
        "--root",
        args.data_root,
        "--n_total",
        str(args.n_total),
        "--n_train",
        str(args.n_train),
        "--device",
        args.device,
        "--mode",
        args.mode,
        "--sim_camera_size",
        str(args.sim_camera_size),
        "--camera_size",
        str(args.camera_size),
    ]
    if args.seed is not None:
        gen_cmd += ["--seed", str(args.seed)]

    train_cmd = [
        sys.executable,
        os.path.join(repo_root, "holographic_display", "train.py"),
        "--data_root",
        args.data_root,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--lambda_src",
        str(args.lambda_src),
        "--lambda_ph",
        str(args.lambda_ph),
        "--ssim_weight",
        str(args.ssim_weight),
        "--phase_loss",
        args.phase_loss,
        "--src_weight_alpha",
        str(args.src_weight_alpha),
        "--ph_weight_alpha",
        str(args.ph_weight_alpha),
        "--camera_size",
        str(args.camera_size),
        "--num_workers",
        str(args.num_workers),
        "--base_ch",
        str(args.base_ch),
    ]
    if args.out_dir is not None:
        train_cmd += ["--out_dir", args.out_dir]
    if args.max_train_batches is not None:
        train_cmd += ["--max_train_batches", str(args.max_train_batches)]
    if args.max_val_batches is not None:
        train_cmd += ["--max_val_batches", str(args.max_val_batches)]

    _run(gen_cmd)
    _run(train_cmd)


if __name__ == "__main__":
    main()
