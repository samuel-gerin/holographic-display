"""debug_dataset.py

Quick dataset sanity checks:
- shape/range/mean/std for input/source/phase
- pairwise distances to detect duplicates

Usage:
    python holographic_display/debug_dataset.py --data_root holographic_display/data --split train --n 8
    python holographic_display/debug_dataset.py --data_root holographic_display/data_overfit --split train --n 4
"""

import os
import argparse
import torch

from dataset import HolographicDataset


def _stats(t: torch.Tensor) -> dict[str, float | tuple[int, ...]]:
    t = t.detach().cpu().float()
    return {
        "shape": tuple(t.shape),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Debug holographic dataset")
    p.add_argument("--data_root", type=str, default="holographic_display/data")
    p.add_argument("--split", type=str, choices=["train", "val"], default="train")
    p.add_argument("--n", type=int, default=8, help="Number of samples to inspect")
    p.add_argument("--camera_size", type=int, default=256)
    args = p.parse_args()

    split_root = os.path.join(args.data_root, args.split)
    ds = HolographicDataset(split_root, camera_size=args.camera_size)

    n = min(args.n, len(ds))
    samples = [ds[i] for i in range(n)]

    X = torch.stack([s["input"] for s in samples]).float().flatten(1)
    S = torch.stack([s["source"] for s in samples]).float().flatten(1)
    P = torch.stack([s["phase"] for s in samples]).float().flatten(1)

    print(f"Root        : {split_root}")
    print(f"Samples     : {len(ds)} (showing {n})")
    print(f"Camera size : {args.camera_size}")

    print("\nINPUT  stats:", _stats(X))
    print("SOURCE stats:", _stats(S))
    print("PHASE  stats:", _stats(P))

    def report(name: str, T: torch.Tensor) -> None:
        d = torch.cdist(T, T)
        off = d[d > 0]
        if off.numel() == 0:
            print(f"\n{name}: all samples identical (pairwise distance=0)")
            return
        print(f"\n{name}: pairwise L2 min(offdiag)={float(off.min()):.6g} max={float(d.max()):.6g}")

    report("INPUT", X)
    report("SOURCE", S)
    report("PHASE", P)


if __name__ == "__main__":
    main()
