"""
generate_dataset.py

Generates 1000 (RGB source, phase) input pairs, runs them through the
holographic forward model, and saves the resulting dataset to disk.

Output structure
----------------
    holographic_display/
    data/
    train/
        sample_0000/
        source.pt      # RGB image  [H, W, 3]  float32  in [0, 1]
        phase.pt       # phase image [H, W]    float32  in [-1, 1]
        cam_8cm.pt     # propagated  [H, W, 3] float32
        cam_10cm.pt    # propagated  [H, W, 3] float32
        sample_0001/
        ...
    val/
        sample_0800/
        ...

Why .pt files?
--------------
torch.save / torch.load preserves dtype, shape, and value range exactly
with zero encoding loss — unlike PNG (uint8 clipping) or numpy (extra
dependency). Loading at training time is a single torch.load() call and
returns a tensor ready to use with no extra transforms.
"""

import os
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from holographic_display.forward_model import propagate
from holographic_display.constants import nm, um, mm

# ── Physical parameters (keep identical to notebook) ──────────────────────────
DEFAULT_CAMERA_SIZE = 256
DEFAULT_SIM_CAMERA_SIZE = 1024
DX_SOURCE   = 55.2 * um
DX_SLM      = 3.74 * um

RED   = 620 * nm
GREEN = 532 * nm
BLUE  = 461 * nm
REFERENCE_WAVELENGTH = 520 * nm          # updated from 460 → 520 as instructed
WAVELENGTHS = [RED, GREEN, BLUE]

Z_SOURCE_SLM      = 250 * mm
Z_OBJECT_CAMERA_1 =  80 * mm            # 8 cm
Z_OBJECT_CAMERA_2 = 100 * mm            # 10 cm
PROPAGATION_DISTANCES = [Z_OBJECT_CAMERA_1, Z_OBJECT_CAMERA_2]

# ── Dataset parameters ─────────────────────────────────────────────────────────
N_TOTAL    = 1000
N_TRAIN    = 800
N_VAL      = 200                         # N_TOTAL - N_TRAIN
DEVICE     = "cpu"                       # change to "cuda" if available

# Source and SLM image sizes used in the notebook
# The SLM phase image is [NY_SLM, NX_SLM], source RGB is [NY_SOURCE, NX_SOURCE, 3]
# Keeping them small enough to be tractable on CPU for generation.
NY_SOURCE = NX_SOURCE = 256             # spatial resolution of source RGB image
NY_SLM    = NX_SLM    = 256             # spatial resolution of SLM phase image

# ── Output directory layout ────────────────────────────────────────────────────
ROOT = "holographic_display/data"


def sample_dir(root: str, split: str, idx: int, n_train: int) -> str:
    """Returns e.g.  data/train/sample_0042"""
    folder = "train" if idx < n_train else "val"
    return os.path.join(root, folder, f"sample_{idx:04d}")


def _gaussian_kernel2d(size: int, sigma: float, device: str) -> torch.Tensor:
    ax = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


def _blur2d(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """img: [H,W], kernel: [K,K]"""
    import torch.nn.functional as F

    k = kernel.shape[0]
    pad = k // 2
    x = img[None, None]
    w = kernel[None, None]
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.conv2d(x, w)
    return y[0, 0]


def generate_inputs(mode: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (source, phase) targets.

    - mode='random': matches the original notebook behavior (i.i.d. uniform noise)
    - mode='structured': generates smoother, more learnable targets (blobs + low-pass phase)
    """
    if mode == "random":
        source = torch.rand(NY_SOURCE, NX_SOURCE, 3, dtype=torch.float32, device=device)
        phase = torch.empty(NY_SLM, NX_SLM, dtype=torch.float32, device=device).uniform_(-1.0, 1.0)
        return source, phase

    if mode != "structured":
        raise ValueError(f"Unknown mode: {mode}")

    # Structured RGB: sum of a few Gaussian blobs per channel.
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, NY_SOURCE, device=device),
        torch.linspace(-1.0, 1.0, NX_SOURCE, device=device),
        indexing="ij",
    )

    source = torch.zeros(NY_SOURCE, NX_SOURCE, 3, dtype=torch.float32, device=device)
    n_blobs = 6
    for c in range(3):
        for _ in range(n_blobs):
            cx = torch.empty((), device=device).uniform_(-0.7, 0.7)
            cy = torch.empty((), device=device).uniform_(-0.7, 0.7)
            sigma = torch.empty((), device=device).uniform_(0.05, 0.25)
            amp = torch.empty((), device=device).uniform_(0.3, 1.0)
            blob = amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
            source[..., c] += blob

    source = source / source.max().clamp_min(1e-12)
    source = source.clamp(0.0, 1.0)

    # Structured phase: low-pass filtered noise mapped to [-1, 1].
    phase = torch.randn(NY_SLM, NX_SLM, dtype=torch.float32, device=device)
    kernel = _gaussian_kernel2d(size=21, sigma=4.0, device=device)
    phase = _blur2d(phase, kernel)
    phase = phase - phase.mean()
    phase = phase / phase.abs().max().clamp_min(1e-12)
    phase = phase.clamp(-1.0, 1.0)

    return source, phase


def get_args():
    parser = argparse.ArgumentParser(description="Generate holographic dataset")
    parser.add_argument("--root", type=str, default=ROOT, help="Output root directory")
    parser.add_argument("--n_total", type=int, default=N_TOTAL)
    parser.add_argument("--n_train", type=int, default=N_TRAIN)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument(
        "--sim_camera_size",
        type=int,
        default=DEFAULT_SIM_CAMERA_SIZE,
        help="Spatial size used in forward propagation.",
    )
    parser.add_argument(
        "--camera_size",
        type=int,
        default=DEFAULT_CAMERA_SIZE,
        help="Spatial size of saved camera images.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="structured",
        choices=["random", "structured"],
        help="Target generation mode.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    return parser.parse_args()


def main():
    args = get_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    torch.set_default_device(args.device)

    n_total = args.n_total
    n_train = args.n_train
    n_val = n_total - n_train

    print(f"Generating {n_total} samples  ({n_train} train / {n_val} val)")
    print(f"Device : {args.device}")
    print(
        f"Source : {NX_SOURCE}×{NY_SOURCE}  |  SLM: {NX_SLM}×{NY_SLM}  |  "
        f"Cam: {args.sim_camera_size}×{args.sim_camera_size} -> {args.camera_size}×{args.camera_size}"
    )
    print(f"Mode   : {args.mode}")
    if args.seed is not None:
        print(f"Seed   : {args.seed}")
    print(f"Saving to: {os.path.abspath(args.root)}\n")

    os.makedirs(args.root, exist_ok=True)

    for i in trange(n_total, desc="Generating samples"):
        source, phase = generate_inputs(mode=args.mode, device=args.device)

        # Forward model: returns [cam_8cm, cam_10cm], each [NY_CAMERA, NX_CAMERA, 3]
        results = propagate(
            source,
            phase,
            WAVELENGTHS,
            REFERENCE_WAVELENGTH,
            DX_SOURCE,
            DX_SLM,
            Z_SOURCE_SLM,
            PROPAGATION_DISTANCES,
            args.sim_camera_size,
            args.sim_camera_size,
        )

        cam_8cm = results[0]
        cam_10cm = results[1]
        if args.camera_size != args.sim_camera_size:
            size = (args.camera_size, args.camera_size)
            cam_8cm = F.interpolate(
                cam_8cm.permute(2, 0, 1).unsqueeze(0),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            cam_10cm = F.interpolate(
                cam_10cm.permute(2, 0, 1).unsqueeze(0),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        cam_8cm = cam_8cm.cpu()
        cam_10cm = cam_10cm.cpu()

        # Save all four tensors for this sample
        out_dir = sample_dir(args.root, "train" if i < n_train else "val", i, n_train=n_train)
        os.makedirs(out_dir, exist_ok=True)

        torch.save(source.cpu(),  os.path.join(out_dir, "source.pt"))
        torch.save(phase.cpu(),   os.path.join(out_dir, "phase.pt"))
        torch.save(cam_8cm,       os.path.join(out_dir, "cam_8cm.pt"))
        torch.save(cam_10cm,      os.path.join(out_dir, "cam_10cm.pt"))

    print(f"\nDone. Dataset saved to {os.path.abspath(args.root)}")


if __name__ == "__main__":
    main()