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
import torch
from tqdm.auto import trange

from holographic_display.forward_model import propagate
from holographic_display.constants import nm, um, mm

# ── Physical parameters (keep identical to notebook) ──────────────────────────
NX_CAMERA   = NY_CAMERA   = 1024
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


def sample_dir(split: str, idx: int) -> str:
    """Returns e.g.  data/train/sample_0042"""
    folder = "train" if idx < N_TRAIN else "val"
    return os.path.join(ROOT, folder, f"sample_{idx:04d}")


def generate_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Random RGB source in [0, 1]  shape [NY_SOURCE, NX_SOURCE, 3]
    Random phase image in [-1, 1] shape [NY_SLM, NX_SLM]
    """
    source = torch.rand(NY_SOURCE, NX_SOURCE, 3, dtype=torch.float32)
    phase  = torch.empty(NY_SLM, NX_SLM, dtype=torch.float32).uniform_(-1.0, 1.0)
    return source, phase


def main():
    torch.set_default_device(DEVICE)

    print(f"Generating {N_TOTAL} samples  ({N_TRAIN} train / {N_VAL} val)")
    print(f"Device : {DEVICE}")
    print(f"Source : {NX_SOURCE}×{NY_SOURCE}  |  SLM: {NX_SLM}×{NY_SLM}")
    print(f"Saving to: {os.path.abspath(ROOT)}\n")

    os.makedirs(ROOT, exist_ok=True)

    for i in trange(N_TOTAL, desc="Generating samples"):
        source, phase = generate_inputs()

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
            NX_CAMERA,
            NY_CAMERA,
        )

        cam_8cm  = results[0].cpu()     # [NY_CAMERA, NX_CAMERA, 3]
        cam_10cm = results[1].cpu()     # [NY_CAMERA, NX_CAMERA, 3]

        # Save all four tensors for this sample
        out_dir = sample_dir("train" if i < N_TRAIN else "val", i)
        os.makedirs(out_dir, exist_ok=True)

        torch.save(source.cpu(),  os.path.join(out_dir, "source.pt"))
        torch.save(phase.cpu(),   os.path.join(out_dir, "phase.pt"))
        torch.save(cam_8cm,       os.path.join(out_dir, "cam_8cm.pt"))
        torch.save(cam_10cm,      os.path.join(out_dir, "cam_10cm.pt"))

    print(f"\nDone. Dataset saved to {os.path.abspath(ROOT)}")


if __name__ == "__main__":
    main()