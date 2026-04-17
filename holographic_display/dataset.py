"""
dataset.py

PyTorch Dataset and DataLoader factory for the holographic display dataset.

Each sample folder contains:
    source.pt   – RGB image  [H, W, 3]  float32  in [0, 1]
    phase.pt    – phase map  [H, W]     float32  in [-1, 1]
    cam_8cm.pt  – camera at  8 cm [H, W, 3] float32
    cam_10cm.pt – camera at 10 cm [H, W, 3] float32

The network input is the two camera images concatenated along the
channel dimension after permuting to [C, H, W]:
    input  → [6, H_cam, W_cam]   (cam_8cm + cam_10cm)

The network outputs are:
    source → [3, H_src, W_src]
    phase  → [1, H_slm, W_slm]

Camera images are optionally downsampled to CAMERA_SIZE to reduce memory.
"""

import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


CAMERA_SIZE = 256   # downsample 1024 → 256 before feeding the network


class HolographicDataset(Dataset):
    """
    Loads pre-generated holographic samples from disk.

    Args:
        root:        Path to either the train or val split directory.
        camera_size: Spatial size to which camera images are resized.
                     Set to None to keep original resolution.
    """

    def __init__(self, root: str, camera_size: int | None = CAMERA_SIZE):
        self.root        = root
        self.camera_size = camera_size
        # Collect all sample subdirectories, sorted for reproducibility
        self.samples = sorted(
            [os.path.join(root, d) for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d))]
        )
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        folder = self.samples[idx]

        # Load tensors — shapes as saved by generate_dataset.py
        source  = torch.load(os.path.join(folder, "source.pt"),  weights_only=True)  # [H, W, 3]
        phase   = torch.load(os.path.join(folder, "phase.pt"),   weights_only=True)  # [H, W]
        cam_8   = torch.load(os.path.join(folder, "cam_8cm.pt"), weights_only=True)  # [H, W, 3]
        cam_10  = torch.load(os.path.join(folder, "cam_10cm.pt"),weights_only=True)  # [H, W, 3]

        # Permute spatial tensors from [H, W, C] → [C, H, W] for PyTorch convolutions
        source = source.permute(2, 0, 1)   # [3, H_src, W_src]
        cam_8  = cam_8.permute(2, 0, 1)    # [3, H_cam, W_cam]
        cam_10 = cam_10.permute(2, 0, 1)   # [3, H_cam, W_cam]
        phase  = phase.unsqueeze(0)        # [1, H_slm, W_slm]

        # Optionally resize camera images
        if self.camera_size is not None:
            size = (self.camera_size, self.camera_size)
            cam_8  = F.interpolate(cam_8.unsqueeze(0),  size=size, mode="bilinear", align_corners=False).squeeze(0)
            cam_10 = F.interpolate(cam_10.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)

        # Concatenate the two camera views along the channel axis → [6, H, W]
        network_input = torch.cat([cam_8, cam_10], dim=0)

        return {
            "input":  network_input,  # [6, camera_size, camera_size]
            "source": source,         # [3, 256, 256]
            "phase":  phase,          # [1, 256, 256]
        }


def make_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 2,
    camera_size: int | None = CAMERA_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).

    Args:
        data_root:   Root data directory containing train/ and val/ splits.
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes (set 0 for debugging).
        camera_size: Resize camera images to this spatial size.
    """
    train_ds = HolographicDataset(os.path.join(data_root, "train"), camera_size=camera_size)
    val_ds   = HolographicDataset(os.path.join(data_root, "val"),   camera_size=camera_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train samples : {len(train_ds)}  |  Val samples : {len(val_ds)}")
    print(f"Batch size    : {batch_size}")
    print(f"Camera size   : {camera_size}")
    return train_loader, val_loader