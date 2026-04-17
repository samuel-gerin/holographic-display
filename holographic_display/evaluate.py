"""
evaluate.py

Loads a trained checkpoint and runs qualitative evaluation on a few
validation samples. Saves a grid of input / predicted / ground-truth
images side-by-side for visual inspection.

Usage
─────
    python evaluate.py                                # uses best_model.pt
    python evaluate.py --checkpoint checkpoints/last_model.pt
    python evaluate.py --n_samples 8
"""

import os
import argparse
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataset import HolographicDataset
from model import HolographicUNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--out_dir",    type=str, default="checkpoints")
    parser.add_argument("--n_samples",  type=int, default=4)
    parser.add_argument("--base_ch",    type=int, default=32)
    parser.add_argument("--camera_size",type=int, default=256)
    return parser.parse_args()


def _tensor_to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert [3,H,W] float tensor in ~[0,1] to uint8 RGB image [H,W,3]."""
    t = t.detach().cpu().clamp(0, 1)
    img = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def _tensor_to_phase(t: torch.Tensor) -> np.ndarray:
    """Convert [1,H,W] float tensor to float32 array [H,W] (keeps [-1,1] range)."""
    t = t.detach().cpu()
    return t[0].numpy().astype(np.float32)


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HolographicUNet(base_ch=args.base_ch).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load validation set
    val_ds = HolographicDataset(
        os.path.join(args.data_root, "val"),
        camera_size=args.camera_size,
    )
    n = min(args.n_samples, len(val_ds))

    fig = make_subplots(
        rows=n,
        cols=6,
        horizontal_spacing=0.01,
        vertical_spacing=0.04,
    )

    with torch.no_grad():
        for i in range(n):
            sample = val_ds[i]
            inp    = sample["input"].unsqueeze(0).to(device)   # [1, 6, H, W]
            gt_src = sample["source"]                           # [3, H, W]
            gt_ph  = sample["phase"]                            # [1, H, W]

            cam_8  = inp[:, :3]
            cam_10 = inp[:, 3:]

            pred_src, pred_ph = model(cam_8, cam_10)
            pred_src = pred_src.squeeze(0).cpu()   # [3, H, W]
            pred_ph  = pred_ph.squeeze(0).cpu()    # [1, H, W]

            row = i + 1
            cam8_img = _tensor_to_rgb_uint8(cam_8.squeeze(0).cpu())
            cam10_img = _tensor_to_rgb_uint8(cam_10.squeeze(0).cpu())
            pred_src_img = _tensor_to_rgb_uint8(pred_src)
            gt_src_img = _tensor_to_rgb_uint8(gt_src)
            pred_ph_z = _tensor_to_phase(pred_ph)
            gt_ph_z = _tensor_to_phase(gt_ph)

            fig.add_trace(go.Image(z=cam8_img), row=row, col=1)
            fig.add_trace(go.Image(z=cam10_img), row=row, col=2)
            fig.add_trace(go.Image(z=pred_src_img), row=row, col=3)
            fig.add_trace(go.Image(z=gt_src_img), row=row, col=4)
            fig.add_trace(
                go.Heatmap(z=pred_ph_z, colorscale="RdBu", zmin=-1, zmax=1, showscale=False),
                row=row,
                col=5,
            )
            fig.add_trace(
                go.Heatmap(z=gt_ph_z, colorscale="RdBu", zmin=-1, zmax=1, showscale=False),
                row=row,
                col=6,
            )

    fig.update_layout(
        title_text="Holographic U-Net — Qualitative Evaluation",
        margin=dict(l=10, r=10, t=50, b=10),
        height=max(300, 260 * n),
        width=1600,
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, "evaluation.png")
    try:
        fig.write_image(out_png, scale=2)
        print(f"Saved evaluation PNG to {out_png}")
    except Exception as e:
        raise RuntimeError(
            "Failed to write evaluation PNG. Install kaleido with 'poetry add kaleido'. "
            f"Original error: {e}"
        )


if __name__ == "__main__":
    main()