"""
train.py

Trains the HolographicUNet to predict (source RGB, phase) from two camera images.

Usage
─────
    python train.py                        # default settings
    python train.py --epochs 50            # more epochs
    python train.py --batch_size 2         # smaller batch for low-memory GPUs
    python train.py --data_root /path/to/data

Outputs
───────
    checkpoints/best_model.pt  – weights with lowest validation loss
    checkpoints/last_model.pt  – weights after final epoch
    losses.pt                  – dict with train_losses and val_losses lists
    train_val_loss.png         – loss curves plot
"""

import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import plotly.graph_objects as go

from dataset import make_dataloaders
from model import HolographicUNet, count_parameters


# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Train Holographic U-Net")
    parser.add_argument("--data_root",   type=str,   default="data")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--base_ch",     type=int,   default=32,
                        help="Base channel count in U-Net encoder")
    parser.add_argument("--camera_size", type=int,   default=256,
                        help="Spatial size to which camera images are resized")
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Limit number of training batches per epoch (smoke test).",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Limit number of validation batches per epoch (smoke test).",
    )
    parser.add_argument("--lambda_src",  type=float, default=1.0,
                        help="Weight for source MSE loss")
    parser.add_argument("--lambda_ph",   type=float, default=1.0,
                        help="Weight for phase MSE loss")
    parser.add_argument(
        "--src_weight_alpha",
        type=float,
        default=0.0,
        help=(
            "Optional pixel-weighting for the source loss. If >0, bright pixels in the GT source "
            "are weighted higher: w = 1 + alpha * mean(GT_RGB)."
        ),
    )
    parser.add_argument(
        "--ph_weight_alpha",
        type=float,
        default=0.0,
        help=(
            "Optional pixel-weighting for the phase loss. If >0, large-magnitude GT phase pixels "
            "are weighted higher: w = 1 + alpha * |GT_phase|."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Output directory. If omitted, a unique run folder is created under 'checkpoints/' "
            "to avoid overwriting previous runs."
        ),
    )
    return parser.parse_args()


# ── Loss function ──────────────────────────────────────────────────────────────

class CombinedMSELoss(nn.Module):
    """Weighted sum of MSE losses on source and phase outputs."""

    def __init__(
        self,
        lambda_src: float = 1.0,
        lambda_ph: float = 1.0,
        src_weight_alpha: float = 0.0,
        ph_weight_alpha: float = 0.0,
    ):
        super().__init__()
        self.lambda_src = lambda_src
        self.lambda_ph  = lambda_ph
        self.src_weight_alpha = src_weight_alpha
        self.ph_weight_alpha = ph_weight_alpha

    def _weighted_mse(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        per_pixel = (pred - target) ** 2
        if weight is None:
            return per_pixel.mean()
        # weight is broadcastable to pred/target
        return (per_pixel * weight).mean()

    def forward(
        self,
        pred_src: torch.Tensor, target_src: torch.Tensor,
        pred_ph:  torch.Tensor, target_ph:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Optional weighting to discourage "predict dark everywhere" for sparse sources.
        w_src = None
        if self.src_weight_alpha and self.src_weight_alpha > 0:
            # [B,1,H,W], values roughly in [0,1]
            luminance = target_src.mean(dim=1, keepdim=True)
            w_src = 1.0 + self.src_weight_alpha * luminance

        # Optional weighting to discourage "predict ~0 phase" collapse.
        w_ph = None
        if self.ph_weight_alpha and self.ph_weight_alpha > 0:
            w_ph = 1.0 + self.ph_weight_alpha * target_ph.abs()

        loss_src = self._weighted_mse(pred_src, target_src, w_src)
        loss_ph  = self._weighted_mse(pred_ph,  target_ph,  w_ph)
        total    = self.lambda_src * loss_src + self.lambda_ph * loss_ph
        return total, loss_src, loss_ph


# ── One epoch helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool, max_batches: int | None = None):
    model.train() if train else model.eval()

    total_loss = total_src = total_ph = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inp    = batch["input"].to(device)     # [B, 6, H, W]
            target_src = batch["source"].to(device) # [B, 3, H, W]
            target_ph  = batch["phase"].to(device)  # [B, 1, H, W]

            # Split the 6-channel input back into two 3-channel views
            cam_8  = inp[:, :3]
            cam_10 = inp[:, 3:]

            pred_src, pred_ph = model(cam_8, cam_10)

            loss, l_src, l_ph = criterion(pred_src, target_src, pred_ph, target_ph)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_src  += l_src.item()
            total_ph   += l_ph.item()
            n_batches  += 1

    if n_batches == 0:
        raise RuntimeError("No batches were processed (empty loader or max_batches=0).")

    return total_loss / n_batches, total_src / n_batches, total_ph / n_batches


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_loss_plot(train_losses: list, val_losses: list, out_path: str):
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines+markers",
            marker=dict(size=8),
            name="Train loss",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_losses,
            mode="lines+markers",
            marker=dict(size=8),
            name="Val loss",
        )
    )
    fig.update_layout(
        title="Holographic U-Net – Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        width=900,
        height=500,
    )

    # Make single-epoch plots readable.
    if len(epochs) == 1:
        fig.update_xaxes(range=[0.5, 1.5], dtick=1)
    else:
        fig.update_xaxes(range=[0.5, len(epochs) + 0.5], dtick=1)

    try:
        fig.write_image(out_path, scale=2)
        print(f"Loss plot saved to {out_path}")
    except Exception as e:
        raise RuntimeError(
            "Failed to write loss plot PNG. Install kaleido with 'poetry add kaleido'. "
            f"Original error: {e}"
        )


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join("checkpoints", f"run_{stamp}")
    print(f"\n{'='*55}")
    print(f"  Holographic U-Net Training")
    print(f"{'='*55}")
    print(f"  Device      : {device}")
    print(f"  Data root   : {args.data_root}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Camera size : {args.camera_size}")
    print(f"  Out dir     : {args.out_dir}")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        data_root   = args.data_root,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        camera_size = args.camera_size,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HolographicUNet(base_ch=args.base_ch).to(device)
    print(f"\nTrainable parameters: {count_parameters(model):,}\n")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = CombinedMSELoss(
        lambda_src=args.lambda_src,
        lambda_ph=args.lambda_ph,
        src_weight_alpha=args.src_weight_alpha,
        ph_weight_alpha=args.ph_weight_alpha,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = None

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t_loss, t_src, t_ph = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            max_batches=args.max_train_batches,
        )
        v_loss, v_src, v_ph = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            train=False,
            max_batches=args.max_val_batches,
        )

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        scheduler.step(v_loss)

        # Save best model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = epoch
            best_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)

            # Keep an epoch-stamped copy so the exact best isn't lost if you train again.
            stamped_path = os.path.join(args.out_dir, f"best_model_epoch_{epoch:03d}.pt")
            torch.save(model.state_dict(), stamped_path)
            best_marker = "  ← best"
        else:
            best_marker = ""

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"| train {t_loss:.5f} (src {t_src:.5f} ph {t_ph:.5f})  "
            f"| val {v_loss:.5f} (src {v_src:.5f} ph {v_ph:.5f})"
            f"{best_marker}"
        )

    # ── Save final model & losses ──────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(args.out_dir, "last_model.pt"))

    losses_dict = {
        "train_losses": train_losses,
        "val_losses":   val_losses,
    }
    torch.save(losses_dict, os.path.join(args.out_dir, "losses.pt"))
    print(f"\nSaved losses to {os.path.join(args.out_dir, 'losses.pt')}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    save_loss_plot(
        train_losses, val_losses,
        out_path=os.path.join(args.out_dir, "train_val_loss.png"),
    )

    if best_epoch is None:
        print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")
    else:
        print(f"\nTraining complete. Best val loss: {best_val_loss:.5f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()