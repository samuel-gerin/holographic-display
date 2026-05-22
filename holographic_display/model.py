"""
model.py

Dual-encoder U-Net for holographic inverse problem.

Architecture overview
─────────────────────

    cam_8cm  [3, H, W] ─┐
                         ├─ shared encoder (4 downsampling stages)
    cam_10cm [3, H, W] ─┘     (two separate encoder branches)
                                        │
                              bottleneck fusion (concat + conv)
                                        │
                        ┌───────────────┴───────────────┐
                   Decoder A                        Decoder B
                (4 upsampling stages,           (4 upsampling stages,
                 skip from both encoders)        skip from both encoders)
                        │                               │
                 source head                      phase head
                 Conv → Sigmoid                   Conv (raw)
                 [3, 256, 256]                    [1, 256, 256]

Skip connections pass feature maps from *both* encoder branches at each
resolution and concatenate them into the corresponding decoder stage.

Value ranges
────────────
    source output : Sigmoid  → [0, 1]    (RGB)
    phase  output : raw conv → (-inf, inf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two Conv2d layers each followed by GroupNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling: MaxPool2d → ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upsampling: bilinear upsample → concat skip connections → ConvBlock.

    skip_ch is the total channels of the skip tensors that will be concatenated.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, *skips: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dimensions don't match perfectly
        for skip in skips:
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, *skips], dim=1)
        return self.conv(x)


# ── Encoder ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    4-stage encoder.
    Input : [B, in_ch, H, W]
    Returns the bottleneck and all intermediate feature maps (for skip connections).

    Channel progression: in_ch → 32 → 64 → 128 → 256
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.stem  = ConvBlock(in_ch,       base_ch)        # H    → H
        self.down1 = Down(base_ch,          base_ch * 2)    # H/2
        self.down2 = Down(base_ch * 2,      base_ch * 4)    # H/4
        self.down3 = Down(base_ch * 4,      base_ch * 8)    # H/8

    def forward(self, x: torch.Tensor):
        s0 = self.stem(x)   # [B, 32,  H,   W  ]
        s1 = self.down1(s0) # [B, 64,  H/2, W/2]
        s2 = self.down2(s1) # [B, 128, H/4, W/4]
        s3 = self.down3(s2) # [B, 256, H/8, W/8]
        return s3, [s0, s1, s2]  # bottleneck + skips


# ── Full model ─────────────────────────────────────────────────────────────────

class HolographicUNet(nn.Module):
    """
    Dual-encoder U-Net for holographic inverse prediction.

    Args:
        base_ch: Base number of channels in the encoder (doubles each stage).
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        B = base_ch

        # Two independent encoders (one per camera view)
        self.enc_8  = Encoder(in_ch=3, base_ch=B)
        self.enc_10 = Encoder(in_ch=3, base_ch=B)

        # Bottleneck fusion: concatenate both bottlenecks (256+256 = 512) → 256
        fuse_groups = 8 if B * 8 >= 8 else 1
        self.fuse = nn.Sequential(
            nn.Conv2d(B * 8 * 2, B * 8, kernel_size=1, bias=False),
            nn.GroupNorm(fuse_groups, B * 8),
            nn.ReLU(inplace=True),
        )

        # Skip channels at each scale: both encoders contribute → doubled
        # Skip sizes: [B*2+B*2, B*4+B*4, B*8+B*8] = [4B, 8B, 16B] going up
        # But our Up layers receive:  (from below) + (skip_ch)

        # Decoder A (source RGB)
        self.up_a3 = Up(in_ch=B * 8,  skip_ch=B * 4 * 2, out_ch=B * 4)   # H/8→H/4
        self.up_a2 = Up(in_ch=B * 4,  skip_ch=B * 2 * 2, out_ch=B * 2)   # H/4→H/2
        self.up_a1 = Up(in_ch=B * 2,  skip_ch=B     * 2, out_ch=B)        # H/2→H
        self.head_source = nn.Sequential(
            nn.Conv2d(B, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        # Decoder B (phase map)
        self.up_b3 = Up(in_ch=B * 8,  skip_ch=B * 4 * 2, out_ch=B * 4)
        self.up_b2 = Up(in_ch=B * 4,  skip_ch=B * 2 * 2, out_ch=B * 2)
        self.up_b1 = Up(in_ch=B * 2,  skip_ch=B     * 2, out_ch=B)
        self.head_phase = nn.Conv2d(B, 1, kernel_size=1)

    def forward(
        self, cam_8: torch.Tensor, cam_10: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cam_8  : [B, 3, H, W]
            cam_10 : [B, 3, H, W]
        Returns:
            source : [B, 3, H, W]  in [0, 1]
            phase  : [B, 1, H, W]  raw (unbounded)
        """
        bot_8,  skips_8  = self.enc_8(cam_8)
        bot_10, skips_10 = self.enc_10(cam_10)

        # Fuse bottlenecks
        fused = self.fuse(torch.cat([bot_8, bot_10], dim=1))  # [B, 256, H/8, W/8]

        # Merge skip connections at each scale (concatenate from both encoders)
        skip3 = torch.cat([skips_8[2], skips_10[2]], dim=1)  # [B, 8B,  H/4, W/4]
        skip2 = torch.cat([skips_8[1], skips_10[1]], dim=1)  # [B, 4B,  H/2, W/2]
        skip1 = torch.cat([skips_8[0], skips_10[0]], dim=1)  # [B, 2B,  H,   W  ]

        # Decoder A → source
        x_a = self.up_a3(fused, skip3)
        x_a = self.up_a2(x_a,   skip2)
        x_a = self.up_a1(x_a,   skip1)
        source = self.head_source(x_a)

        # Decoder B → phase
        x_b = self.up_b3(fused, skip3)
        x_b = self.up_b2(x_b,   skip2)
        x_b = self.up_b1(x_b,   skip1)
        phase = self.head_phase(x_b)

        return source, phase


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = HolographicUNet(base_ch=32)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Smoke test
    B, H = 2, 256
    cam_8  = torch.randn(B, 3, H, H)
    cam_10 = torch.randn(B, 3, H, H)
    src, ph = model(cam_8, cam_10)
    print(f"source output : {src.shape}  range [{src.min():.3f}, {src.max():.3f}]")
    print(f"phase  output : {ph.shape}   range [{ph.min():.3f}, {ph.max():.3f}]")