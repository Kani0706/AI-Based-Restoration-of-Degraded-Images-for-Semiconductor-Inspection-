# ============================================================
# NAFNetSR Model
# Semiconductor Image Restoration
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. LayerNorm2d
# ============================================================

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(1, channels, 1, 1)
        )

        self.bias = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

        self.eps = eps

    def forward(self, x):

        mean = x.mean(
            dim=1,
            keepdim=True
        )

        var = (
            x - mean
        ).pow(2).mean(
            dim=1,
            keepdim=True
        )

        x = (
            x - mean
        ) / torch.sqrt(
            var + self.eps
        )

        return (
            x * self.weight
            + self.bias
        )


# ============================================================
# 2. SimpleGate
# ============================================================

class SimpleGate(nn.Module):

    def forward(self, x):

        x1, x2 = x.chunk(
            2,
            dim=1
        )

        return x1 * x2


# ============================================================
# 3. Simple Channel Attention
# ============================================================

class SimpleChannelAttention(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        y = self.avg_pool(x)

        y = self.conv(y)

        return x * y


# ============================================================
# 4. NAF Block
# ============================================================

class NAFBlock(nn.Module):

    def __init__(
        self,
        channels,
        dw_expand=2,
        ffn_expand=2
    ):

        super().__init__()

        dw_channels = channels * dw_expand

        ffn_channels = channels * ffn_expand

        # ----------------------------------------------------
        # First branch
        # ----------------------------------------------------

        self.norm1 = LayerNorm2d(
            channels
        )

        self.conv1 = nn.Conv2d(
            channels,
            dw_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )

        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True
        )

        self.simple_gate = SimpleGate()

        self.sca = SimpleChannelAttention(
            channels
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            padding=0,
            bias=True
        )

        self.beta = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

        # ----------------------------------------------------
        # FFN branch
        # ----------------------------------------------------

        self.norm2 = LayerNorm2d(
            channels
        )

        self.conv3 = nn.Conv2d(
            channels,
            ffn_channels * 2,
            kernel_size=1,
            padding=0,
            bias=True
        )

        self.dwconv2 = nn.Conv2d(
            ffn_channels * 2,
            ffn_channels * 2,
            kernel_size=3,
            padding=1,
            groups=ffn_channels * 2,
            bias=True
        )

        self.simple_gate2 = SimpleGate()

        self.conv4 = nn.Conv2d(
            ffn_channels,
            channels,
            kernel_size=1,
            padding=0,
            bias=True
        )

        self.gamma = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

    def forward(self, x):

        # ====================================================
        # First NAF branch
        # ====================================================

        y = self.norm1(x)

        y = self.conv1(y)

        y = self.dwconv(y)

        y = self.simple_gate(y)

        y = self.sca(y)

        y = self.conv2(y)

        x = x + self.beta * y

        # ====================================================
        # FFN branch
        # ====================================================

        y = self.norm2(x)

        y = self.conv3(y)

        y = self.dwconv2(y)

        y = self.simple_gate2(y)

        y = self.conv4(y)

        x = x + self.gamma * y

        return x


# ============================================================
# 5. NAFNet Super-Resolution / Restoration Model
# ============================================================

class NAFNetSR(nn.Module):

    def __init__(
        self,
        img_channel=1,
        width=32,
        enc_blocks=4,
        mid_blocks=4,
        dec_blocks=4
    ):

        super().__init__()

        # ====================================================
        # Input convolution
        # ====================================================

        self.intro = nn.Conv2d(
            img_channel,
            width,
            kernel_size=3,
            padding=1
        )

        # ====================================================
        # Encoder
        # ====================================================

        self.encoder = nn.Sequential(
            *[
                NAFBlock(width)
                for _ in range(enc_blocks)
            ]
        )

        # ====================================================
        # Middle / Bottleneck
        # ====================================================

        self.middle = nn.Sequential(
            *[
                NAFBlock(width)
                for _ in range(mid_blocks)
            ]
        )

        # ====================================================
        # 2× Upsampling
        # ====================================================

        self.up = nn.Sequential(

            nn.Conv2d(
                width,
                width * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2)
        )

        # ====================================================
        # Decoder
        # ====================================================

        self.decoder = nn.Sequential(
            *[
                NAFBlock(width)
                for _ in range(dec_blocks)
            ]
        )

        # ====================================================
        # Output convolution
        # ====================================================

        self.ending = nn.Conv2d(
            width,
            img_channel,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # ====================================================
        # Base image
        # 128×128 → 256×256
        # ====================================================

        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        # ====================================================
        # Input features
        # ====================================================

        x = self.intro(x)

        # ====================================================
        # Encoder
        # ====================================================

        x = self.encoder(x)

        # ====================================================
        # Middle
        # ====================================================

        x = self.middle(x)

        # ====================================================
        # 2× Upsampling
        # ====================================================

        x = self.up(x)

        # ====================================================
        # Decoder
        # ====================================================

        x = self.decoder(x)

        # ====================================================
        # Reconstruction
        # ====================================================

        x = self.ending(x)

        # ====================================================
        # Residual reconstruction
        # ====================================================

        x = x + base

        return x
