"""
Fully-convolutional residual network for the self-consistent-potential
deep-learning node.

Ported verbatim from qxcl's `FullyConvolutionalModel` (Keras), which itself
implements `ℱ_U` from Craig et al. 2021 (arXiv:2111.11285):

    Initial Conv2D(1 -> 32, kernel 3, tanh)
    ↓
    5× residual blocks (each: 2× Conv2D(32 -> 32, kernel 3, ReLU) + skip)
    ↓
    Final Conv2D(32 -> 1, kernel 1, tanh)

Default hyperparameters match the paper:
  - n_blocks       = 5
  - channels       = 32
  - kernel_size    = 3
  - dilation       = 1

Resolution-preserving throughout (same-padding everywhere).

Bug note (vs upstream qxcl)
---------------------------
The original qxcl `conv2d_residual_block` has a typo on the second Conv2D:
    x = Conv2D(...)(y)   # second conv takes y instead of x
which silently discards the first Conv2D's output (only one Conv2D per
block effectively runs). We implement the **intended** behaviour here
(chained Conv2D layers inside each block) — this is what makes it a real
two-layer residual block. Revert to single-conv blocks by passing
`n_convs_per_block=1` if you want byte-for-byte agreement with the qxcl
code path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Resolution-preserving residual block.

    Default: 2 chained Conv2D layers + skip connection (the proper-ResNet
    interpretation, what the paper / qxcl docstring intends).
    Set `n_convs=1` to reproduce the qxcl bug (single Conv2D + skip).
    """

    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 3,
        dilation: int = 1,
        n_convs: int = 2,
    ) -> None:
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size, padding=pad, dilation=dilation)
                for _ in range(n_convs)
            ]
        )
        self.inner_activation = nn.ReLU()
        self.post_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for conv in self.convs:
            y = self.inner_activation(conv(y))
        return self.post_activation(x + y)


class FullyConvolutionalSCModel(nn.Module):
    """
    The SC-potential predictor `ℱ_U` (Craig et al. 2021, ported from qxcl).

    Input  : (B, 1, H, W) electrostatic potential
    Output : (B, 1, H, W) self-consistent potential (normalised; the
                          DeepLearningBaseNode's preprocess/postprocess
                          handle the rescaling back to mV)

    Parameters match the paper's defaults; override at instantiation if
    you want to scan architecture.
    """

    def __init__(
        self,
        n_blocks: int = 5,
        channels: int = 32,
        kernel_size: int = 3,
        dilation: int = 1,
        n_convs_per_block: int = 2,
    ) -> None:
        super().__init__()
        self.initial_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.initial_activation = nn.Tanh()
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    n_convs=n_convs_per_block,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.initial_activation(self.initial_conv(x))
        for block in self.blocks:
            y = block(y)
        return self.final_activation(self.final_conv(y))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
