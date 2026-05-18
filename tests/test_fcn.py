"""Tests for the FullyConvolutionalSCModel."""

import pytest

torch = pytest.importorskip("torch")

from simulator.deep_learning.fcn import FullyConvolutionalSCModel, ResidualBlock


def test_default_output_shape() -> None:
    """Output is the same shape as input (resolution-preserving)."""
    model = FullyConvolutionalSCModel()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == x.shape


def test_handles_oxford_grid() -> None:
    """Forward pass on the actual Oxford spatial grid (330, 500)."""
    model = FullyConvolutionalSCModel()
    x = torch.randn(1, 1, 330, 500)
    y = model(x)
    assert y.shape == x.shape


def test_paper_default_parameter_count() -> None:
    """The paper's defaults give a small (~50-100k param) network."""
    model = FullyConvolutionalSCModel(n_blocks=5, channels=32)
    n = model.n_parameters
    # 5 blocks * (2 convs * (3*3*32*32 + 32)) + initial conv + final conv
    # ~2 * 9248 * 5 + (1*32*9 + 32) + (32*1*1 + 1) = ~95k
    assert 80_000 < n < 110_000, f"parameter count {n} not in expected range"


def test_residual_block_chains_when_n_convs_is_2() -> None:
    """With n_convs=2, both convs are in the computation graph."""
    block = ResidualBlock(channels=8, n_convs=2)
    x = torch.randn(1, 8, 16, 16)
    y = block(x)
    assert y.shape == x.shape
    # Both convs should affect output: zero out the SECOND conv's weights;
    # output should change.
    with torch.no_grad():
        y_before = block(x).clone()
        block.convs[1].weight.zero_(); block.convs[1].bias.zero_()
        y_after = block(x)
    assert not torch.allclose(y_before, y_after), (
        "zeroing second conv had no effect — block is not chaining convs"
    )


def test_n_convs_one_reproduces_qxcl_bug() -> None:
    """n_convs=1 gives the buggy qxcl behaviour (one Conv2D per block)."""
    block_buggy = ResidualBlock(channels=8, n_convs=1)
    block_fixed = ResidualBlock(channels=8, n_convs=2)
    assert sum(p.numel() for p in block_fixed.parameters()) > sum(
        p.numel() for p in block_buggy.parameters()
    )
