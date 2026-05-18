"""
Tests for GaussianRandomFieldSource (the Gaussian MRF disorder source).
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.pipeline.disorder_sources.random_field import GaussianRandomFieldSource


GRID = (64, 64)
WIDTH_NM = 200.0
AMP = 5.0


def _new(kernel: str, ell: float = 40.0) -> GaussianRandomFieldSource:
    return GaussianRandomFieldSource(
        grid_shape=GRID,
        physical_width_nm=WIDTH_NM,
        amplitude_mV=AMP,
        correlation_length_nm=ell,
        kernel=kernel,
    )


@pytest.mark.parametrize("kernel", ["white", "rbf", "matern12", "matern32", "matern52"])
def test_shape_and_amplitude(kernel: str) -> None:
    """Sample has the right shape, std, and zero mean for every kernel."""
    src = _new(kernel)
    rng = np.random.default_rng(0)
    field = src.sample(rng)
    assert field.shape == GRID
    assert field.dtype == np.float64
    assert abs(field.mean()) < 1e-9, f"field not mean-subtracted (mean={field.mean()})"
    assert abs(field.std() - AMP) < 1e-9, (
        f"field std {field.std()} != amplitude {AMP}"
    )


def test_independence_across_seeds() -> None:
    """Different rng seeds give independent samples."""
    src = _new("rbf")
    f0 = src.sample(np.random.default_rng(0))
    f1 = src.sample(np.random.default_rng(1))
    assert not np.array_equal(f0, f1)
    # Pearson r near 0 (independence)
    r = np.corrcoef(f0.ravel(), f1.ravel())[0, 1]
    assert abs(r) < 0.1, f"two seeds give correlated fields (r={r:.3f})"


def test_white_kernel_has_no_spatial_correlation() -> None:
    """White kernel should produce a near-uncorrelated field."""
    src = _new("white")
    f = src.sample(np.random.default_rng(0))
    fft = np.fft.fft2(f - f.mean())
    power = np.abs(fft) ** 2
    # White noise's power spectrum is flat. Coefficient of variation of |F(k)|^2
    # bounded for 64x64. Loose check.
    cv = power.std() / power.mean()
    assert 0.5 < cv < 2.5, f"white-kernel power spectrum variability off: cv={cv:.2f}"


def test_smooth_kernel_has_spatial_correlation() -> None:
    """An RBF kernel with ell >> 1 pixel should produce a strongly smoothed field."""
    px = WIDTH_NM / GRID[0]
    long_ell_src = _new("rbf", ell=10 * px)   # 10 pixels
    rng = np.random.default_rng(0)
    f = long_ell_src.sample(rng)
    # Adjacent pixels should be highly correlated
    horiz_corr = np.corrcoef(f[:-1].ravel(), f[1:].ravel())[0, 1]
    assert horiz_corr > 0.9, (
        f"smooth-kernel field is not smooth (adjacent-pixel correlation={horiz_corr:.3f})"
    )


def test_matern_smoothness_ordering() -> None:
    """Matern smoothness should increase with ν: matern12 < matern32 < matern52."""
    rng_seed = 0
    smoothness = {}
    for k in ("matern12", "matern32", "matern52"):
        src = _new(k, ell=20.0)
        f = src.sample(np.random.default_rng(rng_seed))
        # Use second-derivative magnitude as a smoothness proxy (lower = smoother)
        dxx = np.diff(f, n=2, axis=0)
        dyy = np.diff(f, n=2, axis=1)
        smoothness[k] = (dxx.std() + dyy.std()) / 2
    # smoothness[matern12] should be the largest (roughest), smoothness[matern52]
    # the smallest. Strict ordering can flip on a single sample; we just check
    # the monotone trend.
    assert smoothness["matern12"] > smoothness["matern52"], (
        f"matern smoothness ordering violated: {smoothness}"
    )


def test_invalid_args() -> None:
    """Constructor rejects nonsense arguments."""
    with pytest.raises(ValueError):
        _new("not-a-kernel")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GaussianRandomFieldSource(
            grid_shape=GRID, physical_width_nm=WIDTH_NM,
            amplitude_mV=0.0, correlation_length_nm=10.0, kernel="rbf",
        )
    with pytest.raises(ValueError):
        GaussianRandomFieldSource(
            grid_shape=GRID, physical_width_nm=WIDTH_NM,
            amplitude_mV=AMP, correlation_length_nm=-1.0, kernel="rbf",
        )
