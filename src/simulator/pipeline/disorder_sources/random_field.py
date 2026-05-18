"""
`GaussianRandomFieldSource` — the most general practical disorder source.

A Gaussian Markov random field (GMRF) on a 2D grid: a Gaussian process with
stationary covariance kernel, equivalently a sparse-precision MRF on the
device's spatial lattice (Lindgren–Rue 2011).

Setting `kernel="white"` recovers IID Gaussian noise. Setting `kernel="rbf"`
or `"matern{12,32,52}"` gives a spatially smooth field with a tunable
correlation length. The covariance kernel is the *only* knob on disorder
spatial structure that matters at the prior — strict subset of every
parametric model the DT might encounter at training time.

Sampling is done in Fourier space (FFT), so cost is O(N² log N) per draw
for an N×N grid. Sub-ms for 346×346.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from simulator.pipeline.disorder_sources.base import DisorderSource

KernelName = Literal["white", "rbf", "matern12", "matern32", "matern52"]


class GaussianRandomFieldSource(DisorderSource):
    """
    Gaussian random field disorder source. A Gaussian MRF on the device
    spatial lattice.

    Parameters
    ----------
    grid_shape : (nx, ny)
        Shape of the dense disorder field; must match the device's spatial
        grid (the runner enforces this).
    physical_width_nm : float
        Physical width of the grid in nm. Sets the pixel scale used by the
        correlation length.
    amplitude_mV : float
        Target standard deviation of the field. Sample is rescaled to hit
        this after FFT.
    correlation_length_nm : float
        Characteristic spatial length over which disorder values are
        correlated. Ignored when `kernel == "white"`. Set to ~1 pixel for
        near-white; set to a substantial fraction of `physical_width_nm`
        for smoothly varying disorder.
    kernel : one of {"white", "rbf", "matern12", "matern32", "matern52"}
        Covariance kernel.
          - "white"     : C(r) = σ² δ(r)              (recovers IID noise)
          - "rbf"       : C(r) = σ² exp(-r²/(2ℓ²))   (squared exponential / SE)
          - "matern12"  : C(r) = σ² exp(-r/ℓ)         (Matérn ν=1/2, exponential)
          - "matern32"  : Matérn ν=3/2 (smoother than 1/2)
          - "matern52"  : Matérn ν=5/2 (smoother still)
    """

    def __init__(
        self,
        *,
        grid_shape: tuple[int, int],
        physical_width_nm: float,
        amplitude_mV: float = 5.0,
        correlation_length_nm: float = 40.0,
        kernel: KernelName = "rbf",
    ) -> None:
        if kernel not in ("white", "rbf", "matern12", "matern32", "matern52"):
            raise ValueError(f"unknown kernel {kernel!r}")
        if amplitude_mV <= 0:
            raise ValueError(f"amplitude_mV must be > 0; got {amplitude_mV}")
        if correlation_length_nm <= 0 and kernel != "white":
            raise ValueError(
                f"correlation_length_nm must be > 0 for kernel={kernel!r}; "
                f"got {correlation_length_nm}"
            )

        self._shape = (int(grid_shape[0]), int(grid_shape[1]))
        self._physical_width_nm = float(physical_width_nm)
        self._amplitude_mV = float(amplitude_mV)
        self._correlation_length_nm = float(correlation_length_nm)
        self._kernel = kernel

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self._shape

    def _power_spectrum(self) -> np.ndarray:
        """
        Return the (unnormalized) power spectrum P(k) on the grid's FFT
        frequencies. Field std is enforced post-hoc, so an overall constant
        in P(k) doesn't matter — only its k-shape does.
        """
        nx, ny = self._shape
        # Pixel size in nm
        dx = self._physical_width_nm / nx
        # Angular frequency grid (rad / nm)
        kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(ny, d=dx) * 2.0 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k2 = KX ** 2 + KY ** 2
        L = self._correlation_length_nm

        if self._kernel == "white":
            return np.ones_like(k2)
        if self._kernel == "rbf":
            return np.exp(-0.5 * k2 * L * L)
        if self._kernel == "matern12":
            return (1.0 + k2 * L * L) ** (-1.5)
        if self._kernel == "matern32":
            return (3.0 / (L * L) + k2) ** (-2.5)
        if self._kernel == "matern52":
            return (5.0 / (L * L) + k2) ** (-3.5)
        raise AssertionError(f"unreachable kernel {self._kernel!r}")

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one (nx, ny) field, mean-zero, std = amplitude_mV."""
        nx, ny = self._shape
        Pk = self._power_spectrum()

        # Complex white noise in Fourier space
        re = rng.standard_normal((nx, ny))
        im = rng.standard_normal((nx, ny))
        noise_k = re + 1j * im

        field_k = noise_k * np.sqrt(Pk)
        field = np.fft.ifft2(field_k).real

        # Normalize to target std (the sample's empirical std varies sample-to-
        # sample for finite N; this enforces the requested amplitude exactly).
        s = field.std()
        if s > 0:
            field = field * (self._amplitude_mV / s)
        field = field - field.mean()
        return field.astype(np.float64)
