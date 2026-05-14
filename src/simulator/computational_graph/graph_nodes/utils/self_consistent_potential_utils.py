"""
Util functions for self-consistent potential graph nodes

Copyright © 2022 QuantrolOx Ltd
"""
#
import numpy as np
from scipy.signal import convolve  # type:ignore
import torch

from simulator.computational_graph.utils.device_config import DeviceConfig, MaterialConfig  # type: ignore

# from simulator.utils.physics_utils import fermi_dirac_dist # WIP


# pylint: disable-next=too-many-arguments
def kernel_calc(array, dx, dy, screen, device_config):
    """
    NOTE: docstring and return types
    """
    y_n, x_n = array.shape
    # Choose size of kernel based on whether screening or disorder potential
    if screen:
        xr, yr = x_n * dx, y_n * dy
    else:
        xr, yr = x_n * dx / 0.5, y_n * dy / 0.5

    x = np.arange(dx, xr, dx)
    y = np.arange(dy, yr, dy)

    x = np.concatenate([x[::-1], [dx / 2.0], x])
    y = np.concatenate([y[::-1], [dy / 2.0], y])

    mesh_x, mesh_y = np.meshgrid(x, y)
    if screen:
        kernel = coulomb(mesh_x, mesh_y, 0, 0, device_config)
    else:
        kernel = coulomb_donor(mesh_x, mesh_y, 0, 0, device_config)
    return kernel


def get_scale(potential: np.ndarray, device_config: DeviceConfig) -> np.ndarray:
    """
    Args:
        potential: 2D array of shape potential.shape (values unimportant)
        device_config: class of DeviceConfig

    Returns:
        scale: list of pixel widths [x_width, y_width]
    """
    # Determine length scales
    ny, nx = potential.shape

    # NOTE: Looks like scale only makes sense for square devices at the moment
    ly = device_config.width
    lx = device_config.width

    scale_og = np.array([lx / nx, ly / ny])  # nm

    # Average length scales based on measurements in x,y
    scale = np.array([np.mean(scale_og), np.mean(scale_og)])
    scale = scale * 1e-9  # m

    return scale


# pylint: disable-next=too-many-locals
# pylint: disable-next=too-many-arguments
def coulomb(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x: float,
    y: float,
    device_config: DeviceConfig,
) -> np.ndarray:
    """

    Args:
        x_grid: meshgrid X values
        y_grid: meshgrid Y values
        x: centre location for kernel evaluation (x)
        y: centre location for kernel evaluation (y)
        total_config: class of TotalConfig containing parameters

    Returns:
        kernel: coulomb potential from a charge in the carrier plane

    """
    # Return Coulomb kernel for a charge in the charge carrier plane
    # Permittivity
    epsilon_r = device_config.material_config.relative_permittivity
    epsilon = 8.85e-12 * epsilon_r
    # Coulomb Constant
    if device_config.material_config.atomic_units:
        k = 1 / (4 * np.pi * epsilon_r)
    else:
        k = 1 / (4 * np.pi * epsilon)
    # Peak electron density is not exactly on the interface
    displace = 5
    # Depth of 2DEG
    d = (device_config.carrier_depth + displace) * 1e-9
    # Calculate distance from point charge on x-y plane
    r = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
    r_mir = np.sqrt((2 * d) ** 2 + (x_grid - x) ** 2 + (y_grid - y) ** 2)
    # Calculate potential (Gauss' Law)
    kernel = k * (1 / r - 1 / r_mir)

    return kernel


def get_carrier_density(
    pot: np.ndarray,
    material_config: MaterialConfig,
    mu: float = 0,
    use_torch: bool = False,
) -> np.ndarray:
    """
    Estimates the carrier density from the potential using the Thomas-Fermi approximation.
    Args:
        pot: 2D potential landscape
        material_config: class of MaterialConfig containing material parameters
            .atomic_units: use atomic units True/False
        mu: Fermi level value (in eV)

    Returns:
        n: 2D number density in carrier plane

    """
    # Return carrier density using Thomas-Fermi approximation
    if material_config.atomic_units:
        h_bar, e = 1.0, 1.0
        m = material_config.effective_mass
    else:
        # Thomas Fermi local density
        h_bar = 1.05e-34  # J*s
        m = 9.11e-31 * material_config.effective_mass
        e = 1.6e-19  # To convert to J
    # Chemical potential mu, mV
    # Scale energy by 1E-3 to account for potential in mV
    energy = e * (material_config.carrier_charge * pot) * 1e-3
    # energy = potential_energy(pot, total_config=total_config) * 1e-3
    degeneracy = material_config.degeneracy

    # if material_config.temperature > 1e-3:
    #     dist = fermi_dirac_dist(energy, material_config.temperature, mu)
    # else:
    #     dist = np.heaviside(mu - energy, 1)
    dist = torch.heaviside(mu-energy, torch.ones_like(energy)) if use_torch else np.heaviside(mu - energy, 1.)

    # Number density of carriers
    n = degeneracy * (m / (np.pi * h_bar**2)) * (mu - energy) * dist

    return n


# pylint: disable-next=too-many-arguments
def coulomb_donor(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x: float,
    y: float,
    device_config: DeviceConfig,
) -> np.ndarray:
    """

    Args:
        x_grid: meshgrid X values
        y_grid: meshgrid Y values
        x: centre location for kernel evaluation (x)
        y: centre location for kernel evaluation (y)
        total_config: class of TotalConfig containing parameters

    Returns:
        kernel: coulomb potential from a donor ion

    """
    # Return Coulomb potential for a donor ion
    # Permittivity
    epsilon_r = device_config.material_config.relative_permittivity
    epsilon = 8.85e-12 * epsilon_r
    # Coulomb Constant
    k = 1 / (4 * np.pi * epsilon)
    # Depth of 2DEG
    # Peak carrier density is not exactly on the interface
    displace = 5
    d = (device_config.carrier_depth + displace) * 1e-9

    # Height of Donors relative to 2DEG (h is defined as positive if donors are above the 2DEG)
    h = d - (device_config.donor_height + displace) * 1e-9

    # Calculate distance from point charge on x-y plane
    r = np.sqrt(h**2 + (x_grid - x) ** 2 + (y_grid - y) ** 2)
    r_mir = np.sqrt((2 * d - h) ** 2 + (x_grid - x) ** 2 + (y_grid - y) ** 2)
    # Calculate potential (Gauss' Law)
    kernel = k * (1 / r - 1 / r_mir)

    return kernel


def calc_pot(gate: np.ndarray, dx: float, dy: float, d: float) -> np.ndarray:
    """
    Calculates the potential from the gate.

    Args:
        gate: 2D binary mask of gate
        dx: pixel width (x)
        dy: pixel width (y)
        d: depth of carrier plane beneath gate

    Returns:
        potential: potential from gate
    """
    # Calculate potential at depth d from a gate
    kernel = get_kernel(gate, dx, dy, d)
    potential = convolve(gate, kernel, mode="same")

    return potential


# pylint: disable-next=too-many-locals
def get_kernel(array: np.ndarray, dx: float, dy: float, d: float) -> np.ndarray:
    """
    Returns the kernel to generate gate potential.

    Args:
        array: 2D array to generate potential on
        dx: pixel width (x)
        dy: pixel width (y)
        d:  depth of from gates

    Returns:
        kernel: Laplace solution, kernel to generate gate potential

    """
    y_n, x_n = array.shape

    xr, yr = x_n * dx * 2, y_n * dy * 2

    x = np.arange(dx, xr, dx)
    y = np.arange(dy, yr, dy)

    x = np.concatenate([x[::-1], [dx / 2.0], x])
    y = np.concatenate([y[::-1], [dy / 2.0], y])

    mesh_x, mesh_y = np.meshgrid(x, y)

    # Define corners of rectangle/wire
    top, bottom = dy * 0.5, -dy * 0.5
    left, right = -dx * 0.5, dx * 0.5

    def g(u: np.ndarray, v: np.ndarray, d: float) -> np.ndarray:
        # g function as defined in Davies paper:
        # Modelling the patterned two-dimensional electron gas, 1995, eq 3.12
        r = np.sqrt(u**2 + v**2 + d**2)
        frac = (u * v) / (d * r)
        value = (1 / (2 * np.pi)) * np.arctan(frac)

        return value

    # Calculate potential
    kernel = (
        g(mesh_x - bottom, mesh_y - left, d)
        + g(mesh_x - bottom, right - mesh_y, d)
        + g(top - mesh_x, mesh_y - left, d)
        + g(top - mesh_x, right - mesh_y, d)
    )

    return kernel * 10000


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def fftconvolve2d(x, y, mode="full"):
    x = x.float().contiguous()
    y = y.float().contiguous()

    H, W = x.shape[-2:]
    Kh, Kw = y.shape[-2:]

    out_h = H + Kh - 1
    out_w = W + Kw - 1

    # Optional but recommended
    fft_h = next_pow2(out_h)
    fft_w = next_pow2(out_w)

    X = torch.fft.rfft2(x, s=(fft_h, fft_w))
    Y = torch.fft.rfft2(y, s=(fft_h, fft_w))
    full = torch.fft.irfft2(X * Y, s=(fft_h, fft_w))

    # Crop back
    full = full[..., :out_h, :out_w]

    if mode == "full":
        return full
    elif mode == "same":
        sh = (Kh - 1) // 2
        sw = (Kw - 1) // 2
        return full[..., sh:sh + H, sw:sw + W]
    else:  # valid
        vh = abs(H - Kh) + 1
        vw = abs(W - Kw) + 1
        sh = Kh - 1 if H >= Kh else H - 1
        sw = Kw - 1 if W >= Kw else W - 1
        return full[..., sh:sh + vh, sw:sw + vw]
