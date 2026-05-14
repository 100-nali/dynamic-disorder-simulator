"""
Util functions for quantum dot detection and occupation graph nodes

Copyright © 2022 QuantrolOx Ltd
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from scipy.ndimage import center_of_mass  # type: ignore
from scipy.signal import find_peaks  # type: ignore
from skimage.segmentation import flood_fill  # type: ignore

from simulator.computational_graph.graph_nodes.utils.self_consistent_potential_utils import (
    coulomb,
    get_carrier_density,
    get_scale,
)
from simulator.computational_graph.utils.device_config import DeviceConfig, Dot  # type: ignore


def get_arbitrary_kernel(
    array: np.ndarray,
    device_config: DeviceConfig,
) -> np.ndarray:
    """

    Args:
        array: 2D array defining domain
        total_config: class of TotalConfig containing parameters

    Returns:
        kernel: potential from charge in carrier plane

    """
    # Return coulomb kernel for capacitances
    dx, dy = get_scale(array, device_config)
    y_n, x_n = array.shape

    xr, yr = x_n * dx / 2, y_n * dy / 2

    x = np.arange(dx, xr, dx)
    y = np.arange(dy, yr, dy)

    x = np.concatenate([x[::-1], [dx / 2.0], x])
    y = np.concatenate([y[::-1], [dy / 2.0], y])

    x_grid, y_grid = np.meshgrid(x, y)

    kernel = coulomb(
        x_grid,
        y_grid,
        0,
        0,
        device_config=device_config,
    )

    return kernel


def integrate_2d(array: np.ndarray, scale: np.ndarray) -> float:
    """

    Args:
        array: 2D array
        scale: length scales [dx,dy]

    Returns:
        result of integral over 2D space of values in array
    """
    nx, ny = array.shape
    x = np.linspace(0, scale[0] * nx, nx)
    y = np.linspace(0, scale[1] * ny, ny)

    return np.trapz(np.trapz(array, x, axis=0), y)


def get_dot_mask(energy: np.ndarray, location: np.ndarray) -> np.ndarray:
    """
    Args:
        energy: 2D energy landscape
        location: point to start the mask

    Returns:
        mask: 2D binary mask covering dot region

    """
    # Binary mask for electron density
    image = (energy < 0) * 1
    # Mask for single dot
    new_image = flood_fill(image, (location[0], location[1]), 2)
    mask = (new_image == 2) * 1
    return mask


def valid_carrier_path(mask: np.ndarray, test_locale: np.ndarray) -> bool:
    """
    Check if there is a valid carrier along the floodfilled mask to the test_locale also plots the
    mask.
    Args:
        mask: 2D binary mask of sub-Fermi level region connected to start location used in floodfill
    """

    # test if the end point has been reached via the floodfill mask
    return bool(mask[test_locale[0], test_locale[1]] == 1)


def dot_com(dots: list[Dot]) -> np.ndarray:
    """
    Method to find and assign center of mass for each dot in dots

    Args:
        dots: list of dots with locations and masks for each dot area

    Returns:
        coms: List of dot center of mass(es).
    """
    coms = []
    for dot in dots:
        # Generate dot mask
        dot.center_of_mass = np.array(center_of_mass(dot.mask))
        # com = com / np.array([energy.shape[0], energy.shape[1]])
        coms.append(dot.center_of_mass)  # [::-1]
    return np.array(coms)


# pylint: disable=too-many-locals
def dot_bounding_box(energy: np.ndarray, dots: list[Dot]) -> np.ndarray:
    """
    Function to generate bounding box for each dot in dots

    Args:
        energy: 2D energy landscape
        dots: list of dots with locations and masks for each dot area
    Returns:
        box: nparray defining bounding box [center(x), center(y), height, width] for each dot.
    """

    box = np.zeros((len(dots), 4))
    for k, dot in enumerate(dots):
        # Get bounds of  dot in i,j coordinates
        i_extent = np.atleast_1d(np.squeeze(np.argwhere(np.sum(dot.mask, axis=1))))
        j_extent = np.atleast_1d(np.squeeze(np.argwhere(np.sum(dot.mask, axis=0))))
        lower_i, upper_i = i_extent[0].astype(float), i_extent[-1].astype(float) + 1.0
        lower_j, upper_j = j_extent[0].astype(float), j_extent[-1].astype(float) + 1.0
        # Define centre of dot based on bounds
        centre = [
            0.5 * (lower_j + upper_j) / energy.shape[1],
            0.5 * (lower_i + upper_i) / energy.shape[0],
        ]
        # Increase bounds to allow for buffer around edge
        lower_i, lower_j = (lower_i - 1) / energy.shape[0], (lower_j - 1) / energy.shape[1]
        upper_i, upper_j = (upper_i + 1) / energy.shape[0], (upper_j + 1) / energy.shape[1]
        # Define height/width of dot based on bounds
        height = upper_j - lower_j
        width = upper_i - lower_i
        # Data to define a box
        dot_box = [centre[0], centre[1], height, width]
        box[k] = np.array(dot_box)

    return box


def get_capacitive_disc_energy(
    potential: np.ndarray,
    mask: np.ndarray,
    device_config: DeviceConfig,
    use_fudge_factor: bool = True,
) -> np.ndarray:
    """
    Function to model the energy on a given dot mask as the energy of a capacitive disc.
    Args:
        potential: 2D potential landscape
        mask: 2D binary mask of dot region
        device_config: class of DeviceConfig containing parameters
        use_fudge_factor: whether to use the fudge factor for the induced charge
    Returns:
        energy: energy of capacitive disc
    """
    epsilon_r = device_config.material_config.relative_permittivity
    epsilon = 8.85e-12 * epsilon_r  # F/m
    disc_prefactor = 8 * epsilon

    scale = device_config.pixel_lengths
    q = 1.6e-19 * device_config.material_config.carrier_charge

    radius = np.sqrt(np.sum(mask) / np.pi) * scale.mean()

    # TODO: could make more efficient by having option between potential or dot occ/charge number
    density = get_carrier_density(potential * mask, device_config.material_config)
    induced_charge = integrate_2d(density, scale)

    if use_fudge_factor:
        induced_charge *= device_config.material_config.exp_charge_fudge_factor

    return (q * induced_charge) ** 2 / (2 * disc_prefactor * radius)


# pylint: disable=too-many-branches
def dot_bounding_well(energy: np.ndarray, dots: list[Dot]) -> np.ndarray:
    """
    Function to return the bounding energy well for each dot in dots. Identifies the maximum
        energies to the North, South, East and West of the dot center of mass.
    Args:
        energy: 2D energy landscape
        dots: list of dots with locations and masks for each dot area
    Returns:
        bounding_well: nparray indices defining the rescaled bounding energy well for each dot.
            [left, right, bottom, top]
    """

    pos_energy = (energy > 0) * energy
    bounding_well = np.zeros((len(dots), 4), dtype=int)
    shape = pos_energy.shape
    for i, dot in enumerate(dots):
        int_com = np.round(dot.center_of_mass).astype(int)

        # Get right bounds, where energy is above 0 and first starts to dip.
        peak = find_peaks(pos_energy[int_com[0], int_com[1] :])[0]
        if len(peak) > 0:
            peak_loc = peak[0]
        else:
            peak_loc = np.argmax(pos_energy[int_com[0], int_com[1] :])
        r_bound = int_com[1] + peak_loc

        if r_bound < shape[1] - 1:
            assert (
                pos_energy[int_com[0], r_bound - 1]
                <= pos_energy[int_com[0], r_bound]
                >= pos_energy[int_com[0], r_bound + 1]
            ), "Right bound not found correctly"

        # Get left bounds, where energy first starts to increase (as array & gradient reversed).
        peak = find_peaks(pos_energy[int_com[0], : int_com[1]])[0]
        if len(peak) > 0:
            peak_loc = peak[-1]
        else:
            peak_loc = np.argmax(pos_energy[int_com[0], : int_com[1]])
        l_bound = peak_loc

        if l_bound > 0:
            assert (
                pos_energy[int_com[0], l_bound - 1]
                <= pos_energy[int_com[0], l_bound]
                >= pos_energy[int_com[0], l_bound + 1]
            ), "Left bound not found correctly"

        # Get upper bounds, where energy is above 0 and first starts to dip.
        peak = find_peaks(pos_energy[int_com[0] :, int_com[1]])[0]
        if len(peak) > 0:
            peak_loc = peak[0]
        else:
            peak_loc = np.argmax(pos_energy[int_com[0] :, int_com[1]])
        u_bound = int_com[0] + peak_loc

        if u_bound < shape[0] - 1:
            assert (
                pos_energy[u_bound - 1, int_com[1]]
                <= pos_energy[u_bound, int_com[1]]
                >= pos_energy[u_bound + 1, int_com[1]]
            ), "Upper bound not found correctly"

        # Get lower bounds, where energy first starts to increase (as array & gradient reversed).
        peak = find_peaks(pos_energy[: int_com[0], int_com[1]])[0]
        if len(peak) > 0:
            peak_loc = peak[-1]
        else:
            peak_loc = np.argmax(pos_energy[: int_com[0], int_com[1]])
        b_bound = peak_loc

        if b_bound > 0:
            assert (
                pos_energy[b_bound - 1, int_com[1]]
                <= pos_energy[b_bound, int_com[1]]
                >= pos_energy[b_bound + 1, int_com[1]]
            ), "Lower bound not found correctly"

        # Define bounding well
        bounding_well[i, :] = np.array([l_bound, r_bound, b_bound, u_bound], dtype=int)

        dot.bounding_well = bounding_well[i, :]

    return bounding_well


def plot_dots(energy: np.ndarray, dots: list[Dot], origin: str = "upper") -> None:
    """
    Method to plot all dots on energy landscape

    Args:
        energy: 2D energy landscape
        dots: list of dots with locations and masks for each dot area
        origin: origin of plot
    """
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(1.0 * (energy < 0), extent=(0, 1, 0, 1), origin="lower", cmap="gnuplot")
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(1.0 * (energy < 0), extent=(0, 1, 0, 1), origin=origin, cmap="gnuplot")

    for dot in dots:
        # Get bounds of  dot in i,j coordinates
        i_extent = np.atleast_1d(np.squeeze(np.argwhere(np.sum(dot.mask, axis=1))))
        j_extent = np.atleast_1d(np.squeeze(np.argwhere(np.sum(dot.mask, axis=0))))
        lower_i, upper_i = i_extent[0].astype(float), i_extent[-1].astype(float) + 1.0
        lower_j, upper_j = j_extent[0].astype(float), j_extent[-1].astype(float) + 1.0

        # Increase bounds to allow for buffer around edge
        lower_i, lower_j = (lower_i - 1) / energy.shape[0], (lower_j - 1) / energy.shape[1]
        upper_i, upper_j = (upper_i + 1) / energy.shape[0], (upper_j + 1) / energy.shape[1]
        # Define height/width of dot based on bounds
        height = upper_j - lower_j
        width = upper_i - lower_i

        # For origin in upper left corner
        if origin == "upper":
            lower_i = 1 - lower_i - width

        rect = plt.Rectangle(
            (lower_j, lower_i), height, width, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    plt.show(block=False)
