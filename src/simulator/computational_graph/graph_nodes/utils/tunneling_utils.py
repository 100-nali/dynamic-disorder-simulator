"""
Util functions for tunneling graph node

Copyright © 2023 QuantrolOx Ltd
"""

from typing import Optional

import kwant  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy as np
import scipy.sparse.linalg as sla  # type:ignore
from scipy.constants import physical_constants  # type:ignore
from scipy.interpolate import RegularGridInterpolator  # type:ignore

from simulator.computational_graph.utils.device_config import DeviceConfig  # type: ignore

# Bohr radius in nm
a0 = 1e9 * physical_constants["Bohr radius"][0]

# Hartree energy in meV
EH = 1e3 * physical_constants["Hartree energy in eV"][0]


def lorentz(x, x0, w, h):
    """
    Lorentzian function
    Args:
        x: independent variable
        x0: center of Lorentzian
        w: standard dev or width of Lorentzian
        h: height of Lorentzian
    """
    return h / ((x - x0) ** 2 / w**2 + 1)


# pylint: disable=too-many-locals
# pylint: disable=no-member
def make_system(
    potential: np.ndarray,
    device_config: DeviceConfig,
    lattice_shape: tuple,
    add_leads: bool = False,
    scale: Optional[tuple] = None,
):
    """
    Kwant tight-binding system builder.

    Args:
        potential: potential energy data (numpy array)


    """

    if scale is None:
        # Computation done in nm
        scale = device_config.pixel_lengths * 1e9

    # Get input data scales
    axd, ayd = scale
    nxd, nyd = potential.shape  # takes entire region over which potential is defined

    # Number of lattice sites in x and y directions
    nx, ny = lattice_shape

    # Lattice spacing in x and y directions
    ax, ay = axd * nxd / nx, ayd * nyd / ny

    # Kwant tight-binding energies for motion along x and y (in meV)
    meff = device_config.material_config.effective_mass
    tx, ty = (0.5 / meff) * (a0 / ax) ** 2 * EH, (0.5 / meff) * (a0 / ay) ** 2 * EH

    lat = kwant.lattice.general([(ax, 0), (0, ay)], norbs=1)

    # pylint: disable-next=no-member
    syst = kwant.Builder()

    def onsite(site):  # , xd, yd, tx, ty):
        """
        Converts potential energy data to python function using a scipy interpolator (possibly
        extrapolate)
        """
        xd, yd = axd * np.arange(nxd), ayd * np.arange(nyd)

        double_dot_potential = RegularGridInterpolator(
            (xd, yd), potential, bounds_error=False, fill_value=None
        )

        return double_dot_potential(site.pos) + 2 * tx + 2 * ty

    for i in range(nx):
        for j in range(ny):
            syst[lat(i, j)] = onsite
            if j > 0:
                syst[lat(i, j), lat(i, j - 1)] = -ty

            if i > 0:
                syst[lat(i, j), lat(i - 1, j)] = -tx

    # Add leads
    if add_leads:
        syst_attach_leads(syst, ax, ny, lat, onsite, ty, tx)

    return syst


# pylint: disable-next=too-many-arguments
def syst_attach_leads(syst, ax, ny, lat, onsite, ty, tx):
    """Adds leads to the system in place"""
    sym_left_lead = kwant.TranslationalSymmetry((-ax, 0))
    left_lead = kwant.Builder(sym_left_lead)

    for j in range(ny):
        left_lead[lat(0, j)] = onsite
        if j > 0:
            left_lead[lat(0, j), lat(0, j - 1)] = -ty
            left_lead[lat(1, j), lat(0, j)] = -tx

    syst.attach_lead(left_lead)

    sym_right_lead = kwant.TranslationalSymmetry((ax, 0))
    right_lead = kwant.Builder(sym_right_lead)

    for j in range(ny):
        right_lead[lat(0, j)] = onsite
        if j > 0:
            right_lead[lat(0, j), lat(0, j - 1)] = -ty
            right_lead[lat(1, j), lat(0, j)] = -tx

    syst.attach_lead(right_lead)


def get_eigenstates(ham_mat, k):
    """
    Computes Hamiltonian eigenstates corresponding to the kth lowest (quasi-degenerate) energies.

    Args:
        ham_mat: Hamiltonian matrix
        k: number of eigenstates and eigenvalues to compute
    Returns:
        evals_sorted: sorted eigenvalues
        evecs_sorted: sorted eigenvectors
    """
    evals, evecs = sla.eigsh(ham_mat.tocsc(), k, which="SA")
    idx = np.argsort(evals)
    evecs_sorted = evecs[:, idx]
    evals_sorted = evals[idx]

    return evals_sorted, evecs_sorted[:, 0:k]


# TODO: generalize to 1 to n dots
# def get_reduced_potential(potential: np.ndarray, dots: list[Dot]):
#         """
#         Get reduced potential energy data for the two dots in the device.
#         """
#         bounding_wells = np.vstack((dots[0].bounding_well, dots[1].bounding_well))
#         mins = np.min(bounding_wells[:, 0::2], axis=0).astype(int)
#         maxes = np.max(bounding_wells[:, 1::2], axis=0).astype(int)

#         return potential[mins[1] : maxes[1], mins[0] : maxes[0]]


def interplotate_potential(potential: np.ndarray):
    nyd, nxd = potential.shape
    xd, yd = np.arange(nxd), np.arange(nyd)

    return RegularGridInterpolator(
        (xd, yd), potential.transpose(), bounds_error=False, fill_value=None
    )


def plot_interpolated_potential(potential: np.ndarray, lattice_shape: tuple):
    """
    Plot interpolated potential
    args:
        potential: potential energy data (numpy array)
        N: number of points to plot
    """
    nyd, nxd = potential.shape
    double_dot_potential = interplotate_potential(potential)

    nx, ny = lattice_shape
    x = np.linspace(0, nxd, nx)
    y = np.linspace(0, nyd, ny)

    xm, ym = np.meshgrid(x, y, indexing="ij")
    zm = double_dot_potential((xm, ym))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xm, ym, zm)
    plt.show()
