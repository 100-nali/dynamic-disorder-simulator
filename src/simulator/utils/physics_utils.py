"""
Util functions for physical distributions and quantities

Copyright © 2023 QuantrolOx Ltd
"""

import numpy as np


def fermi_dirac_dist(energy: np.ndarray, temperature: float, mu: float = 0) -> np.ndarray:
    """
    Fermi-Dirac distribution
    Args:
        energy: energy values
        temperature: temperature in K
        mu: Fermi level value (in eV)
    """

    # Boltzmann constant
    # k_b = 8.617333262145e-5  # eV/K
    k_b = 1.38064852e-23  # J/K

    # Fermi-Dirac distribution
    # dist = 1 / (1 + np.exp(((energy - mu)) / (k_b * temperature)))
    denom = np.logaddexp(0, (energy - mu) / (k_b * temperature))
    dist = np.exp(-denom)

    return dist
