"""
Graph node for estimating dot occupation using the energy given by the Thomas-Fermi approximation.

Copyright © 2022 QuantrolOx Ltd
"""

import itertools
from typing import List, Optional, Union

import numpy as np
from scipy.signal import convolve  # type:ignore

from simulator.computational_graph.graph_nodes.abstract_graph_node import AbstractGraphNode
from simulator.computational_graph.graph_nodes.node_configs import CapacitanceNodeConfig
from simulator.computational_graph.graph_nodes.utils.dot_physics_utils import (
    get_arbitrary_kernel,
    integrate_2d,
)
from simulator.computational_graph.graph_nodes.utils.self_consistent_potential_utils import (
    get_carrier_density,
)
from simulator.computational_graph.utils.device_config import Dot


class DotOccupationCapacitanceNode(AbstractGraphNode):
    """
    Graph node for estimating quantum dot occupation using the Thomas-Fermi model.
    """

    def __init__(self, node_config: CapacitanceNodeConfig) -> None:
        """
        Initializes node with name and optional connections.

        Args:
            node_config: configuration parameters for node call
        """
        super().__init__(node_config)

        assert (
            node_config.device_config is not None
        ), "Device config must be provided to capacitance node"
        self.device_config = node_config.device_config
        self.material_config = self.device_config.material_config
        self.scale = self.device_config.pixel_lengths
        self.induced_charge = np.array(None)

    def compute(
        self,
        potential: Optional[np.ndarray] = None,
        dots: Optional[Union[list[Dot], list]] = None,
    ) -> List:
        """
        Method to return number of quantum dots.

        Args:
            potential: Total potential of the device.
        Returns: Number of 2D potential wells that may be viable quantum dots.
        """
        if dots is None:
            raise ValueError("Dots argument not provided in capacitance node call method")
        if potential is None:
            raise ValueError(
                "Self-consistent potential argument not provided in \
                capacitance node call method"
            )

        n_dot = len(dots)

        if n_dot == 0:
            return [np.array([]), np.array([[], []])]

        e_matrix = self.capacitance_matrix(potential, dots)
        occupation = self.equilibrium_charge(e_matrix / np.max(e_matrix), dots)

        for ii, dot in enumerate(dots):
            dot.charge_occupation = occupation[ii]

        return [dots, e_matrix]

    def capacitance_matrix(self, potential, dots) -> np.ndarray:  # pylint: disable=too-many-locals
        """
        Args:
            potential: Total potential of the device.
        Returns:
            e_matrix: Normalized energy matrix of the quantum dots.
        """
        n_dot = len(dots)
        # Define electrostatic energy and capacitance matrices
        e_matrix = np.zeros((n_dot, n_dot))

        if n_dot == 0:
            self.induced_charge = np.array([-1])
            return e_matrix

        # Find electrostatic energy and capacitance matrices between dots
        energy = self.material_config.carrier_charge * potential
        if self.material_config.atomic_units:
            q = float(self.material_config.carrier_charge)
        else:
            q = (1.6e-19) * self.material_config.carrier_charge

        # Induced charge from integrating charge density over dot area
        induced_charge = []
        for dot in dots:
            # Charge density in dot
            density = self.material_config.exp_charge_fudge_factor * get_carrier_density(
                dot.mask * potential, self.material_config
            )
            induced_charge.append(integrate_2d(density, self.scale))
        self.induced_charge = np.array(induced_charge)

        coulomb = get_arbitrary_kernel(energy, self.device_config)
        # Using a and b for easier reading than i and j
        for a in range(n_dot):
            # Charge density on a th dot
            mask_a = dots[a].mask
            density_a = get_carrier_density(mask_a * potential, self.material_config)

            potential_a = convolve(density_a, coulomb, mode="same")
            for b in range(n_dot):
                # Energy matrix element
                # Charge density on j th dot
                mask_b = dots[b].mask
                density_b = get_carrier_density(mask_b * potential, self.material_config)
                # Kinetic energy of dot (Thomas Fermi)
                # if a == b: e_ab += integrate_2d(density_a * density_a, self.scale) * 0.5*np.pi
                # Energy on b th dot from charge on a th dot
                potential_a_at_b = potential_a * mask_b
                inter_dot = 0.5 * q**2 * integrate_2d(potential_a_at_b * density_b, self.scale)
                # Final matrix element normalised by induced charges
                e_matrix[a, b] = inter_dot

        return e_matrix

        # self.c_matrix = 2 * np.linalg.inv(self.e_matrix)

    def equilibrium_charge(self, e_matrix: np.ndarray, dots) -> np.ndarray:
        """
        Args:
            potential: Total potential of the device.
        Returns: The occupation of each dot.
        """
        n_dot = len(dots)
        if n_dot == 0:
            return np.array([-1])

        # Initial guess for integer charge occupation
        n_charge_0 = np.floor(self.induced_charge).astype(int)
        charge_energy_0 = np.dot(n_charge_0.T, np.dot(e_matrix, n_charge_0))
        # Possible values for integer charge occupation
        search_range = [-1, 1]

        # Loop over every combination of charges around induced charge (floor,ceil)
        for perm in itertools.product(search_range, repeat=n_dot):
            perm_array = np.array(list(perm))
            n_charge = np.round(self.induced_charge + 0.5 * perm_array)
            q_vector = n_charge - self.induced_charge

            charge_energy = np.dot(q_vector.T, np.dot(e_matrix, q_vector))
            # Store charge occupation which gives the lowest energy
            if charge_energy <= charge_energy_0:
                charge_energy_0 = charge_energy
                n_charge_0 = n_charge
                # print(n_charge_0, charge_energy_0)

        # Experimental fudge factor to match experimental charge occupation data
        return n_charge_0.astype(int)
