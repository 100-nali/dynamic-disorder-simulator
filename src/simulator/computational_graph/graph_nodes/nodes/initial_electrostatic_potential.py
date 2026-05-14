"""
Graph node to compute the initial electrostatic potential.

Copyright © 2022 QuantrolOx Ltd
"""
from os.path import join
from typing import List, Optional

import numpy as np
import skimage  # type:ignore
from scipy.signal import convolve  # type:ignore

from simulator.computational_graph.graph_nodes.abstract_graph_node import AbstractGraphNode
from simulator.computational_graph.graph_nodes.node_configs import InitialPotentialNodeConfig
from simulator.computational_graph.graph_nodes.utils.self_consistent_potential_utils import (
    calc_pot,
    get_scale,
    kernel_calc,
)


class InitialElectroStaticPotential(AbstractGraphNode):
    """
    Graph node for generating initial electrostatic potential.
    """

    def __init__(self, node_config: InitialPotentialNodeConfig) -> None:
        """
        Initializes node with name and optional connections.

        Args:
            node_config: configuration parameters for node call
        """
        super().__init__(node_config)

        assert (
            node_config.device_config is not None
        ), "Device config must be provided to initial potential node"
        self.device_config = node_config.device_config
        self.gate_config = self.device_config.gate_config
        self.material_config = self.device_config.material_config
        self.gate_split: np.ndarray = np.array(None)

        if self.gate_config.gate_split is not None:
            self.gate_split = self.gate_config.gate_split
        elif self.gate_config.gate_split_path is not None:
            self.gate_split = np.load(self.gate_config.gate_split_path)

        if len(self.gate_split.shape) == 0:  # array is still None
            if self.gate_config.gate_img_array is not None:
                self.gate_img_array = np.array(self.gate_config.gate_img_array)
            elif (
                self.gate_config.gate_design_file is not None
                and self.gate_config.gate_design_dir is not None
            ):
                self.gate_img_array = np.array(
                    skimage.io.imread(
                        join(self.gate_config.gate_design_dir, self.gate_config.gate_design_file)
                    )
                )
            else:
                raise ValueError(
                    "Gates not specified by gate config input to initial potential node."
                )

            self.gate_split = self.get_gate_split()

        self.scale: Optional[np.ndarray] = None
        self.potential_split = self.get_potential_split()

    def compute(self, gate_voltages: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Combines individual electrostatic potential contributions from each voltage gate."""

        if gate_voltages is None:
            raise ValueError("Arguments not provided in initial electrostatic node call method")

        # Calculate potential landscape
        pot = self.weighted_potential(gate_voltages, self.potential_split)
        # Determine length scales
        self.scale = get_scale(pot, self.device_config)

        if bool(self.material_config.donors):
            dis = self.disorder(pot)
        else:
            dis = self.material_config.mean_disorder

        pot = pot * self.material_config.scale_factor
        total_potential = pot + dis + self.material_config.surface_potential

        return [total_potential]

    def weighted_potential(
        self, gate_voltages: np.ndarray, potential_split: np.ndarray
    ) -> np.ndarray:
        """
        Sums the potential from each gate's potential contribution.
        """
        n_gate = potential_split.shape[0]

        if not len(gate_voltages) == n_gate:
            print(
                "warning, number of input gate voltages does not match number of identified gates"
            )

        pot_sum = np.tensordot(potential_split, gate_voltages[:n_gate] / 10000, axes=((0), (0)))

        return pot_sum

    # pylint: disable-next=too-many-locals
    def disorder(self, pot_small_sum, frac_order=0):
        """
        Args:
            pot_small_sum: electrostatic potential generated from gate voltages
            frac_order: fraction of donors that are ordered

        Returns:
            dis: The electrostatic potential contribution from randomly disordered ionized donors
        """
        pot = pot_small_sum
        scale_x, scale_y = self.scale[0], self.scale[1]
        # Calculate Coulomb Kernel
        kernel = kernel_calc(
            pot,
            scale_x,
            scale_y,
            screen=False,
            device_config=self.device_config,
        )

        n_0, m_0 = pot.shape
        n, m = kernel.shape

        # Number of donors
        area = m * scale_x * n * scale_y  # m^2
        # Fraction of donors which are ionised
        frac_ion = 0.202  # 0.39
        # Donor density (experimental)
        density = self.device_config.donor_density  # 6.2e16  # m^-2
        p = int(area * density * frac_ion)

        #####################################
        # Disorder
        #####################################
        p_dis = int((1 - frac_order) * p)

        num_points = n * m
        # Random Donor Locations
        donor_idxs = np.random.randint(num_points, size=p_dis)
        donor_map = np.bincount(donor_idxs, minlength=n * m).reshape(n, m)
        # Multiply by magnitude of electron charge
        charge = donor_map * 1.6e-19
        # Convolution using Coulomb Kernel
        dis = convolve(charge, kernel, mode="same")
        # Return array of correct size
        n_cut, m_cut = int((n - n_0) / 2), int((m - m_0) / 2)
        dis = dis[n_cut : n - n_cut, m_cut : m - m_cut]

        if dis.shape[0] == n_0 + 1:
            dis = dis[1:, :]
        if dis.shape[1] == m_0 + 1:
            dis = dis[:, 1:]

        dis = dis * 1e3  # mV

        if self.device_config.material.lower() == "sige":
            return dis - np.mean(dis)
        return dis

    # pylint: disable-next=too-many-locals
    def get_potential_split(self) -> np.ndarray:
        """
        Returns:
            potential_split: Independently calculated potentials generated by each voltage gate.
        """

        _, nx, ny = self.gate_split.shape
        pad_frac = 4
        gate_split = np.pad(
            self.gate_split,
            (
                (0, 0),
                (int(nx / pad_frac), int(nx / pad_frac)),
                (int(ny / pad_frac), int(ny / pad_frac)),
            ),
            mode="edge",
        )

        n_gates, *_ = gate_split.shape

        length = self.device_config.width
        s_x = length / nx
        s_y = length / ny

        # Average length scales taken from x,y direction measurements
        scale_x = (s_x + s_y) / 2
        scale_y = (s_x + s_y) / 2

        # scale in metres
        scale_x = scale_x * 1e-9
        scale_y = scale_y * 1e-9

        # Potential
        potential = np.zeros(gate_split.shape)

        # Depth from gates
        d = (self.device_config.carrier_depth + 5) * 1e-9

        #################################################################
        # Loop over gates
        #################################################################

        for k in range(n_gates):
            potential[k] = calc_pot(gate_split[k], scale_x, scale_y, d)

        potential_split = potential[
            :, int(nx / pad_frac) : -int(nx / pad_frac), int(ny / pad_frac) : -int(ny / pad_frac)
        ]

        return potential_split

    def get_gate_split(self) -> np.ndarray:
        """
        Returns 3d array in form [2D image of gate 1, 2D image of gate 2, ... ].

        Args:
            recompute: if false, reads gate split from file if it exists, returns it.

        Returns:
            3d array of the gate split
        """
        data = self.gate_img_array

        # convert RGB to black/white image
        if len(data.shape) > 2:
            data = np.mean(data, axis=2)

        # gate_img_array attribute is loaded as np array, error thrown below is unnecessary
        values = np.unique(data)  # type: ignore

        gate_split = []
        for u in values[:-1]:
            gate_split.append(1 * (data == u))

        gate_split_array = np.array(gate_split)

        return gate_split_array
