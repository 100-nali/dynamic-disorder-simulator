"""
Graph node for calculating the semiclassical transport

Copyright © 2023 QuantrolOx Ltd
"""

from typing import List, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from simulator.computational_graph.graph_nodes.abstract_graph_node import AbstractGraphNode
from simulator.computational_graph.graph_nodes.node_configs import TransportNodeConfig
from simulator.computational_graph.graph_nodes.utils.self_consistent_potential_utils import get_scale
from simulator.computational_graph.graph_nodes.utils.transport_utils import path_finder


class SemiClassicalTransportNode(AbstractGraphNode):
    """
    Graph node for calculating the semiclassical transport within the device and between dots.
    """

    def __init__(self, node_config: TransportNodeConfig) -> None:
        """
        Initializes node with name and optional connections.

        Args:
            node_config: configuration parameters for node call
        """
        super().__init__(node_config)

        assert (
            node_config.device_config is not None
        ), "Device config must be provided to transport node"

        self.node_config = node_config
        self.device_config = node_config.device_config
        self.scale = np.array(None)
        self.minimax = np.array(None)

    def compute(self, potential: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Estimates the transport path through the device using a minimum spanning tree and path
        finding algorithm.

        Args:
            potential: self-consistent potential
        Returns:
            path_map: binary mask of the path through the device
            current_possible: bool, whether current along path is possible
        """
        if potential is None:
            raise ValueError(
                "Electric potential must be calculated for transport measurement. "
                " Ensure the graph includes a self-consistent potential calculating node."
            )

        self.scale = get_scale(potential, self.device_config)
        reduced_potential = potential[
            :: self.node_config.potential_cut, :: self.node_config.potential_cut
        ]

        # TODO: do a floodfill to test if path is even possible.

        energy_path, path, path_ij, path_map = path_finder(
            reduced_potential,
            self.device_config,
            random_trajectory=self.node_config.random_trajectory,
        )
        path_ij = np.array(path_ij).T

        # evaluate minimax point
        self.minimax = np.max(energy_path)
        loc = np.argmax(energy_path)
        minimax_point = np.array([path[loc, 1], path[loc, 0]])

        if self.node_config.information:
            # Determine length scales
            ny, nx = potential.shape

            y = np.arange(-self.scale[1] * ny / 2, self.scale[1] * ny / 2, self.scale[1])  # nm
            x = np.arange(-self.scale[0] * nx / 2, self.scale[0] * nx / 2, self.scale[0])  # nm

            xl, yl = x[:: self.node_config.potential_cut], y[:: self.node_config.potential_cut]

            plt.figure(figsize=(8, 6))
            plt.title("1D Path: Source -> Drain", fontsize=20)
            plt.plot(range(len(path)), energy_path, "b-")
            plt.plot(np.ones(len(path)) * -1 * self.device_config.fermi_level, "r--")
            plt.xlabel("Step Number", fontsize=16)
            plt.ylabel("Potential Energy [" + r"$meV$" + "]", fontsize=16)
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.title("Self Consistent Potential")
            plt.imshow(
                reduced_potential, extent=[x[0], x[-1], y[0], y[-1]], cmap="hot", origin="lower"
            )
            plt.colorbar()
            plt.scatter(xl[path_ij[1]], yl[path_ij[0]], s=4, c="b", alpha=0.7)
            plt.scatter(xl[minimax_point[0]], yl[minimax_point[1]], s=40, c="c")
            plt.text(xl[path_ij[1]][0], yl[path_ij[0]][0], "S", color="white", fontsize=14)
            plt.text(xl[path_ij[1]][-1], yl[path_ij[0]][-1], "D", color="white", fontsize=14)
            plt.xlabel("x [" + r"$nm$" + "]", fontsize=16)
            plt.ylabel("y [" + r"$nm$" + "]", fontsize=16)
            plt.show()

        return [path_map, self.minimax]

    def current_through_channel_possible(self):
        # Return current based on minimax value
        return 1 * (self.minimax < 0)
