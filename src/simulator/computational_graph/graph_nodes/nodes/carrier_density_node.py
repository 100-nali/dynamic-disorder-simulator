"""
Graph node for computing the electron density in a device.
Copyright © 2022 QuantrolOx Ltd
"""

from typing import List, Optional

import numpy as np

from simulator.computational_graph.graph_nodes.abstract_graph_node import AbstractGraphNode
from simulator.computational_graph.graph_nodes.node_configs import CarrierDensityNodeConfig
from simulator.computational_graph.graph_nodes.utils.self_consistent_potential_utils import (
    get_carrier_density,
)


class CarrierDensityNode(AbstractGraphNode):
    """
    Graph node for estimating quantum dot occupation using the Thomas-Fermi model.
    """

    def __init__(self, node_config: CarrierDensityNodeConfig) -> None:
        """
        Initializes node with name and optional connections.

        Args:
            node_config: configuration parameters for node call
        """
        super().__init__(node_config)

        assert (
            node_config.device_config is not None
        ), "Device config must be provided to carrier density node"

        self.device_config = node_config.device_config
        self.material_config = node_config.device_config.material_config

    def compute(
        self,
        potential: Optional[np.ndarray] = None,
    ) -> List:
        """
        Method to return carrier density.

        Args:
            potential: Total potential of the device.
        Returns:
            the carrier density in the device
        """
        if potential is None:
            raise ValueError(
                "Self-consistent potential argument not provided in \
                capacitance node call method"
            )
        carrier_density = get_carrier_density(
            pot=potential, material_config=self.material_config, mu=0.0
        )
        return [carrier_density]
