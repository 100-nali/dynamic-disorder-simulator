"""
Graph node to iteratively solve for the self-consistent electrostatic potential
and charge carrier density due to the presence of charge carriers.

<TODO>: Could modify to only require material config and not full device config.

Copyright © 2022 QuantrolOx Ltd
"""

from typing import List, Optional

import numpy as np
from skimage.feature import blob_log  # type: ignore

# pylint: disable-next=no-name-in-module
from skimage.filters import gaussian  # type: ignore

from simulator.computational_graph.graph_nodes.abstract_graph_node import AbstractGraphNode
from simulator.computational_graph.graph_nodes.node_configs import LoGDotDetectionNodeConfig
from simulator.computational_graph.graph_nodes.utils.dot_physics_utils import (
    dot_bounding_well,
    dot_com,
    get_dot_mask,
)
from simulator.computational_graph.utils.device_config import Dot


class LoGQuantumDotDetectionNode(AbstractGraphNode):
    """
    Graph node for detecting the presence of quantum dots (or 2D potential wells) in the device.
    The total potential is input to a Laplacian of Gaussian (LoG) blob detection method which
    identifies coordinates of what could be quantum dots. The blob coordinates are then tested
    and if the blobs are completely surrounded by a high potential, then charge carriers may exhibit
    quantum confinement within these low potential regions and are identified as quantum dots.
    """

    def __init__(self, node_config: LoGDotDetectionNodeConfig) -> None:
        """
        Initializes node with name and optional connections.

        Args:
            node_config: configuration parameters for node call
        """
        super().__init__(node_config)

        assert (
            node_config.device_config is not None
        ), "Device and material config must be provided to LoG dot detection node"
        self.material_config = node_config.device_config.material_config
        self.image_slice = node_config.image_slice

    def compute(self, potential: Optional[np.ndarray] = None) -> List[List[Dot]]:
        """
        Method to return number of quantum dots and each dot's center of mass coordinates.

        Args:
            potential: Total potential of the device.
        Returns:
            Number of 2D potential wells that may be viable quantum dots and their centers of mass.
        """
        n_dots = 0

        if potential is None:
            raise ValueError(
                "Self-consistent potential argument not provided in LoG dot \
                    detection node"
            )

        energy = self.material_config.carrier_charge * potential

        # Method detects energy<0 blobs using the Laplacian of Gaussian blob detection method.
        dot_candidates = (
            self._detect_dots(energy[:: self.image_slice, :: self.image_slice]) * self.image_slice
        )

        # For each blob detected above, the below method cycles through and tests if the energy<0
        # blobs are completely surrounded by energy>0 regions which confines charges and forms dots.
        dots = self._check_dot_candidates(energy, dot_candidates)

        n_dots = len(dots)

        if n_dots > 0:
            # Assign locations and mask dependent centres of mass and bounding wells for each dot
            _ = dot_com(dots)
            _ = dot_bounding_well(energy, dots)

        results = sorted(
            dots, key=lambda x: np.linalg.norm(0 - getattr(x, "center_of_mass")), reverse=False
        )

        return [results]

    def _detect_dots(self, energy: np.ndarray) -> np.ndarray:
        """
        Method to perform blob detection.

        Args:
            energy (np.ndarray): The energy landscape on the chip.

        Returns:
            a list of dot candidates that will be further checked.
        """
        mask = gaussian(1.0 * (energy < 0), sigma=1)
        # Blob detection
        return blob_log(mask, min_sigma=1, max_sigma=50, num_sigma=40, threshold=0.4)

    def _check_dot_candidates(self, energy: np.ndarray, dot_candidates: np.ndarray) -> List[Dot]:
        """
        Method to check whether identified blobs are dots.

        Args:
            energy (np.ndarray): The energy landscape on the chip.
            dot_candidates (np.ndarray): The candidates for potential dots.

        Returns:
            a list of confirmed dots
        """
        assert dot_candidates is not None, "No dot candidates provided"
        source_list = []
        check_mask = np.zeros_like(energy)
        # Check dot candidates against carrier density

        dot_list: List[Dot] = []

        for dot in dot_candidates:
            y, x, _ = dot
            loc = np.array(
                [np.round(y, decimals=0).astype(int), np.round(x, decimals=0).astype(int)]
            )
            # Only consider location if area has not already been checked
            if check_mask[loc[0], loc[1]] < 1e-6 and energy[loc[0], loc[1]] < 0.0:
                # Add to mask of checked areas
                new_mask = get_dot_mask(energy, loc)
                check_mask = check_mask + new_mask
                if self.is_dot(energy, loc):
                    dot_list.append(Dot(mask=new_mask.astype(bool)))
                    source_list.append(loc)

        source_list_array = np.array(source_list)
        # Sort source list by location
        if len(source_list) > 1:
            source_list_array = source_list_array[source_list_array[:, 0].argsort()]

        return dot_list

    def is_dot(self, energy: np.ndarray, location: np.ndarray) -> bool:
        """

        Args:
            energy: the energy landscape of the chip
            location: location to test for dot

        Returns:
            boolean. True for location being inside a dot, False otherwise.

        """
        # Return boolean as to where a dot exists at location
        mask = get_dot_mask(energy, location)
        # Outline of grid
        outline = np.pad(np.zeros_like(energy[2:, 2:]), ((1, 1), (1, 1)), constant_values=1)
        # If the dot mask doesn't reach the boundary we say it is a dot
        return np.sum(outline * mask) == 0
