"""
Graph node to compute the tunnel coupling between two quantum dots, defined as half of the splitting
of the ground and first-excited (bonding and anti-bonding) quantum states, and obtained by solving 
the discrete Schroedinger equation in two dimensions. 

Copyright © 2023 QuantrolOx Ltd
"""
from typing import List, Optional, Union

import kwant  # type:ignore
import numpy as np
from scipy.optimize import minimize  # type:ignore

from simulator.computational_graph.graph_nodes.abstract_graph_node import AbstractGraphNode
from simulator.computational_graph.graph_nodes.node_configs import DoubleDotTunnelCouplingNodeConfig
from simulator.computational_graph.graph_nodes.utils.tunneling_utils import get_eigenstates, make_system
from simulator.computational_graph.utils.device_config import Dot


# pylint: disable=too-many-locals
class DoubleDotTunnelCouplingNode(AbstractGraphNode):
    """
    Graph node to compute the tunnel coupling between two quantum dots, defined as half of the
    splitting of the ground and first-excited (bonding and anti-bonding) quantum states, and
    obtained by solving the discrete Schroedinger equation in two dimensions. Takes as input the
    self-consistant potential data, and assumes that two charge configurations have equal total
    energies corresponding to special values of the applied voltages (boundary of stable charge
    regions). [May lift this requirement in a later stage]
    """

    def __init__(self, node_config: DoubleDotTunnelCouplingNodeConfig) -> None:
        """
        Initializes node with name and optional connections.

        Args:
            node_config: configuration parameters for node call

        """
        super().__init__(node_config)

        assert (
            node_config.device_config is not None
        ), "Device config must be provided to tunneling node"

        self.node_config = node_config
        self.device_config = node_config.device_config

        self.mode = node_config.tunneling_estimator
        self.plot_pdensity = node_config.plot_probability_density

    def compute(
        self, potential: Optional[np.ndarray] = None, dots: Optional[Union[list[Dot], list]] = None
    ) -> List:
        """
        Args:
            potential: potential energy data (numpy array)
            dots: list of dots in the device

        Returns: value of inter-dot tunnel coupling energy in meV.
        """

        if dots is None:
            raise ValueError("Dots argument not provided in tunneling node call method")
        if potential is None:
            raise ValueError(
                "Self-consistent potential argument not provided in \
                tunneling node call method"
            )

        if len(dots) != 2:
            print(
                f"Node {self.name} assumes two quantum dots, tunnel coupling not calculated, "
                "returning None"
            )
            return [None]

        # Reduce potential to left, right, bottom, and top most limits of the two bounding wells
        bounding_wells = np.vstack((dots[0].bounding_well, dots[1].bounding_well))
        mins = np.min(bounding_wells[:, 0::2], axis=0).astype(int)
        maxes = np.max(bounding_wells[:, 1::2], axis=0).astype(int)

        red_potential = potential[mins[1] : maxes[1], mins[0] : maxes[0]]

        syst = make_system(red_potential, self.device_config, self.node_config.lattice_shape)
        syst = syst.finalized()

        ham_mat = syst.hamiltonian_submatrix(sparse=True)
        evals, evs = get_eigenstates(ham_mat, self.node_config.k)

        if self.mode.lower() == "excited_state":
            tunnel_coupling_energy = 0.5 * (evals[1] - evals[0])

        elif self.mode.lower() == "wannier":
            psi0, psi1 = evs[:, 0], evs[:, 1]
            l = len(psi0)

            def pr(x):
                """
                Computes the participation ratio of the wavefunction after unitary transformation
                in the two-dimensional quasi-degenerate subspace.
                """
                alpha, beta, theta = x
                psi0p = (
                    np.exp(complex(0, alpha)) * np.cos(theta) * psi0
                    + np.exp(complex(0, beta)) * np.sin(theta) * psi1
                )
                return 1 / np.sum([abs(psi0p[i]) ** 4 for i in range(l)])

            sol = minimize(
                pr, x0=(1, 1, 1), bounds=[(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)]
            )

            alpha, beta, theta = sol.x

            psi0p = (
                np.exp(complex(0, alpha)) * np.cos(theta) * psi0
                + np.exp(complex(0, beta)) * np.sin(theta) * psi1
            )
            psi1p = (
                -np.exp(-complex(0, beta)) * np.sin(theta) * psi0
                + np.exp(-complex(0, alpha)) * np.cos(theta) * psi1
            )

            tunnel_coupling_energy = abs(np.vdot(psi1p, (ham_mat * psi0p)))

        if self.plot_pdensity:
            kwant.plotter.map(syst, np.abs(psi0) ** 2, colorbar=True, oversampling=1, cmap="plasma")
            kwant.plotter.map(syst, np.abs(psi1) ** 2, colorbar=True, oversampling=1, cmap="plasma")

        return [tunnel_coupling_energy]
