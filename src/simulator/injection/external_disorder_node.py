"""
Injection module: InitialElectroStaticPotential subclass that accepts
an external disorder field instead of running the built-in random donor model.

Usage
-----
from simulator.injection.external_disorder_node import ExternalDisorderNode

node = ExternalDisorderNode(node_config, disorder_field_mV=my_array)
result = node.compute(gate_voltages=voltages)
"""

from typing import List, Optional

import numpy as np

from simulator.computational_graph.graph_nodes.nodes.initial_electrostatic_potential import (
    InitialElectroStaticPotential,
)
from simulator.computational_graph.graph_nodes.node_configs import InitialPotentialNodeConfig
from simulator.computational_graph.graph_nodes.utils.self_consistent_potential_utils import (
    get_scale,
)


class ExternalDisorderNode(InitialElectroStaticPotential):
    """
    Drop-in replacement for InitialElectroStaticPotential that accepts
    an externally supplied disorder field, bypassing the built-in
    random donor model entirely.

    The external field must be:
      - shape : (nx, ny) matching the gate image dimensions
      - units : mV
      - mean-subtracted (SiGe convention)  enforced automatically

    All other behaviour (gate potential computation, surface potential,
    scale factor) is inherited unchanged from the parent class.
    """

    def __init__(
        self,
        node_config: InitialPotentialNodeConfig,
        disorder_field_mV: Optional[np.ndarray] = None,
    ):
        super().__init__(node_config)
        self.set_disorder_field(disorder_field_mV)

    def set_disorder_field(self, field_mV: Optional[np.ndarray]) -> None:
        """
        Set or update the external disorder field.
        Pass None to use zero disorder (matches parent baseline when
        material_config.donors is falsy and mean_disorder == 0).
        """
        if field_mV is not None:
            field_mV = np.array(field_mV, dtype=np.float64)
            field_mV = field_mV - np.mean(field_mV)
        self._external_disorder_field = field_mV

    def compute(self, gate_voltages: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Compute total potential = gate potential + external disorder + surface potential.
        Replaces parent's disorder() call with the externally supplied field.
        """
        if gate_voltages is None:
            raise ValueError("gate_voltages must be provided.")

        pot = self.weighted_potential(gate_voltages, self.potential_split)
        self.scale = get_scale(pot, self.device_config)

        if self._external_disorder_field is not None:
            nx, ny = pot.shape
            fx, fy = self._external_disorder_field.shape
            assert (fx, fy) == (nx, ny), (
                f"Disorder field shape {(fx, fy)} does not match "
                f"gate potential shape {(nx, ny)}. "
                f"Resample your disorder field at the correct resolution."
            )
            dis = self._external_disorder_field
        else:
            dis = 0.0

        pot = pot * self.material_config.scale_factor
        total_potential = pot + dis + self.material_config.surface_potential

        return [total_potential]
