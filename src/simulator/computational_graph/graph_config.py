"""
Top-level config for a ComputationalGraph instance: an ordered list of node
configs plus a shared device config.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

from simulator.computational_graph.graph_nodes.node_configs import NodeConfigsGroup
from simulator.computational_graph.utils.device_config import DeviceConfig


class GraphConfig(BaseModel):
    """Configuration for a ComputationalGraph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    node_configs: list[NodeConfigsGroup]

    device_config: Optional[DeviceConfig] = None

    num_cpu: int = 1

    @model_validator(mode="after")
    def propagate_device_config_and_check_cpu(self) -> "GraphConfig":
        """Fan the shared device_config into every node that needs one, and
        validate num_cpu is within range."""
        for node_config in self.node_configs:
            node_config.num_cpu = self.num_cpu
            if hasattr(node_config, "device_config"):
                if node_config.device_config is None:  # type: ignore[attr-defined]
                    assert self.device_config is not None, (
                        "must enter device config if any node(s) in graph require it. "
                        f"Required by (at least) node: {node_config.name}."
                    )
                    node_config.device_config = self.device_config  # type: ignore[attr-defined]

        if self.num_cpu != 1:
            if self.num_cpu < 1 or self.num_cpu > (os.cpu_count() or 1):
                raise ValueError(
                    f"num_cpu in GraphConfig must be between 1 and {os.cpu_count()}, "
                    f"not {self.num_cpu}"
                )
        return self
