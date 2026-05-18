"""
Simulator package — extracted from qxcl. Generates (disorder, CSD) training
pairs for the dynamic-disorder-potential project.

Top-level shortcuts; everything else lives in `simulator.pipeline`,
`simulator.injection`, and `simulator.computational_graph`.
"""

from simulator.injection.external_disorder_node import ExternalDisorderNode
from simulator.pipeline import (
    DisorderSource,
    GaussianRandomFieldSource,
    PipelineConfig,
    SimulatorDataset,
    TLFDisorderSource,
    generate_sample,
    load_config,
    run,
)

__all__ = [
    "DisorderSource",
    "ExternalDisorderNode",
    "GaussianRandomFieldSource",
    "PipelineConfig",
    "SimulatorDataset",
    "TLFDisorderSource",
    "generate_sample",
    "load_config",
    "run",
]
