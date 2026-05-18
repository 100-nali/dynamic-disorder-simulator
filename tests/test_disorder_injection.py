"""
Regression test for the disorder-injection bug.

Pre-fix history: `ComputationalGraph._process_components` bound component
references in partials at __init__ time, so later substitutions
(`graph.components[0] = patched_node`) had no effect.
`build_graph_with_disorder` did exactly that, which silently dropped the
disorder field. All "disorder-injected" CSDs prior to this fix were
identical regardless of the input disorder.

This test asserts that two graphs built with two different disorder fields
produce different SC potentials at the same gate voltages.
"""

from pathlib import Path

import numpy as np
import pytest

from simulator.pipeline.sweep import build_graph_with_disorder

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_YAML = REPO_ROOT / "configs" / "graph.yaml"

# Skip if heavy native deps (kwant, etc.) aren't installed — keeps the test
# runnable from any dev env without forcing a full simulator install.
try:
    import kwant  # noqa: F401
except ImportError:
    pytest.skip("kwant not installed; skipping disorder-injection regression test",
                allow_module_level=True)


def _voltages() -> np.ndarray:
    """Dot-forming voltages on IST_10721_S14 (accelerated-graph style)."""
    v = np.array([1300.0, 2700.0, 1300.0, 1800.0, 0.0, 0.0,
                  1100.0, 1300.0, 1200.0, 1300.0, 1000.0, 1800.0])
    return v


def test_disorder_actually_propagates() -> None:
    """Two different disorder fields must produce different SC potentials."""
    nx, ny = 346, 346

    rng = np.random.default_rng(0)
    disorder_a = rng.standard_normal((nx, ny)).astype(np.float64) * 5.0
    rng = np.random.default_rng(1)
    disorder_b = rng.standard_normal((nx, ny)).astype(np.float64) * 5.0

    g_a = build_graph_with_disorder(disorder_field_mV=disorder_a, config_path=GRAPH_YAML)
    g_b = build_graph_with_disorder(disorder_field_mV=disorder_b, config_path=GRAPH_YAML)

    v = _voltages().astype(np.float64)
    sc_a = g_a.run(
        target=["self_consistent_potential_iterative"],
        set_value_dict={"gate_voltages": v},
    )["self_consistent_potential_iterative"]
    sc_b = g_b.run(
        target=["self_consistent_potential_iterative"],
        set_value_dict={"gate_voltages": v},
    )["self_consistent_potential_iterative"]

    assert not np.array_equal(sc_a, sc_b), (
        "SC potentials are bit-identical for two different disorder fields — "
        "the disorder field is not propagating through the graph."
    )

    rel_change = np.linalg.norm(sc_a - sc_b) / np.linalg.norm(sc_a)
    # 5 mV disorder std on a ~hundreds-of-mV SC potential -> ~1-5% RMS response
    assert rel_change > 1e-3, (
        f"SC potential responded too weakly to disorder: rel change {rel_change:.4g}; "
        "expected at least 0.001 for 5 mV-std injected disorder."
    )
