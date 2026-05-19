"""Tests for calibrated dot-forming voltage helpers."""

import numpy as np

from simulator.utils.operating_points import (
    get_dot_forming_voltages_for_ge_holes_double_dot,
    get_dot_forming_voltages_for_oxford_double_dot,
)


def test_ge_holes_double_dot_shape_and_dtype() -> None:
    v = get_dot_forming_voltages_for_ge_holes_double_dot()
    assert v.shape == (12,), f"expected 12 gates, got {v.shape}"
    assert v.dtype == np.float64
    # Sanity: every entry in a reasonable range for this device
    assert (v >= 0).all() and (v <= 3000).all(), "voltage out of expected range [0, 3000] mV"


def test_ge_holes_double_dot_specific_calibration() -> None:
    """The two calibration-defining knobs should match what we found in the scan."""
    v = get_dot_forming_voltages_for_ge_holes_double_dot()
    # Bottom qubit plunger suppressed
    assert v[4] == 0.0
    assert v[5] == 0.0
    # Top qubit plunger raised to deplete the over-populated top dot
    assert v[9] == 2300.0
    # Sensor plunger strongly biased
    assert v[1] == 2700.0


def test_oxford_double_dot_shape_and_dtype() -> None:
    v = get_dot_forming_voltages_for_oxford_double_dot()
    assert v.shape == (11,), f"expected 11 gates, got {v.shape}"
    assert v.dtype == np.float64
    assert (v >= 0).all() and (v <= 3000).all()


def test_oxford_double_dot_specific_calibration() -> None:
    """
    The voltage values that define the (1, 1) sensor-blockaded operating point
    at the calibrated carrier_depth=47 nm (Jirovec-consistent z-stack).
    """
    v = get_dot_forming_voltages_for_oxford_double_dot()
    # Baseline 1100 mV (the d=47 baseline that allows qubit dots to form).
    assert v[0] == 1100.0   # separator
    assert v[1] == 1100.0   # outer barrier
    assert v[3] == 1100.0   # outer barrier
    # Qubit-array barriers raised to confine two separate (1, 1) wells.
    assert v[4] == 1300.0
    assert v[8] == 1300.0
    # Bottom plunger raised (becomes active at the lower baseline; would
    # otherwise host a spurious 3 e dot at y=191).
    assert v[5] == 1300.0
    # Two active qubit plungers — host the two (1, 1) dots.
    assert v[6] == 900.0
    assert v[7] == 700.0
    # Sensor barriers + plunger: 2 e in deep blockade.
    assert v[2] == 1350.0
    assert v[9] == 1350.0
    assert v[10] == 3100.0
