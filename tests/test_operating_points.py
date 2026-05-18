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
    """The plunger bias values are what define the (1,1) operating point."""
    v = get_dot_forming_voltages_for_oxford_double_dot()
    # Baseline 1250 mV (chosen because uniform 1250 first allows qubit dots to form)
    assert v[0] == 1250.0   # separator
    assert v[1] == 1250.0   # outer barrier
    assert v[5] == 1250.0   # plunger held at baseline (no third dot)
    # The two active plungers biased below baseline to host dots
    assert v[6] == 1050.0
    assert v[7] == 1150.0
