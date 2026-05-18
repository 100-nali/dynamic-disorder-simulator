"""
Calibrated dot-forming voltage sets for the IST_10721_S14 device.

Each helper returns a numpy array of length 12 (one entry per gate) that
puts the device in a specific dot configuration under the iterative SC
solver. Calibrated empirically by sweeping the relevant plunger gates and
checking the dot-detection + capacitance output.

Gate role conventions (from qxcl's FullDeviceRampTaskDTO and the device
exploration in this project):

  index   role
  -----   ----
  0, 2    sensor barriers
  1       sensor plunger
  3, 11   separator gates
  4, 6, 8, 10  qubit barriers
  5, 7, 9      qubit plungers  (5 = bottom, 7 = middle, 9 = top)
"""

from __future__ import annotations

import numpy as np


def get_dot_forming_voltages_for_ge_holes_double_dot() -> np.ndarray:
    """
    Few-electron, balanced double-dot operating point on IST_10721_S14
    using the physically-consistent SiGe-hole config
    (`configs/graph_ge_holes.yaml`).

    Resulting dot inventory at these voltages:

      - 1 sensor dot   (occupation ~20)
      - 2 qubit dots   (occupation ~1 each, balanced)

    Calibrated by scanning gate 9 from 1300 -> 2500 mV and selecting the
    point where both qubit dots come out at single-electron occupation.
    The bottom qubit-plunger (gate 5) is suppressed to remove the third
    qubit dot, reducing the system to the standard charge-sensed
    double-dot configuration used in spin-qubit experiments.

    Returns
    -------
    np.ndarray of shape (12,) in mV.
    """
    return np.array([
        1300.0,   # 0  sensor barrier
        2700.0,   # 1  sensor plunger (biases sensor on)
        1300.0,   # 2  sensor barrier
        1800.0,   # 3  separator
        0.0,      # 4  qubit barrier (right end suppressed)
        0.0,      # 5  qubit plunger (right end suppressed)
        1100.0,   # 6  qubit barrier
        1300.0,   # 7  middle qubit plunger
        1200.0,   # 8  qubit barrier
        2300.0,   # 9  top qubit plunger  (depleted from 1300 to balance)
        1000.0,   # 10 qubit barrier
        1800.0,   # 11 separator
    ])
