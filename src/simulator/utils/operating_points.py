"""
Calibrated dot-forming voltage sets, one helper per device.

Each helper returns a numpy array (length matches the device's gate count)
that puts the device in a specific dot configuration under the iterative SC
solver. Calibrated empirically by sweeping the relevant plunger gates and
checking the dot-detection + capacitance output.

IST_10721_S14 gate-role conventions
-----------------------------------
12 gates total.

  index   role
  -----   ----
  0, 2    sensor barriers
  1       sensor plunger
  3, 11   separator gates
  4, 6, 8, 10  qubit barriers
  5, 7, 9      qubit plungers  (5 = bottom, 7 = middle, 9 = top)

Oxford (graph_oxford_device.yaml) gate-role conventions
-------------------------------------------------------
11 gates total. The screening bar (originally split as 2 colors in the
source PNG) was merged into a single gate at index 0.

  index   role
  -----   ----
  0       central screening / separator bar
  2, 9    sensor barriers
  10      sensor plunger
  3, 8, 4, 1   qubit barriers (3 = outer top, 1 = outer bottom)
  6, 7, 5      qubit plungers  (6 = top, 7 = central+smallest, 5 = bottom)
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


def get_dot_forming_voltages_for_oxford_double_dot() -> np.ndarray:
    """
    Few-electron, balanced double-dot operating point on the Oxford device
    (`configs/graph_oxford_device.yaml`).

    Resulting dot inventory at these voltages:

      - 1 sensor dot   (occupation ~42 — heavily populated; sensor tune is
                        a separate follow-up if you want it depleted)
      - 2 qubit dots   (occupation 1 each, balanced — formed under qubit
                        plungers 5 and 7 between barriers 8 and 4)

    Calibrated by:
      1. Finding that uniform 1250 mV across all 11 gates puts the device
         in a regime where one qubit dot can form (lower than IST's
         operating point because the Oxford qubit array is more depleted
         at higher uniform V).
      2. Selectively lowering qubit plungers {6, 7} below baseline to pull
         holes into their wells, producing two adjacent dots.
      3. Tuning v6=1050, v7=1150 so both dots come out at single-electron
         (1, 1) without one of them disappearing.

    Plunger 5 (bottom) is held at the 1250 baseline (effectively no extra
    bias), so the third qubit plunger doesn't host a dot. This is the
    standard charge-sensed double-dot operating mode.

    Returns
    -------
    np.ndarray of shape (11,) in mV.
    """
    return np.array([
        1250.0,   # 0  separator (central screening bar)
        1250.0,   # 1  qubit barrier (outer bottom)
        1250.0,   # 2  sensor barrier
        1250.0,   # 3  qubit barrier (outer top)
        1250.0,   # 4  qubit barrier (between plungers 7 and 5)
        1250.0,   # 5  qubit plunger (bottom — no extra bias, doesn't host a dot)
        1050.0,   # 6  qubit plunger (top of active pair — biased to form dot)
        1150.0,   # 7  qubit plunger (central, smallest — biased to form dot)
        1250.0,   # 8  qubit barrier (between plungers 6 and 7)
        1250.0,   # 9  sensor barrier
        1250.0,   # 10 sensor plunger
    ])
