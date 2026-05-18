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
    (`configs/graph_oxford_device.yaml`) with **sensor tuned into the
    Coulomb-blockade regime** (sensor occupation 3 e, not 42 e as in the
    initial calibration).

    Resulting dot inventory:
      - 1 sensor dot   (occupation 3, clean charge-blockade regime; ideal
                        for sensing nearby qubit transitions)
      - 2 qubit dots   (occupation 1 each, balanced — formed under
                        qubit plungers 6 and 7 between barriers 8 and 4)

    Calibration steps:
      1. Baseline = uniform 1250 mV (the lowest uniform voltage at which
         the qubit array can host any dots).
      2. Lower qubit plungers v6=1050, v7=1150 to form (1, 1) qubit dots.
      3. Raise sensor plunger v10=2800 to deplete the sensor dot.
      4. Raise sensor barriers v2=v9=1500 to tighten the sensor well.
      5. Since steps 3-4 partially deplete the qubit region too, bump
         qubit plungers down: v6=800, v7=900 (vs the unsensored config's
         1050, 1150) to recover (1, 1).

    Net change from no-sensor-tune version:
      v10: 1250 -> 2800   (sensor plunger up)
      v2, v9: 1250 -> 1500 (sensor barriers up)
      v6: 1050 -> 800     (compensate qubit dot 1)
      v7: 1150 -> 900     (compensate qubit dot 2)

    Returns
    -------
    np.ndarray of shape (11,) in mV.
    """
    return np.array([
        1250.0,   # 0  separator (central screening bar)
        1250.0,   # 1  qubit barrier (outer bottom)
        1500.0,   # 2  sensor barrier (raised to tighten sensor well)
        1250.0,   # 3  qubit barrier (outer top)
        1250.0,   # 4  qubit barrier (between plungers 7 and 5)
        1250.0,   # 5  qubit plunger (bottom — no extra bias, doesn't host a dot)
        800.0,    # 6  qubit plunger (top of active pair — compensated for sensor tune)
        900.0,    # 7  qubit plunger (central — compensated for sensor tune)
        1250.0,   # 8  qubit barrier (between plungers 6 and 7)
        1500.0,   # 9  sensor barrier (raised to tighten sensor well)
        2800.0,   # 10 sensor plunger (depleting bias: sensor goes 42 -> 3 e)
    ])
