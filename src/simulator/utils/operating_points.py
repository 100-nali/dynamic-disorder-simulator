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
  6, 7, 5      qubit plungers

The two qubit dots in the (1, 1) operating point sit at COM y=191 and y=233
(in pixel coords on the 330x500 grid). At carrier_depth=47 these are hosted
by v[5] (y=191 dot, the BOTTOM plunger in the canonical naming) and v[7]
(y=233 dot, the CENTRAL plunger); v[6] tunes their relative depth. At
carrier_depth=40 the dot-to-gate mapping was different (v[6] and v[7]
hosted the two dots and v[5] was inert at baseline) — the weaker gate
coupling at larger d shifts which plunger geometry dominates each well.
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
    Coulomb-blockade regime**.

    This calibration is for `carrier_depth = 47 nm` (the value that matches
    the Jirovec et al. 2011.13755 Ge/SiGe z-stack; see CALIBRATIONS.md). The
    previous calibration for `carrier_depth = 40 nm` produced a different
    voltage vector — see git history if you ever revert the config.

    Resulting dot inventory:
      - 1 sensor dot   (occupation 2, deep Coulomb blockade)
      - 2 qubit dots   (occupation 1 each, balanced)

    Calibration journey at d=47 (full notes in CALIBRATIONS.md):
      1. The d=40 voltages give NOTHING at d=47 (weaker gate coupling +
         non-uniform per-gate scaling — pure beta * V_old does not work).
      2. Uniform-baseline scan: the qubit-array baseline that supports
         dots dropped from 1250 (d=40) to 1100 (d=47).
      3. At base=1100, qubit-array barriers v[4]=v[8] need to be raised
         from 1250 to 1300 to confine two separate wells (otherwise the
         qubit region forms a single ~20 e puddle).
      4. Sensor depletion: v[10] needs to be 3100 (vs 2800 at d=40); v[2]
         and v[9] tightened to 1350 (vs 1500 at d=40 — slightly looser).
      5. Bottom plunger v[5] becomes active at the lower baseline and
         hosts a spurious 3 e dot at y=191. Raise v[5] to 1300 to deplete
         it (at d=40 with base=1250 v[5] was inert at 1250).
      6. Final dot tuning: v[6]=900, v[7]=700 to land (1, 1). Three (1,1)
         configs exist in a stable plateau; this one sits at the center.

    Final voltage vector (mV):
      gate:  0    1    2    3    4    5    6    7    8    9    10
      v:   1100 1100 1350 1100 1300 1300 900  700  1300 1350 3100

    Returns
    -------
    np.ndarray of shape (11,) in mV.
    """
    return np.array([
        1100.0,   # 0  separator (central screening bar) - baseline
        1100.0,   # 1  qubit barrier (outer bottom) - baseline
        1350.0,   # 2  sensor barrier (raised to tighten sensor well)
        1100.0,   # 3  qubit barrier (outer top) - baseline
        1300.0,   # 4  qubit barrier (between plungers 7 and 5) - raised
        1300.0,   # 5  bottom plunger (raised to deplete spurious dot)
        900.0,    # 6  qubit plunger (top of active pair)
        700.0,    # 7  qubit plunger (central)
        1300.0,   # 8  qubit barrier (between plungers 6 and 7) - raised
        1350.0,   # 9  sensor barrier (raised to tighten sensor well)
        3100.0,   # 10 sensor plunger (depletes sensor to 2 e)
    ])
