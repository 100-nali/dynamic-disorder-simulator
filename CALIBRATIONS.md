# Calibrated operating points

Voltage configurations that have been empirically tuned to produce specific
dot inventories under the iterative SC solver. Each entry below corresponds
to a helper in `simulator/utils/operating_points.py`.

To use programmatically:

```python
from simulator.utils import (
    get_dot_forming_voltages_for_ge_holes_double_dot,
    get_dot_forming_voltages_for_oxford_double_dot,
)
```

---

## IST_10721_S14 — Ge/SiGe-hole double-dot

| | |
|---|---|
| Helper | `get_dot_forming_voltages_for_ge_holes_double_dot()` |
| Config | `configs/graph_ge_holes.yaml` (degeneracy=2, m_eff=0.07 — physically consistent) |
| Result | 1 sensor (20 e) + 2 qubit dots (1 e each, balanced) |
| Gates | 12 |

### Calibration journey

| Step | Action | Result |
|---|---|---|
| 0 | Start from `get_dot_forming_voltages_for_accelerated_graph()` (qxcl) | Sensor 23 e + qubits (10, 2, 1) — over-populated |
| 1 | Zero out `v[4] = v[5] = 0` (suppress one end) | Sensor + 2 qubit dots remaining |
| 2 | Scan `gate 9` (top plunger) from 1300 to 2500 mV | At `v[9]=2300`, top dot drops from 10 → 1 |
| 3 | Lock in calibrated config | **Sensor 20 e, qubits (1, 1)** |

### Gate roles

```
indices    role
-------    ----
0, 2       sensor barriers
1          sensor plunger
3, 11      separator gates
4, 6, 8, 10  qubit barriers
5, 7, 9    qubit plungers (5 = bottom, 7 = middle, 9 = top)
```

### Final voltages (mV)

```
gate:   0     1     2     3     4    5     6     7     8     9    10    11
v:    1300  2700  1300  1800   0    0   1100  1300  1200  2300  1000  1800
```

---

## Oxford device — Ge/SiGe-hole double-dot with sensor in blockade

| | |
|---|---|
| Helper | `get_dot_forming_voltages_for_oxford_double_dot()` |
| Config | `configs/graph_oxford_device.yaml` |
| Image | `data/gate_designs/Oxford_Device_Disorder_Colored.png` (cleaned + merged) |
| Result | 1 sensor (3 e, Coulomb-blockade regime) + 2 qubit dots (1 e each, balanced) |
| Gates | 11 |

### Calibration journey

| Step | Action | Result |
|---|---|---|
| 0 | Uniform 1500 mV across all gates | 1 sensor (2 e) only — qubit array fully depleted |
| 1 | Lower baseline to **1250 mV** | 1 sensor (42 e) + 1 qubit dot (1 e at y=192) |
| 2 | Selectively lower paired plungers `v[6]=v[7]=800` | 1 sensor + 2 qubit dots (forming!) |
| 3 | Tune `v[6]=1050, v[7]=1150` | **Sensor 42 e, qubits (1, 1)** — first balanced config |
| 4 | Raise sensor plunger `v[10]=2800` | Sensor 16 e — but lost one qubit dot |
| 5 | Compensate: lower active plungers further `v[6]=800, v[7]=900` | Sensor 16 e, qubits (1, 1) recovered |
| 6 | Also raise sensor barriers `v[2]=v[9]=1500` | **Sensor 3 e, qubits (1, 1)** — final calibration |

### Gate roles

```
indices    role
-------    ----
0          central screening / separator bar
2, 9       sensor barriers
10         sensor plunger
3, 8, 4, 1   qubit barriers (3 = outer top, 1 = outer bottom)
6, 7, 5    qubit plungers (6 = top of active pair, 7 = central/smallest, 5 = bottom)
```

Note: the user confirmed gates 8 = B, 7 = P, 4 = B; the rest of the B-P-B-P-B-P-B
pattern was inferred from gate sizes (large outermost gates 3, 1 = outer barriers)
and verified empirically.

### Final voltages (mV)

```
gate:   0    1    2    3    4    5    6    7    8    9     10
v:    1250 1250 1500 1250 1250 1250 800  900  1250 1500  2800
```

### Notes

- Plunger 5 (bottom qubit plunger) is held at the 1250 baseline. The bottom
  qubit dot doesn't form there at this bias — standard double-dot operation.
- The Oxford qubit array is *very* sensitive to sensor-region gate voltages:
  every step that raises a sensor gate had to be compensated by lowering qubit
  plungers further. The two regions are not independent under this geometry.
- The original PNG had two slightly-different colors for the central screening
  bar (gates 0 and 1 in the raw load) that I merged into a single physical
  gate (now index 0). See `clean_oxford_image.py` for the preprocessing.

---

## Adding a new device

To add a calibration helper for another device:

1. Add the gate-design PNG to `data/gate_designs/` (per-gate distinct
   grayscale values; the simulator splits by `np.unique(mean(channels))`).
2. Add a `configs/graph_<device>.yaml` config (use `graph_oxford_device.yaml`
   as template).
3. Calibrate empirically (the scripts under `/tmp/oxford_*.py` in the working
   server directory are a worked example: uniform-V baseline scan, then
   selective plunger lowering, then sensor depletion + qubit compensation).
4. Add a `get_dot_forming_voltages_for_<device>_*()` helper in
   `simulator/utils/operating_points.py` with a docstring describing the
   calibration steps + a header table with the gate roles.
5. Add at least two tests in `tests/test_operating_points.py`:
   one for shape/dtype, one asserting the specific calibration values.
6. Append an entry to this file documenting the journey + final voltages.
