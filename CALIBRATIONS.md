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
| Config | `configs/graph_oxford_device.yaml` (`carrier_depth = 47 nm`) |
| Image | `data/gate_designs/Oxford_Device_Disorder_Colored.png` (cleaned + merged) |
| Result | 1 sensor (2 e, Coulomb-blockade regime) + 2 qubit dots (1 e each, balanced) |
| Gates | 11 |

### z-stack rationale (carrier_depth = 47 nm)

The simulator's gate-image-plane → 2DHG distance enters via the Davies (1995)
Laplace kernel as `d = (carrier_depth + 5) nm`. Matching Jirovec et al.
2011.13755's Ge/SiGe-hole heterostructure:

| Layer (top → bottom) | Thickness |
|---|---|
| Ti/Pd gate metal | 30 nm |
| ALD Al₂O₃ oxide | **20 nm** |
| Si cap | **2 nm** |
| Si₀.₃Ge₀.₇ top barrier | **20 nm** |
| Ge QW (to well center) | **10 nm** |
| **Gate-bottom → QW-center** | **52 nm** |

`carrier_depth = 47 nm` ⇒ effective `d = 52 nm` ✓. (The previous configuration
used `carrier_depth = 40 nm` ⇒ `d = 45 nm`, a ~14 % underestimate.)

### Calibration journey (carrier_depth = 47 nm)

| Step | Action | Result |
|---|---|---|
| 0 | Try the old d=40 voltages verbatim at d=47 | Nothing — qubit array fully depleted everywhere |
| 1 | Try pure scaling `V = β · V_d40` (β ∈ {0.85 … 1.5}) | β ≥ 1.0 gives nothing (Davies kernel is gate-size-dependent — pure scaling doesn't restore the calibration) |
| 2 | Uniform-baseline scan | At d=47 the qubit-array baseline that supports dots is **1100 mV** (was 1250 at d=40) |
| 3 | Raise qubit-array barriers `v[4] = v[8] = 1300` | Two SEPARATE qubit wells form (otherwise ~20 e puddle) |
| 4 | Raise sensor plunger `v[10] = 3100`, barriers `v[2] = v[9] = 1350` | Sensor → 2 e |
| 5 | Raise bottom plunger `v[5] = 1300` | Removes the spurious 3 e dot under v[5] at y=191 (v[5] becomes active at the lower baseline; was inert at the d=40 baseline of 1250) |
| 6 | Tune `v[6] = 900, v[7] = 700` | **Sensor 2 e, qubits (1, 1)** — final calibration (center of a (1,1) plateau spanning v[6] ∈ {900, 1000, 1100}) |

### Gate roles

```
indices    role
-------    ----
0          central screening / separator bar
2, 9       sensor barriers
10         sensor plunger
3, 8, 4, 1   qubit barriers (3 = outer top, 1 = outer bottom)
6, 7, 5    qubit plungers
```

At `carrier_depth = 47`, the two (1, 1) qubit dots sit at pixel COM y=191
(under v[5]) and y=233 (under v[7]); v[6] tunes their relative depth. At
`carrier_depth = 40`, v[6] and v[7] hosted the dots and v[5] was inert at
baseline 1250 — the weaker gate coupling at larger d shifts which plunger
geometry dominates each well.

### Final voltages (mV)

```
gate:   0    1    2    3    4    5    6    7    8    9    10
v:    1100 1100 1350 1100 1300 1300 900  700  1300 1350 3100
```

### Notes

- Pure voltage scaling between `d=40` and `d=47` does NOT work because the
  Davies kernel falloff is gate-size-dependent: small gates lose more coupling
  than large ones (kernel ~ 1/2 for L ≫ d, ~ LW/d² for L ≪ d). The Oxford
  qubit-array gates are tiny (~30 nm), so a 7 nm depth bump is a much larger
  fractional loss for them than for the sensor/screening gates.
- The Oxford qubit array is *very* sensitive to sensor-region gate voltages:
  every step that raises a sensor gate had to be compensated by lowering qubit
  plungers further. The two regions are not independent under this geometry.
- The original PNG had two slightly-different colors for the central screening
  bar (gates 0 and 1 in the raw load) that I merged into a single physical
  gate (now index 0). See `clean_oxford_image.py` for the preprocessing.

---

## Canonical disorder source for data generation

For training the distribution transformer (and the DL SC node when we build
it), the **default disorder source** is:

```python
from simulator import GaussianRandomFieldSource

source = GaussianRandomFieldSource(
    grid_shape=(330, 500),           # Oxford device grid (or 346x346 for IST)
    physical_width_nm=1000.0,        # Oxford (880 for IST)
    amplitude_mV=5.0,
    correlation_length_nm=100.0,     # 10% of device width — features ~plunger-size
    kernel="rbf",                    # smooth squared-exponential
)
```

Rationale for ℓ = 100 nm:
- Larger than a single qubit dot (~30 nm) — disorder shifts the *potential
  landscape* rather than the *fine structure* of an individual dot.
- Smaller than the inter-dot spacing (~150–200 nm) — disorder still couples
  the two qubit dots differently (which is what the DT learns to invert).
- Larger than the GMRF generator's smallest meaningful scale (~2× pixel size,
  i.e. ~6 nm) — well-resolved by the simulator's spatial grid.

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
