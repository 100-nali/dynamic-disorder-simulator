# Simulator

Charge-stability-diagram (CSD) simulator extracted from [qxcl](https://gitlab.com/quantrolox/main).

Generates `(disorder_field, CSD)` pairs by running a physics-based device-simulation graph (electrostatics → self-consistent Poisson → semi-classical transport) for an externally supplied disorder potential. Designed to live as a git submodule inside [dynamic-disorder-potential](https://github.com/GWhittle110/dynamic-disorder-potential) and act as the training-data source for distribution-transformer training there.

## What's here

- **`simulator.injection.ExternalDisorderNode`** — drop-in replacement for the `InitialElectroStaticPotential` node that accepts an arbitrary disorder field instead of running a built-in donor model. This is the seam where any disorder source plugs in.
- **`simulator.pipeline`** — batched GPU/torch CSD sweep, HDF5 storage, single-process runner.
- **`simulator.pipeline.disorder_sources`** — `DisorderSource` ABC. Subclass to plug in a random GP, learned generator, dataset prior, etc.
- **`simulator.pipeline.inducing_points`** — sample a dense disorder field at N user-specified inducing points → length-N vector (DT input/target).
- **`simulator.pipeline.dataset`** — torch `Dataset` that generates+caches `(inducing_vector, CSD)` pairs on first access and reads from disk on subsequent epochs.
- **`simulator.computational_graph`** — pruned subset of the qxcl graph framework: 7 physics nodes + 1 deep-learning node, no TF/keras dependency.

## What's not here (relative to qxcl)

- TLF (telegraph-fluctuator) sampling
- Hyperprior parameter sampling
- Time trajectories — every sample is a single `(disorder, CSD)` pair
- Cloud / FastAPI / data_generation legacy code
- Unused graph nodes (dummy, mape, depth-first dot detection)

## How this connects to dynamic-disorder-potential

```
github.com/GWhittle110/dynamic-disorder-potential   ←  parent repo (DT code)
└── src/dynamic_disorder_potential/
    ├── distribution_transformer.py                ←  DT code
    ├── observations.py
    └── simulator/   ──► submodule pin ──► github.com/100-nali/dynamic-disorder-simulator (this repo)
        ├── src/simulator/
        ├── configs/
        └── data/gate_designs/
```

The DT repo stores a commit SHA pointing at this repo. `git submodule update --init --recursive` from inside the DT clone fetches the simulator code at that pinned SHA.

## Install

Requires **Python ≥ 3.12** and a working `kwant` install (use `conda install -c conda-forge kwant tinyarray` on Linux — pip-installing kwant is fragile).

Standalone:

```bash
git clone https://github.com/100-nali/dynamic-disorder-simulator.git
cd dynamic-disorder-simulator
poetry install
```

From inside a DT-repo clone:

```bash
git submodule update --init --recursive
pip install -e src/dynamic_disorder_potential/simulator
```

After install, `from simulator.pipeline.dataset import SimulatorDataset` works from any module in the DT repo.

### Torch / CUDA matching

The `torch (>=2.5,<2.9)` pin in `pyproject.toml` keeps you on a release whose default PyPI wheels are built against **CUDA 12.8** — compatible with CUDA 12.x drivers (`nvidia-smi` shows `CUDA Version: 12.x`). Torch ≥ 2.9 ships wheels built against CUDA 13, which silently falls back to CPU on 12.x drivers and tanks the SC iteration runtime ~15×.

If your driver supports CUDA 13.x or you want a specific torch build, install torch explicitly from the matching wheel index **before** `pip install -e .`:

```bash
# CUDA 12.x drivers (most current Linux setups, including oums-dlgpu1):
pip install 'torch>=2.5,<2.9' --index-url https://download.pytorch.org/whl/cu128

# Verify:
python -c "import torch; print(torch.cuda.is_available())"   # must be True
```

## Day-to-day (working from the DT repo)

**Pull latest simulator changes:**
```bash
git submodule update --remote src/dynamic_disorder_potential/simulator
git add src/dynamic_disorder_potential/simulator
git commit -m "Bump simulator submodule"
```

**Edit simulator code** (the submodule worktree is a normal git repo):
```bash
cd src/dynamic_disorder_potential/simulator
# edit ...
git add . && git commit && git push                # commit + push in the simulator repo
cd ../../../
git add src/dynamic_disorder_potential/simulator   # bump the pin in the DT repo
git commit && git push
```

## Using the simulator from DT training code

```python
from torch.utils.data import DataLoader
from simulator.pipeline.config import PipelineConfig, SweepConfig, InducingPointsConfig
from simulator.pipeline.disorder_sources.base import DisorderSource
from simulator.pipeline.dataset import SimulatorDataset


class MyDisorderSource(DisorderSource):
    """Whatever generates a dense disorder field for your run."""
    def __init__(self, nx, ny):
        self._shape = (nx, ny)

    @property
    def grid_shape(self):
        return self._shape

    def sample(self, rng):
        return rng.standard_normal(self._shape) * 5.0   # mV


cfg = PipelineConfig(
    n_samples=2000,
    output_dir="data/runs/exp_01",
    seed=0,
    graph_config_path="src/dynamic_disorder_potential/simulator/configs/graph.yaml",
    inducing_points=InducingPointsConfig(layout="grid", n_per_side=8),  # 64-D vector
    sweep=SweepConfig(
        plunger_indices=(9, 6), plunger_range=(1000, 2600), n_points=128,
        base_voltage_mV=1400.0, batched=True, sc_chunk_size=100, transport_workers=16,
    ),
    n_workers=1,
)

dataset = SimulatorDataset(cfg, source_factory=lambda: MyDisorderSource(128, 128))
# epoch 0: missing samples generated + cached as HDF5 under cfg.output_dir
# epoch 1+: cache hit, read from disk
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for epoch in range(n_epochs):
    for inducing_vec, csd in loader:
        # inducing_vec: (B, 64) float32  — DT target
        # csd:          (B, 128, 128)    — DT input
        ...
```

For a quick run-everything driver, `simulator.pipeline.runner.run(cfg, source_factory)` generates all `cfg.n_samples` up front (serial or via `mp.Pool` when `cfg.n_workers > 1`).

## Testing it works

End-to-end smoke — builds the real graph, generates one CSD with a dummy Gaussian disorder source, reads back the HDF5, and confirms a cache hit on the second call. ~10s for an 8×8 sweep, ~1–2 min for the full 128×128.

```bash
cd dynamic-disorder-potential
git submodule update --init --recursive
pip install -e src/dynamic_disorder_potential/simulator

cd src/dynamic_disorder_potential/simulator
PYTHONPATH=src python - <<'PY'
from pathlib import Path
import numpy as np

from simulator.pipeline.config import PipelineConfig, SweepConfig, InducingPointsConfig
from simulator.pipeline.disorder_sources.base import DisorderSource
from simulator.pipeline.generate import generate_sample
from simulator.pipeline.storage import read_sample_h5

class GaussianDummySource(DisorderSource):
    def __init__(self, nx, ny, scale_mV=5.0):
        self._shape = (nx, ny); self._scale = scale_mV
    @property
    def grid_shape(self): return self._shape
    def sample(self, rng):
        return rng.standard_normal(self._shape) * self._scale

cfg = PipelineConfig(
    n_samples=1, output_dir=Path("data/runs/smoke"), seed=0,
    graph_config_path=Path("configs/graph.yaml").resolve(),
    inducing_points=InducingPointsConfig(layout="grid", n_per_side=8),
    sweep=SweepConfig(plunger_indices=(9, 6), plunger_range=(1000, 2600),
                      n_points=8, base_voltage_mV=1400.0, batched=True,
                      sc_chunk_size=16, transport_workers=4),
)

# Probe to find the device grid shape.
import yaml
from simulator.computational_graph.graph_config import GraphConfig
from simulator.computational_graph.abstract_computational_graph import ComputationalGraph
with open(cfg.graph_config_path) as f: raw = yaml.safe_load(f)
raw["device_config"]["gate_config"]["gate_design_dir"] = str(Path("data/gate_designs").resolve())
nx, ny = ComputationalGraph(GraphConfig(**raw)).components[0].potential_split.shape[1:3]

out = generate_sample(0, cfg, GaussianDummySource(nx, ny))
s = read_sample_h5(out)
print(f"OK: wrote {out}")
print(f"   disorder: {s['disorder']['field_mV'].shape}, "
      f"inducing: {s['inducing']['values'].shape}, "
      f"csd: {s['csd']['data'].shape}")
PY
```

If it prints `OK: ...` you're wired up. Failure modes you might see:

- `ModuleNotFoundError: No module named 'kwant'` → install kwant from conda-forge (don't pip)
- `ModuleNotFoundError: No module named 'simulator'` → you skipped `pip install -e src/dynamic_disorder_potential/simulator`
- `gate_design_dir not found` → run from inside `src/dynamic_disorder_potential/simulator/` (the gate-design path in `configs/graph.yaml` is relative)
