# Simulator

Charge-stability-diagram (CSD) simulator extracted from [qxcl](https://gitlab.com/quantrolox/main).

Generates `(disorder_field, CSD)` pairs by running a physics-based device-simulation graph (electrostatics → self-consistent Poisson → semi-classical transport) for an externally supplied disorder potential. Designed as a data source for distribution-transformer training in the `dynamic-disorder-potential` project.

## What's here

- **`simulator.injection.ExternalDisorderNode`** — drop-in replacement for the InitialElectroStaticPotential node that accepts an arbitrary disorder field instead of running a built-in donor model. This is the seam where any disorder source plugs in.
- **`simulator.pipeline`** — batched GPU/torch CSD sweep, HDF5 storage, single-process runner.
- **`simulator.pipeline.disorder_sources`** — `DisorderSource` ABC. Plug in your own (random GP, learned model, dataset prior).
- **`simulator.pipeline.inducing_points`** — sample a dense disorder field at N user-specified inducing points → length-N vector (DT input/target).
- **`simulator.pipeline.dataset`** — torch `Dataset` that generates+caches `(inducing_vector, CSD)` pairs on first access and reads from disk on subsequent epochs.
- **`simulator.computational_graph`** — pruned subset of the qxcl graph framework: 7 physics nodes + 1 deep-learning node, no TF/keras dependency.

## What's not here (relative to qxcl)

- TLF (telegraph-fluctuator) sampling
- Hyperprior parameter sampling
- Time trajectories — every sample is a single `(disorder, CSD)` pair
- Cloud / FastAPI / data_generation legacy code
- Unused graph nodes (dummy, mape, depth-first dot detection)

## Install

```bash
poetry install
```

Requires Python ≥ 3.12. The torch-based DL node replaces the original keras one — train your own weights on freshly generated SC iterative-solver data.

## Quick start

```python
from simulator.pipeline.config import load_config
from simulator.pipeline.runner import run

cfg = load_config("configs/pipeline.yaml")
run(cfg)
```

For training, use `simulator.pipeline.dataset.SimulatorDataset` as a torch `Dataset`.
