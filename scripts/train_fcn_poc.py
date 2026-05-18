"""
POC training script for the deep-learning self-consistent-potential node.

Goal
----
Validate end-to-end that `FullyConvolutionalSCModel` can learn to predict the
output of `IterativeSelfConsistentPotentialNode` from the electrostatic-potential
input — at small scale, on the Oxford device, with the canonical disorder source
(RBF, ℓ=100 nm, σ=5 mV).

Pipeline
--------
1. Generate `n_samples` (x_electrostatic, y_self_consistent) pairs on the fly:
     - one fresh disorder draw per sample (`GaussianRandomFieldSource`)
     - voltages = calibrated Oxford operating point + small jitter on the two
       active qubit plungers (gates 6 and 7)
     - x = init_node.compute()  (electrostatic potential, before SC iteration)
     - y = sc_node.compute(input_potential=x)[0]  (iterative SC output)
   Pairs are cached to `out_dir / "data" / "sample_{i:05d}.npz"` and reloaded on
   reruns so the script is restart-safe.

2. Train `FullyConvolutionalSCModel` with the *same normalization the inference
   node uses* (`normalize_image_input_output_pair`):
        x → x / (1.25 * max|x|);  x → sqrt(|x|) * sign(x)
   ensuring the trained checkpoint slots into `SelfConsistentPotentialDeepLearningNode`
   without any normalization drift.

3. Save artifacts in the layout `SelfConsistentPotentialDeepLearningNode` expects:
        out_dir/
          final_model.pt             # torch.save(module, ...)
          configs/graph_config.yaml  # copy of the training graph config
          configs/model_config.yaml  # {"image_slice": 1}
        out_dir/training/
          train_log.csv              # epoch, train_mse, val_mse, lr
          loss_curve.png
          qualitative_triples.png    # 4 held-out (input, pred, target) rows

POC scale (defaults)
--------------------
  n_samples = 1500   (≈ 8 min generation on a single CPU process)
  epochs    = 50
  batch     = 8
  res       = full Oxford grid (330, 500)

Run paper-scale by overriding via CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from simulator import GaussianRandomFieldSource
from simulator.deep_learning.fcn import FullyConvolutionalSCModel
from simulator.deep_learning.normalization import normalize_image_input_output_pair
from simulator.pipeline.sweep import _find_node, build_graph_with_disorder, load_graph_config
from simulator.utils.operating_points import (
    get_dot_forming_voltages_for_oxford_double_dot,
)


# -----------------------------------------------------------------------------
# data generation
# -----------------------------------------------------------------------------

def _make_disorder_source(grid_shape: Tuple[int, int], physical_width_nm: float) -> GaussianRandomFieldSource:
    """Canonical disorder: RBF, ℓ=100 nm, σ=5 mV (see CALIBRATIONS.md)."""
    return GaussianRandomFieldSource(
        grid_shape=grid_shape,
        physical_width_nm=physical_width_nm,
        amplitude_mV=5.0,
        correlation_length_nm=100.0,
        kernel="rbf",
    )


def _jittered_voltages(rng: np.random.Generator, base: np.ndarray, jitter_mV: float = 150.0) -> np.ndarray:
    """
    Base = calibrated Oxford operating point. Jitter only the two active
    qubit plungers (gates 6 and 7) so we sample (1,1)↔(0,1)↔(1,0)↔(0,0)
    neighbourhoods rather than always re-evaluating the same charge state.
    """
    v = base.copy().astype(np.float64)
    v[6] += rng.uniform(-jitter_mV, jitter_mV)
    v[7] += rng.uniform(-jitter_mV, jitter_mV)
    return v


def _generate_one_pair(graph, init_node, sc_node, source: GaussianRandomFieldSource, voltages: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw one disorder field, set it on the init node, compute the
    electrostatic and iterative-SC potentials at `voltages`.
    Returns x (nx, ny), y (nx, ny) as float32 numpy arrays.
    """
    disorder = source.sample(rng).astype(np.float64)
    init_node.set_disorder_field(disorder)

    # 2-D electrostatic potential (mV), shape (nx, ny)
    x = init_node.compute(gate_voltages=voltages)[0].astype(np.float32)

    # Iterative SC reference (mV), same shape; sc_node wants (B, nx, ny)
    y = sc_node.compute(input_potential=x[None, ...])[0][0].astype(np.float32)
    return x, y


def generate_dataset(
    config_path: Path,
    n_samples: int,
    out_dir: Path,
    seed: int = 0,
    voltage_jitter_mV: float = 150.0,
) -> List[Path]:
    """
    Generate (or re-load) `n_samples` (x, y) pairs cached to disk.

    Each pair lives at `out_dir/sample_{i:05d}.npz` with keys `x`, `y`, `v`.
    Returns the list of sample paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("sample_*.npz"))
    if len(existing) >= n_samples:
        print(f"[gen] reusing {n_samples} cached samples from {out_dir}")
        return existing[:n_samples]

    graph_config = load_graph_config(config_path)
    physical_width_nm = float(graph_config.device_config.width)

    graph = build_graph_with_disorder(disorder_field_mV=None, config_path=config_path)
    init_node = graph.components[0]
    sc_node = _find_node(graph, "IterativeSelfConsistentPotentialNode")

    nx, ny = init_node.potential_split.shape[1:3]
    source = _make_disorder_source((nx, ny), physical_width_nm)
    base_v = get_dot_forming_voltages_for_oxford_double_dot()
    assert len(base_v) == graph.num_gates, (
        f"Oxford operating-point has {len(base_v)} gates but graph has {graph.num_gates}"
    )

    print(f"[gen] generating {n_samples - len(existing)} new samples "
          f"(have {len(existing)}); grid={nx}x{ny}, width={physical_width_nm:.0f} nm")
    t0 = time.time()
    paths = list(existing)
    for i in range(len(existing), n_samples):
        # Reproducible per-sample seeding so a restart re-creates the SAME data
        rng = np.random.default_rng((seed, i))
        v = _jittered_voltages(rng, base_v, jitter_mV=voltage_jitter_mV)
        try:
            x, y = _generate_one_pair(graph, init_node, sc_node, source, v, rng)
        except Exception as e:
            print(f"  [gen] sample {i} failed ({type(e).__name__}: {e}); skipping")
            continue
        p = out_dir / f"sample_{i:05d}.npz"
        np.savez_compressed(p, x=x, y=y, v=v)
        paths.append(p)
        if (i + 1) % 50 == 0:
            rate = (i + 1 - len(existing)) / (time.time() - t0)
            eta = (n_samples - i - 1) / max(rate, 1e-9)
            print(f"  [gen] {i+1}/{n_samples}  ({rate:.1f} samples/s, ETA {eta/60:.1f} min)")

    print(f"[gen] done in {(time.time()-t0)/60:.1f} min, {len(paths)} samples on disk")
    return paths


# -----------------------------------------------------------------------------
# dataset / training
# -----------------------------------------------------------------------------

class SCDataset(Dataset):
    """
    Loads cached (x, y) pairs, normalizes them with the same routine the
    inference node uses (`normalize_image_input_output_pair`), and returns
    torch tensors with a channel dim: (1, H, W).
    """

    def __init__(self, paths: List[Path]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        npz = np.load(self.paths[idx])
        x = npz["x"]  # (H, W) float32
        y = npz["y"]
        x_n, y_n = normalize_image_input_output_pair(x, y)
        # normalize_* returns leading-axis batched (1, H, W); strip and re-add channel dim
        x_n = np.asarray(x_n[0], dtype=np.float32)[None, ...]
        y_n = np.asarray(y_n[0], dtype=np.float32)[None, ...]
        return (
            torch.from_numpy(x_n),
            torch.from_numpy(y_n),
            float(np.abs(x).max()),
        )


def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    running, n = 0.0, 0
    for xb, yb, _norm in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.shape[0]
        n += xb.shape[0]
    return running / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device) -> float:
    model.eval()
    running, n = 0.0, 0
    for xb, yb, _norm in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        running += loss.item() * xb.shape[0]
        n += xb.shape[0]
    return running / max(n, 1)


# -----------------------------------------------------------------------------
# artifacts
# -----------------------------------------------------------------------------

def save_inference_bundle(out_dir: Path, model: nn.Module, training_graph_config: Path, image_slice: int) -> None:
    """
    Write the exact layout `SelfConsistentPotentialDeepLearningNode` expects:
      out_dir/final_model.pt
      out_dir/configs/graph_config.yaml
      out_dir/configs/model_config.yaml
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "configs").mkdir(exist_ok=True)
    torch.save(model.cpu(), out_dir / "final_model.pt")
    shutil.copy(training_graph_config, out_dir / "configs" / "graph_config.yaml")
    with open(out_dir / "configs" / "model_config.yaml", "w") as f:
        yaml.safe_dump({"image_slice": image_slice}, f)


def plot_loss_curve(log_path: Path, png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, tr, vl = [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_mse"]))
            vl.append(float(row["val_mse"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, tr, label="train", lw=1.6)
    ax.plot(epochs, vl, label="val", lw=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalized space)")
    ax.set_yscale("log")
    ax.set_title("FCN POC — MSE vs epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def plot_qualitative_triples(model: nn.Module, dataset: SCDataset, indices: List[int], png_path: Path, device: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for r, i in enumerate(indices):
            x, y, _ = dataset[i]
            xb = x[None].to(device)
            yp = model(xb)[0, 0].cpu().numpy()
            xn = x[0].numpy()
            yn = y[0].numpy()

            vmax_xy = max(np.abs(xn).max(), np.abs(yn).max(), np.abs(yp).max())
            for col, (img, title) in enumerate(
                [(xn, "input (norm)"), (yp, "predicted SC (norm)"), (yn, "target SC (norm)")]
            ):
                ax = axes[r, col]
                im = ax.imshow(img.T, origin="lower", cmap="RdBu_r", vmin=-vmax_xy, vmax=vmax_xy)
                ax.set_title(f"sample {i}  —  {title}", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=axes[r, :], shrink=0.7, pad=0.02)
    fig.suptitle("FCN POC — held-out qualitative comparison", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "graph_oxford_device.yaml")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "fcn_poc")
    p.add_argument("--n-samples", type=int, default=1500, help="number of (x, y) pairs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--voltage-jitter-mV", type=float, default=150.0)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    training_dir = args.out / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. data ----
    paths = generate_dataset(
        config_path=args.config,
        n_samples=args.n_samples,
        out_dir=args.out / "data",
        seed=args.seed,
        voltage_jitter_mV=args.voltage_jitter_mV,
    )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(paths))
    n_val = max(1, int(len(paths) * args.val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_paths = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    print(f"[split] train={len(train_paths)}  val={len(val_paths)}")

    train_ds = SCDataset(train_paths)
    val_ds = SCDataset(val_paths)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ---- 2. model + opt ----
    model = FullyConvolutionalSCModel(
        n_blocks=args.n_blocks,
        channels=args.channels,
        kernel_size=3,
        dilation=1,
        n_convs_per_block=2,
    ).to(args.device)
    print(f"[model] FullyConvolutionalSCModel  n_params = {model.n_parameters:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )
    loss_fn = nn.MSELoss()

    # ---- 3. train ----
    log_path = training_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "val_mse", "lr", "wall_s"])

    best_val = float("inf")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_mse = train_epoch(model, train_loader, optimizer, loss_fn, args.device)
        val_mse = eval_epoch(model, val_loader, loss_fn, args.device)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_mse:.6f}", f"{val_mse:.6f}", f"{cur_lr:.6f}", f"{elapsed:.1f}"])
        msg = f"[epoch {epoch:3d}/{args.epochs}] train={train_mse:.5f}  val={val_mse:.5f}  lr={cur_lr:.5f}  (epoch {time.time()-t0:.1f}s, total {elapsed/60:.1f}min)"
        if val_mse < best_val:
            best_val = val_mse
            msg += "  <- new best"
        print(msg)

    # ---- 4. artifacts ----
    save_inference_bundle(args.out, model, args.config, image_slice=1)
    plot_loss_curve(log_path, training_dir / "loss_curve.png")
    qualitative_idx = list(range(min(4, len(val_ds))))
    plot_qualitative_triples(
        model.to(args.device), val_ds, qualitative_idx,
        training_dir / "qualitative_triples.png", device=args.device,
    )
    print(f"\n[done] artifacts in {args.out}")
    print(f"  - final_model.pt + configs/  (load via SelfConsistentPotentialDeepLearningNode)")
    print(f"  - training/loss_curve.png")
    print(f"  - training/qualitative_triples.png")
    print(f"  best val MSE (normalized) = {best_val:.5f}")


if __name__ == "__main__":
    main()
