"""
Orchestrate generation of `cfg.n_samples` samples.

Serial or multi-process, idempotent across re-runs (sample_NNNNNN.h5 on disk
is the source of truth — the manifest is just an append-only log).
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Callable

from simulator.pipeline.config import PipelineConfig
from simulator.pipeline.disorder_sources.base import DisorderSource
from simulator.pipeline.generate import generate_sample
from simulator.pipeline.storage import sample_path


# Module-level holder so worker processes can rebuild the disorder source
# without pickling it (some sources are not picklable, e.g. torch generators).
_WORKER_SOURCE: DisorderSource | None = None


def _init_worker(source_factory: Callable[[], DisorderSource]) -> None:
    global _WORKER_SOURCE
    _WORKER_SOURCE = source_factory()


def _worker(args):
    sample_id, cfg = args
    try:
        assert _WORKER_SOURCE is not None, "worker source not initialised"
        return ("ok", sample_id, generate_sample(sample_id, cfg, _WORKER_SOURCE))
    except Exception as e:  # noqa: BLE001
        return ("err", sample_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def run(
    cfg: PipelineConfig,
    source_factory: Callable[[], DisorderSource],
) -> dict:
    """
    Generate `cfg.n_samples` samples.

    `source_factory` returns a `DisorderSource` instance. With `n_workers=1`
    it's called once in the main process. With `n_workers > 1` it's called
    once per worker — this sidesteps the need to pickle the source itself.

    Returns `{n_done, n_skipped, n_failed, failures}`.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    already = {
        i for i in range(cfg.n_samples)
        if sample_path(cfg.output_dir, i).exists()
    }
    todo = [i for i in range(cfg.n_samples) if i not in already]
    n_skip = cfg.n_samples - len(todo)

    print(
        f"[runner] n_samples={cfg.n_samples}, already_done={n_skip}, "
        f"to_run={len(todo)}, n_workers={cfg.n_workers}"
    )

    n_done = 0
    failures: list = []

    if cfg.n_workers <= 1:
        source = source_factory()
        for sid in todo:
            try:
                out = generate_sample(sid, cfg, source)
                print(f"[runner]   sample {sid:>6} -> {out}")
                n_done += 1
            except Exception as e:  # noqa: BLE001
                payload = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(f"[runner]   sample {sid:>6} FAILED:\n{payload}")
                failures.append((sid, payload))
    else:
        with mp.Pool(
            cfg.n_workers,
            initializer=_init_worker,
            initargs=(source_factory,),
        ) as pool:
            for kind, sid, payload in pool.imap_unordered(
                _worker, [(s, cfg) for s in todo]
            ):
                if kind == "ok":
                    print(f"[runner]   sample {sid:>6} -> {payload}")
                    n_done += 1
                else:
                    print(f"[runner]   sample {sid:>6} FAILED:\n{payload}")
                    failures.append((sid, payload))

    summary = {
        "n_done": n_done,
        "n_skipped": n_skip,
        "n_failed": len(failures),
        "failures": failures,
    }
    print(f"[runner] done. ok={n_done}, skipped={n_skip}, failed={len(failures)}")
    return summary
