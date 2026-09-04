"""Compare every way of running TimesFM 3.0 available in this repo.

Covers the original PyTorch path (MPS and CPU) and the ONNX exports that back
the browser page, so the cost of moving inference into a tab is measurable
rather than assumed.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

HORIZON = 96
WARMUP, RUNS = 2, 7


def series(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    return (np.sin(t * 2 * np.pi / 24) * 10 + 100 + rng.normal(0, 0.5, n)).astype(np.float32)


def timed(fn, warmup: int = WARMUP, runs: int = RUNS) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), min(samples)


def bench_torch(rows: list, contexts: tuple[int, ...]) -> None:
    import torch
    import timesfm

    for device in ("mps", "cpu"):
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        fc = timesfm.TimesFM3Forecaster.from_pretrained(
            "google/timesfm-3.0-pytorch", device=device)
        for ctx in contexts:
            data = series(ctx)
            med, best = timed(lambda: fc.predict(data, horizon=HORIZON, return_quantiles=True))
            rows.append((f"PyTorch fp32 · {device.upper()}", ctx, ctx, med, best))
        del fc


def bench_onnx(rows: list, out: Path, label: str, contexts: tuple[int, ...]) -> None:
    import json

    import onnxruntime as ort

    cfg = json.loads((out / "config.json").read_text())
    window = cfg["window"]
    for name, tag in (("timesfm3-decode.onnx", "fp32"), (cfg["file"], "int8")):
        path = out / name
        if not path.exists():
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        seen = set()
        for ctx in contexts:
            used = min(ctx, window)
            if used in seen:      # contexts past the window all clamp to the same run
                continue
            seen.add(used)
            pad = window - used
            x = np.concatenate([np.zeros(pad, np.float32), series(used)])[None, None, :]
            mk = np.concatenate([np.ones(pad, bool), np.zeros(used, bool)])[None, :]
            med, best = timed(lambda: sess.run(None, {"context": x, "mask": mk}))
            rows.append((f"ONNX {tag} · CPU · {label}", used, window, med, best))
        del sess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 4096])
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()
    contexts = tuple(args.contexts)

    rows: list = []
    if not args.skip_torch:
        bench_torch(rows, contexts)
    for out, label in ((Path("web/models"), "win 512"),
                       (Path("web/models-4096"), "win 4096")):
        if (out / "config.json").exists():
            bench_onnx(rows, out, label, contexts)

    print(f"\nhorizon {HORIZON}, median of {RUNS} runs after {WARMUP} warmups\n")
    print(f"{'setup':<32}{'real ctx':>10}{'computed':>10}{'median':>11}{'best':>10}")
    print("-" * 73)
    for setup, used, computed, med, best in rows:
        print(f"{setup:<32}{used:>10}{computed:>10}{med:>10.0f}ms{best:>9.0f}ms")
    print("\n'computed' is what the graph actually processes: the ONNX exports use a")
    print("fixed window, so a short series still pays for the whole thing.")


if __name__ == "__main__":
    main()
