"""Local FastAPI server for the TimesFM 3.0 playground.

Loads the checkpoint once per device in a background thread, then serves
forecasts to the single-page frontend in web/index.html.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import datasets

CHECKPOINT = os.environ.get("TIMESFM_CHECKPOINT", "google/timesfm-3.0-pytorch")
DEFAULT_DEVICE = os.environ.get("TIMESFM_DEVICE", "mps")
WEB_DIR = Path(__file__).parent / "web"

_models: dict[str, object] = {}
_lock = threading.Lock()
_state = {"status": "cold", "detail": "", "load_seconds": None}


def _load(device: str):
    """Return a forecaster for `device`, loading it on first use."""
    with _lock:
        if device in _models:
            return _models[device]
    import timesfm

    t0 = time.time()
    fc = timesfm.TimesFM3Forecaster.from_pretrained(CHECKPOINT, device=device)
    took = time.time() - t0
    with _lock:
        _models[device] = fc
        _state["load_seconds"] = round(took, 1)
    return fc


def _warm():
    """Preload the default device so the first real forecast is not cold."""
    try:
        _state["status"] = "loading"
        fc = _load(DEFAULT_DEVICE)
        fc.predict(np.linspace(0, 1, 512).astype(np.float32), horizon=64, return_quantiles=True)
        _state["status"] = "ready"
    except Exception as exc:  # surfaced verbatim in the UI status lamp
        _state["status"] = "error"
        _state["detail"] = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="TimesFM 3.0 playground", lifespan=lifespan)


class ForecastRequest(BaseModel):
    series_id: str | None = None
    values: list[float] | None = None
    horizon: int = Field(96, ge=1, le=1024)
    context: int = Field(512, ge=32, le=15360)
    backtest: bool = True
    device: str = DEFAULT_DEVICE


def _metrics(actual: np.ndarray, pred: np.ndarray, lo: np.ndarray, hi: np.ndarray,
             naive: np.ndarray) -> dict:
    """Point-accuracy, interval calibration, and skill against seasonal naive."""
    err = pred - actual
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    nonzero = np.abs(actual) > 1e-9
    mae = float(np.mean(np.abs(err)))
    naive_mae = float(np.mean(np.abs(naive - actual)))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
        "smape": float(np.mean(np.abs(err)[denom > 0] / denom[denom > 0]) * 100) if (denom > 0).any() else None,
        "coverage80": float(np.mean((actual >= lo) & (actual <= hi)) * 100),
        "naive_mae": naive_mae,
        "skill": float(1 - mae / naive_mae) if naive_mae > 0 else None,
    }


def _seasonal_naive(context: np.ndarray, horizon: int, period: int) -> np.ndarray:
    """Repeat the last full season; fall back to the last value when aperiodic."""
    if period and period > 0 and len(context) >= period:
        season = context[-period:]
        return np.array([season[i % period] for i in range(horizon)])
    return np.full(horizon, context[-1])


@app.get("/api/meta")
def meta():
    import torch

    ready = _state["status"] == "ready"
    info = {
        "status": _state["status"],
        "detail": _state["detail"],
        "load_seconds": _state["load_seconds"],
        "checkpoint": CHECKPOINT,
        "device": DEFAULT_DEVICE,
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "loaded_devices": sorted(_models.keys()),
    }
    if ready:
        fc = _models[DEFAULT_DEVICE]
        m = fc.model
        info |= {
            "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
            "dtype": str(next(m.parameters()).dtype).replace("torch.", ""),
            "quantiles": list(m.quantiles),
            "max_context": int(fc.global_context),
            "input_patch_len": int(m.input_patch_len),
            "output_patch_len": int(m.output_patch_len),
        }
    return info


@app.get("/api/datasets")
def list_datasets():
    return datasets.catalog()


@app.get("/api/series/{series_id}")
def get_series(series_id: str):
    try:
        return datasets.get(series_id).as_dict()
    except KeyError:
        raise HTTPException(404, f"unknown series {series_id!r}")


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    if _state["status"] == "error":
        raise HTTPException(503, f"model failed to load: {_state['detail']}")

    period = 0
    if req.values is not None:
        full = np.asarray(req.values, dtype=np.float32)
        label = "pasted series"
    elif req.series_id:
        try:
            s = datasets.get(req.series_id)
        except KeyError:
            raise HTTPException(404, f"unknown series {req.series_id!r}")
        full, label, period = np.asarray(s.values, dtype=np.float32), s.name, s.period
    else:
        raise HTTPException(400, "provide either series_id or values")

    if not np.isfinite(full).all():
        raise HTTPException(400, "series contains NaN or infinite values")

    horizon = req.horizon
    if req.backtest:
        if len(full) < horizon + 32:
            raise HTTPException(400, f"series has {len(full)} points; need at least {horizon + 32} to hold out {horizon}")
        actual = full[-horizon:]
        history = full[:-horizon]
    else:
        actual, history = None, full

    context = history[-req.context:].astype(np.float32)
    if len(context) < 32:
        raise HTTPException(400, f"context is {len(context)} points; need at least 32")

    fc = _load(req.device)
    with _lock:
        t0 = time.perf_counter()
        out = fc.predict(context, horizon=horizon, return_quantiles=True)
        elapsed = (time.perf_counter() - t0) * 1000

    q = np.asarray(out.quantiles, dtype=float)  # (horizon, 9) for deciles p10..p90
    point = np.asarray(out.forecast, dtype=float)
    quantiles = {f"{0.1 * (i + 1):.1f}": [round(float(v), 4) for v in q[:, i]] for i in range(q.shape[1])}

    result = {
        "label": label,
        "context": [round(float(v), 4) for v in context],
        "forecast": [round(float(v), 4) for v in point],
        "quantiles": quantiles,
        "actual": None if actual is None else [round(float(v), 4) for v in actual],
        "metrics": None,
        "baseline": None,
        "timing_ms": round(elapsed, 1),
        "device": req.device,
        "context_used": int(len(context)),
        "horizon": horizon,
        "period": period,
    }

    if actual is not None:
        naive = _seasonal_naive(context, horizon, period)
        result["baseline"] = [round(float(v), 4) for v in naive]
        result["metrics"] = _metrics(np.asarray(actual, float), point, q[:, 0], q[:, 8], naive)
    return result


class BenchRequest(BaseModel):
    device: str = DEFAULT_DEVICE
    horizon: int = 128
    contexts: list[int] = [128, 512, 2048, 8192]


@app.post("/api/benchmark")
def benchmark(req: BenchRequest):
    fc = _load(req.device)
    rows = []
    for n in req.contexts:
        ctx = (np.sin(np.arange(n) * 2 * np.pi / 24) * 10 + 100).astype(np.float32)
        with _lock:
            fc.predict(ctx, horizon=req.horizon, return_quantiles=True)  # warm this shape
            runs = []
            for _ in range(3):
                t0 = time.perf_counter()
                fc.predict(ctx, horizon=req.horizon, return_quantiles=True)
                runs.append((time.perf_counter() - t0) * 1000)
        rows.append({"context": n, "ms": round(float(np.median(runs)), 1)})
    return {"device": req.device, "horizon": req.horizon, "rows": rows}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
