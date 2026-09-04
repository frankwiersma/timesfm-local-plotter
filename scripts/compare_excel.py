"""Compare TimesFM 3.0 against what you could do in Excel on the same series.

Excel's FORECAST.ETS (and Data > Forecast Sheet) fits an ETS AAA model —
additive error, trend and seasonality, i.e. Holt-Winters. statsmodels'
ExponentialSmoothing with the same configuration is the closest faithful
stand-in, so the comparison is method-to-method rather than button-to-button.

Also included is seasonal naive: "same month last year", which is a single
Excel formula and a genuinely hard baseline on seasonal data.

Usage:
    python scripts/compare_excel.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SCRATCH = Path("/tmp/claude-502/-Users-309212-repos-timesfm/"
                "d6ec7893-15f2-4dba-bd39-adc434894115/scratchpad")
HOLDOUT, PERIOD = 24, 12
ORIGINS, GAP = 6, 6      # rolling-origin evaluation: one split proves nothing

SERIES = [
    ("Aardgasverbruik NL", "cbs-gas.json", "TotaalVerbruik_25"),
    ("Elektriciteitsverbruik NL", "cbs-elec.json", "NettoVerbruikBerekend_30"),
    ("Schiphol passagiers", "cbs-air.json", "TotaalAantalPassagiers_12"),
]


def load(fn: str, key: str) -> np.ndarray:
    rows = json.loads((SCRATCH / fn).read_text())["value"]
    return np.array([float(r[key]) for r in rows if r.get(key) is not None],
                    dtype=np.float32)


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    d = (np.abs(actual) + np.abs(pred)) / 2
    ok = d > 0
    return float(np.mean(np.abs(pred - actual)[ok] / d[ok]) * 100)


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def seasonal_naive(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    season = history[-period:]
    return np.array([season[i % period] for i in range(horizon)])


def holt_winters(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    """The model behind Excel's FORECAST.ETS: additive trend and seasonality."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    fit = ExponentialSmoothing(history.astype(float), trend="add",
                               seasonal="add", seasonal_periods=period,
                               initialization_method="estimated").fit()
    return np.asarray(fit.forecast(horizon))


def timesfm_forecast(history: np.ndarray, horizon: int, forecaster) -> np.ndarray:
    out = forecaster.predict(history.astype(np.float32), horizon=horizon,
                             return_quantiles=False)
    return np.asarray(out.forecast, dtype=float)


def main() -> None:
    import timesfm

    print("loading TimesFM 3.0 ...")
    fc = timesfm.TimesFM3Forecaster.from_pretrained(
        "google/timesfm-3.0-pytorch", device="mps")

    print(f"\nrolling origin: {ORIGINS} forecasts per series, {HOLDOUT}-month "
          f"horizon, origins {GAP} months apart\n")
    print(f"{'series':<26}{'method':<28}{'sMAPE':>8}{'vs naive':>11}{'wins':>7}")
    print("-" * 80)
    for name, fn, key in SERIES:
        values = load(fn, key)
        methods = {"Seasonal naive (1 Excel formula)": seasonal_naive,
                   "Excel FORECAST.ETS (ETS AAA)": holt_winters,
                   "TimesFM 3.0 (zero-shot)": None}
        acc = {k: {"smape": [], "mae": []} for k in methods}
        wins = {k: 0 for k in methods}
        for o in range(ORIGINS):
            end = len(values) - o * GAP
            history, actual = values[:end - HOLDOUT], values[end - HOLDOUT:end]
            if len(history) < 5 * PERIOD:
                continue
            preds = {
                "Seasonal naive (1 Excel formula)": seasonal_naive(history, HOLDOUT, PERIOD),
                "Excel FORECAST.ETS (ETS AAA)": holt_winters(history, HOLDOUT, PERIOD),
                "TimesFM 3.0 (zero-shot)": timesfm_forecast(history, HOLDOUT, fc),
            }
            for k, pr in preds.items():
                acc[k]["smape"].append(smape(actual, pr))
                acc[k]["mae"].append(mae(actual, pr))
            best = min(preds, key=lambda k: mae(actual, preds[k]))
            wins[best] += 1
        base = float(np.mean(acc["Seasonal naive (1 Excel formula)"]["mae"]))
        for k in methods:
            sm = float(np.mean(acc[k]["smape"]))
            sk = (1 - float(np.mean(acc[k]["mae"])) / base) * 100
            sk_s = "—" if k.startswith("Seasonal") else f"{sk:+.0f}%"
            first = name if k.startswith("Seasonal") else ""
            n = len(acc[k]["smape"])
            print(f"{first:<26}{k:<28}{sm:>7.1f}%{sk_s:>11}{wins[k]:>4}/{n}")
        print()


if __name__ == "__main__":
    main()
