"""Demo time series for the TimesFM 3.0 playground.

Every series is either genuinely REAL public data or clearly labelled SYNTHETIC.
Synthetic series are generated from fixed seeds, so a given series id always
yields identical values across runs.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Box & Jenkins airline passengers, monthly totals in thousands, Jan 1949 - Dec 1960.
# Public domain; the canonical trend + multiplicative-seasonality benchmark.
AIRLINE = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]


@dataclasses.dataclass(frozen=True)
class Series:
    """A demo series plus the metadata the frontend needs to render it."""

    id: str
    name: str
    unit: str
    freq: str
    real: bool
    difficulty: str
    blurb: str
    values: list[float]
    period: int  # dominant seasonal period, in samples; 0 if none

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["n"] = len(self.values)
        return d


def _ou(n: int, theta: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck path: mean-reverting noise with realistic persistence."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (-x[i - 1]) + sigma * rng.standard_normal()
    return x


def _grid_load() -> Series:
    rng = np.random.default_rng(11)
    n = 24 * 7 * 10  # ten weeks, hourly
    t = np.arange(n)
    hour, dow = t % 24, (t // 24) % 7

    # Twin weekday peaks: morning ramp ~08:00, evening peak ~19:00.
    morning = 1500 * np.exp(-0.5 * ((hour - 8.5) / 2.0) ** 2)
    evening = 2600 * np.exp(-0.5 * ((hour - 19.0) / 2.4) ** 2)
    night = -1800 * np.exp(-0.5 * ((hour - 4.0) / 3.0) ** 2)
    weekend = np.where(dow >= 5, -1700.0, 0.0)
    # Weekend load is flatter as well as lower.
    shape = np.where(dow >= 5, 0.55, 1.0)

    seasonal_drift = 900 * np.sin(2 * np.pi * t / (24 * 365)) 
    base = 12400 + seasonal_drift + weekend + shape * (morning + evening + night)
    values = base + _ou(n, 0.05, 90, rng) + rng.normal(0, 55, n)
    return Series(
        id="grid-load",
        name="National grid load",
        unit="MW",
        freq="hourly",
        real=False,
        difficulty="easy",
        blurb="Twin weekday peaks, flatter weekends. Strong daily and weekly cycles — the regime foundation models handle best.",
        values=[round(float(v), 1) for v in values],
        period=24,
    )


def _wind_generation() -> Series:
    rng = np.random.default_rng(23)
    n = 24 * 7 * 10
    capacity = 6000.0
    # Wind speed as a persistent OU process around 9 m/s.
    speed = 9.0 + 3.4 * _ou(n, 0.012, 1.0, rng)
    speed = np.clip(speed, 0, None)
    # Turbine power curve: cut-in 3, rated 13, cut-out 25 m/s.
    p = np.clip((speed - 3.0) / (13.0 - 3.0), 0, 1) ** 3
    p[speed < 3.0] = 0.0
    p[speed > 25.0] = 0.0
    values = np.clip(capacity * p + rng.normal(0, 70, n), 0, capacity)
    return Series(
        id="wind-generation",
        name="Wind farm output",
        unit="MW",
        freq="hourly",
        real=False,
        difficulty="hard",
        blurb="Weather-regime persistence through a turbine power curve. No seasonality to lean on — skill decays fast past a day.",
        values=[round(float(v), 1) for v in values],
        period=0,
    )


def _imbalance_price() -> Series:
    rng = np.random.default_rng(37)
    n = 96 * 21  # three weeks at 15-minute settlement
    t = np.arange(n)
    qh = t % 96
    daily = 22 * np.sin(2 * np.pi * (qh - 26) / 96) + 9 * np.sin(4 * np.pi * (qh - 10) / 96)
    base = 62 + daily + 14 * _ou(n, 0.03, 1.0, rng)
    # Scarcity spikes: rare, signed, and short-lived.
    values = base + rng.normal(0, 4.5, n)
    for _ in range(26):
        i = int(rng.integers(0, n - 4))
        mag = float(rng.normal(0, 1)) * 210
        width = int(rng.integers(1, 4))
        values[i : i + width] += mag
    return Series(
        id="imbalance-price",
        name="Imbalance settlement price",
        unit="EUR/MWh",
        freq="15-minute",
        real=False,
        difficulty="brutal",
        blurb="Mean-reverting with fat-tailed scarcity spikes. Point forecasts cannot catch the spikes — watch the quantile fan instead.",
        values=[round(float(v), 2) for v in values],
        period=96,
    )


def _airline() -> Series:
    return Series(
        id="airline",
        name="Airline passengers",
        unit="thousands",
        freq="monthly",
        real=True,
        difficulty="easy",
        blurb="Box & Jenkins, 1949-1960. Real data. Growing trend with multiplicative yearly seasonality — the textbook case.",
        values=[float(v) for v in AIRLINE],
        period=12,
    )


def _web_traffic() -> Series:
    rng = np.random.default_rng(5)
    n = 730  # two years, daily
    t = np.arange(n)
    dow = t % 7
    weekly = np.where(dow >= 5, -0.30, 0.06) * 1.0
    growth = np.log1p(t / 90.0) * 0.55
    # A launch on day 430 steps the baseline up for good.
    step = np.where(t >= 430, 0.34, 0.0)
    level = 8.2 + growth + weekly + step + 0.035 * _ou(n, 0.02, 1.0, rng)
    values = np.exp(level) * np.exp(rng.normal(0, 0.045, n))
    return Series(
        id="web-traffic",
        name="Daily page views",
        unit="views",
        freq="daily",
        real=False,
        difficulty="medium",
        blurb="Weekly weekday/weekend split, log growth, and a permanent step up at a product launch on day 430.",
        values=[round(float(v)) for v in values],
        period=7,
    )


def _sunspots() -> Series:
    rng = np.random.default_rng(71)
    n = 480  # forty years, monthly
    t = np.arange(n)
    phase = 2 * np.pi * t / 132.0  # ~11-year cycle
    # Solar cycles rise faster than they fall; skew the sine accordingly.
    raw = np.sin(phase - 0.55 * np.sin(phase))
    amp = 78 + 26 * np.sin(2 * np.pi * t / 1050.0)
    values = np.clip(amp * (raw + 1) / 2 * 1.7 + 6 * _ou(n, 0.08, 1.0, rng) + rng.normal(0, 6, n), 0, None)
    return Series(
        id="solar-cycle",
        name="Sunspot number",
        unit="count",
        freq="monthly",
        real=False,
        difficulty="medium",
        blurb="An ~11-year cycle with a fast rise and slow decay. Tests whether the model locks onto a period far longer than its patches.",
        values=[round(float(v), 1) for v in values],
        period=132,
    )


def _random_walk() -> Series:
    rng = np.random.default_rng(101)
    n = 600
    steps = rng.normal(0, 1.0, n)
    values = 100 + np.cumsum(steps)
    return Series(
        id="random-walk",
        name="Random walk",
        unit="index",
        freq="daily",
        real=False,
        difficulty="impossible",
        blurb="A driftless random walk. There is no signal to find, so the only correct forecast is flat with a widening fan. A good honesty check.",
        values=[round(float(v), 3) for v in values],
        period=0,
    )


_BUILDERS = [
    _grid_load,
    _wind_generation,
    _imbalance_price,
    _airline,
    _web_traffic,
    _sunspots,
    _random_walk,
]

CATALOG: dict[str, Series] = {}
for _b in _BUILDERS:
    _s = _b()
    CATALOG[_s.id] = _s


def catalog() -> list[dict]:
    """Metadata for every demo series, without the payload of values."""
    out = []
    for s in CATALOG.values():
        d = s.as_dict()
        d.pop("values")
        out.append(d)
    return out


def get(series_id: str) -> Series:
    if series_id not in CATALOG:
        raise KeyError(series_id)
    return CATALOG[series_id]
