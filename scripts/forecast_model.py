"""
Per-FSA time-series feeder capacity forecasting.
Uses exponential EV growth + seasonal factors + Monte Carlo confidence bands.
No heavy dependencies — pure numpy.
"""
from __future__ import annotations
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# Monthly seasonality index (Jan=0 ... Dec=11)
# Ontario grid: winter demand peaks Dec-Feb, summer AC peak Jul-Aug
SEASONAL = np.array([1.35, 1.30, 1.10, 0.95, 0.90, 0.92,
                     1.00, 1.05, 0.93, 0.90, 0.95, 1.28])

GROWTH_RATES = {
    "low":      0.20,   # 20% YoY EV growth
    "moderate": 0.35,   # 35% — baseline
    "high":     0.50,   # 50% — accelerated adoption
}


def _ev_load_kw(ev_count: float, avg_charger_kw: float,
                peak_share: float, growth: float, months_ahead: int) -> float:
    future_evs = ev_count * ((1 + growth) ** (months_ahead / 12))
    return future_evs * peak_share * avg_charger_kw


def forecast_fsa(
    fsa: str,
    ev_count: float,
    base_load_kw: float,
    feeder_capacity_kw: float,
    avg_charger_kw: float = 7.2,
    peak_share: float = 0.12,
    horizon_months: int = 36,
    n_simulations: int = 200,
) -> dict:
    """
    Returns monthly forecast for one FSA over horizon_months.
    Includes moderate projection + p10/p90 confidence band via Monte Carlo.
    """
    today = date.today()
    months = []
    labels = []
    for i in range(horizon_months + 1):
        d = today + relativedelta(months=i)
        months.append(i)
        labels.append(d.strftime("%b %Y"))

    season = np.array([SEASONAL[(today.month - 1 + i) % 12] for i in range(horizon_months + 1)])

    def _series(growth: float) -> np.ndarray:
        return np.array([
            (base_load_kw + _ev_load_kw(ev_count, avg_charger_kw, peak_share, growth, m)) * season[m]
            for m in months
        ])

    moderate = _series(GROWTH_RATES["moderate"])
    low      = _series(GROWTH_RATES["low"])
    high     = _series(GROWTH_RATES["high"])

    # Monte Carlo: randomise growth rate + peak_share to get p10/p90
    rng = np.random.default_rng(42)
    sims = []
    for _ in range(n_simulations):
        g  = rng.uniform(0.15, 0.55)
        ps = rng.uniform(0.08, 0.18)
        sims.append(np.array([
            (base_load_kw + _ev_load_kw(ev_count, avg_charger_kw, ps, g, m)) * season[m]
            for m in months
        ]))
    sims = np.array(sims)
    p10 = np.percentile(sims, 10, axis=0)
    p90 = np.percentile(sims, 90, axis=0)

    # Find when moderate projection crosses capacity thresholds
    def _months_to_threshold(series: np.ndarray, threshold: float) -> int | None:
        for i, v in enumerate(series):
            if v >= threshold:
                return i
        return None

    months_to_80  = _months_to_threshold(moderate, feeder_capacity_kw * 0.80)
    months_to_90  = _months_to_threshold(moderate, feeder_capacity_kw * 0.90)
    months_to_100 = _months_to_threshold(moderate, feeder_capacity_kw)

    def _label(m):
        if m is None:
            return "Beyond 3 years"
        d = today + relativedelta(months=m)
        return d.strftime("%b %Y")

    urgency = "Safe"
    if months_to_100 and months_to_100 <= 12:
        urgency = "Critical"
    elif months_to_100 and months_to_100 <= 24:
        urgency = "High"
    elif months_to_90 and months_to_90 <= 24:
        urgency = "Elevated"
    elif months_to_80 and months_to_80 <= 36:
        urgency = "Moderate"

    current_ratio = moderate[0] / feeder_capacity_kw

    return {
        "fsa":              fsa,
        "labels":           labels,
        "moderate":         [round(v, 1) for v in moderate],
        "low":              [round(v, 1) for v in low],
        "high":             [round(v, 1) for v in high],
        "p10":              [round(v, 1) for v in p10],
        "p90":              [round(v, 1) for v in p90],
        "feeder_capacity":  feeder_capacity_kw,
        "current_load":     round(moderate[0], 1),
        "current_ratio":    round(current_ratio, 3),
        "months_to_80pct":  months_to_80,
        "months_to_90pct":  months_to_90,
        "months_to_overload": months_to_100,
        "overload_date_moderate": _label(months_to_100),
        "overload_date_early":    _label(_months_to_threshold(high, feeder_capacity_kw)),
        "urgency":          urgency,
        "horizon_months":   horizon_months,
    }


def forecast_all(areas_df, horizon_months: int = 36) -> list[dict]:
    results = []
    for _, row in areas_df.iterrows():
        results.append(forecast_fsa(
            fsa=row["fsa"],
            ev_count=float(row.get("ev_count", 0) or 0),
            base_load_kw=float(row.get("base_load_kw", 5000) or 5000),
            feeder_capacity_kw=float(row.get("feeder_capacity_kw", 8000) or 8000),
            avg_charger_kw=float(row.get("avg_charger_power_kw", 7.2) or 7.2),
            peak_share=float(row.get("peak_charging_share", 0.12) or 0.12),
            horizon_months=horizon_months,
        ))
    return results
