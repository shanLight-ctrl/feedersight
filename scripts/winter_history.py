"""
Historical winter analysis + future projection for FeederSight.

Fetches 3 past GTA winters from Open-Meteo Archive (free, no key) and
projects forward using the same EV growth assumptions as forecast_risk.py.

Covers:
  - Coldest day, worst storm, peak severity per winter season
  - Per-FSA estimated grid stress during historical cold peaks
  - Future winter stress projections (next 3 seasons)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GTA_LAT, GTA_LON = 43.6532, -79.3832

# Dec–Feb windows for the last 3 seasons
WINTERS = [
    {"label": "Winter 2022–23", "start": "2022-12-01", "end": "2023-02-28"},
    {"label": "Winter 2023–24", "start": "2023-12-01", "end": "2024-02-29"},
    {"label": "Winter 2024–25", "start": "2024-12-01", "end": "2025-02-28"},
]

EV_GROWTH_RATES = {"conservative": 0.20, "moderate": 0.35, "aggressive": 0.50}


def _fetch_winter_weather(start: str, end: str, cache_key: str) -> pd.DataFrame:
    cache = RAW / f"winter_archive_{cache_key}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    params = {
        "latitude": GTA_LAT,
        "longitude": GTA_LON,
        "start_date": start,
        "end_date": end,
        "daily": ",".join([
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
        ]),
        "timezone": "America/Toronto",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("daily", {})

    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temp_min_c": data["temperature_2m_min"],
        "temp_max_c": data["temperature_2m_max"],
        "precip_mm": data["precipitation_sum"],
        "snowfall_cm": data["snowfall_sum"],
        "wind_kmh": data["wind_speed_10m_max"],
    })

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def _severity_score(temp_min: float, wind: float, snow: float, precip: float) -> float:
    score = 0.0
    if temp_min <= -15:
        score += min((-15 - temp_min) / 15, 1.0) * 0.35
    elif temp_min <= 0:
        score += min(-temp_min / 15, 1.0) * 0.15
    score += min(wind / 60, 1.0) * 0.30
    score += min(snow / 5, 1.0) * 0.20
    score += min(precip / 10, 1.0) * 0.15
    return round(min(score, 1.0), 3)


def _classify_event(temp_min: float, wind: float, snow: float) -> str:
    if temp_min <= -15 and snow >= 3:
        return "Blizzard"
    if temp_min <= -15:
        return "Extreme Cold"
    if snow >= 5:
        return "Heavy Snow"
    if wind >= 60:
        return "High Wind"
    if temp_min <= -5:
        return "Cold Snap"
    return "Normal Winter"


def _weather_factor(temp_min: float) -> float:
    """Battery efficiency penalty + heating load at cold temperatures."""
    if temp_min <= -20:
        return 1.45
    if temp_min <= -10:
        return 1.35
    if temp_min <= 0:
        return 1.25
    if temp_min <= 5:
        return 1.15
    return 1.0


def _grid_stress(areas: pd.DataFrame, weather_factor: float) -> float:
    """Mean grid risk ratio across all FSAs under a given weather factor."""
    ev_load = (
        areas["ev_count"]
        * areas["peak_charging_share"]
        * areas["avg_charger_power_kw"]
        * weather_factor
    )
    total_load = areas["base_load_kw"] + ev_load
    risk_ratios = total_load / areas["feeder_capacity_kw"].replace(0, float("nan"))
    return round(float(risk_ratios.mean(skipna=True)), 3)


def _overloaded_count(areas: pd.DataFrame, weather_factor: float, ev_multiplier: float = 1.0) -> int:
    ev_load = (
        areas["ev_count"] * ev_multiplier
        * areas["peak_charging_share"]
        * areas["avg_charger_power_kw"]
        * weather_factor
    )
    total_load = areas["base_load_kw"] + ev_load
    risk_ratios = total_load / areas["feeder_capacity_kw"].replace(0, float("nan"))
    return int((risk_ratios > 1.0).sum())


def analyse_winter(label: str, df: pd.DataFrame, areas: pd.DataFrame) -> dict:
    df = df.copy()
    df["severity"] = df.apply(
        lambda r: _severity_score(r.temp_min_c, r.wind_kmh, r.snowfall_cm, r.precip_mm), axis=1
    )
    df["event"] = df.apply(
        lambda r: _classify_event(r.temp_min_c, r.wind_kmh, r.snowfall_cm), axis=1
    )
    df["weather_factor"] = df["temp_min_c"].apply(_weather_factor)

    worst_idx = df["severity"].idxmax()
    coldest_idx = df["temp_min_c"].idxmin()
    snowiest_idx = df["snowfall_cm"].idxmax()

    worst_day = df.loc[worst_idx]
    coldest_day = df.loc[coldest_idx]
    snowiest_day = df.loc[snowiest_idx]

    peak_factor = float(df["weather_factor"].max())
    avg_factor = float(df["weather_factor"].mean())

    peak_stress = _grid_stress(areas, peak_factor)
    avg_stress = _grid_stress(areas, avg_factor)
    overloaded_at_peak = _overloaded_count(areas, peak_factor)

    cold_days = int((df["temp_min_c"] < -10).sum())
    extreme_cold_days = int((df["temp_min_c"] < -20).sum())
    storm_days = int((df["severity"] >= 0.25).sum())

    return {
        "label": label,
        "coldest_day": {
            "date": str(coldest_day["date"])[:10],
            "temp_min_c": round(float(coldest_day["temp_min_c"]), 1),
            "event": coldest_day["event"],
        },
        "worst_storm": {
            "date": str(worst_day["date"])[:10],
            "temp_min_c": round(float(worst_day["temp_min_c"]), 1),
            "snowfall_cm": round(float(worst_day["snowfall_cm"]), 1),
            "wind_kmh": round(float(worst_day["wind_kmh"]), 1),
            "severity": round(float(worst_day["severity"]), 3),
            "event": worst_day["event"],
        },
        "snowiest_day": {
            "date": str(snowiest_day["date"])[:10],
            "snowfall_cm": round(float(snowiest_day["snowfall_cm"]), 1),
        },
        "season_stats": {
            "cold_days_below_minus10": cold_days,
            "extreme_cold_days_below_minus20": extreme_cold_days,
            "storm_days": storm_days,
            "avg_temp_min_c": round(float(df["temp_min_c"].mean()), 1),
            "total_snowfall_cm": round(float(df["snowfall_cm"].sum()), 1),
        },
        "grid_impact": {
            "peak_weather_factor": round(peak_factor, 2),
            "avg_weather_factor": round(avg_factor, 2),
            "mean_risk_ratio_at_peak": peak_stress,
            "overloaded_fsas_at_peak": overloaded_at_peak,
        },
    }


def project_future_winters(areas: pd.DataFrame, base_year: int = 2025) -> list[dict]:
    """
    Project what the next 3 winters will look like assuming a typical cold winter
    (coldest day around -18°C, several storm events) and moderate EV growth.
    """
    # Baseline: a typical severe Toronto winter (not the worst, not the mildest)
    typical_cold_factor = 1.35    # -15°C night
    severe_cold_factor = 1.45     # -22°C extreme cold event

    projections = []
    for offset in range(1, 4):
        year = base_year + offset
        label = f"Winter {year}–{str(year + 1)[-2:]}"

        season_projections = {}
        for scenario, cagr in EV_GROWTH_RATES.items():
            ev_multiplier = (1 + cagr) ** offset

            overloaded_typical = _overloaded_count(areas, typical_cold_factor, ev_multiplier)
            overloaded_severe = _overloaded_count(areas, severe_cold_factor, ev_multiplier)
            stress_typical = _grid_stress_with_ev(areas, typical_cold_factor, ev_multiplier)
            stress_severe = _grid_stress_with_ev(areas, severe_cold_factor, ev_multiplier)

            season_projections[scenario] = {
                "ev_multiplier": round(ev_multiplier, 2),
                "overloaded_fsas_typical_cold": overloaded_typical,
                "overloaded_fsas_severe_cold": overloaded_severe,
                "mean_risk_ratio_typical": stress_typical,
                "mean_risk_ratio_severe": stress_severe,
            }

        projections.append({
            "label": label,
            "year": year,
            "scenarios": season_projections,
        })

    return projections


def _grid_stress_with_ev(areas: pd.DataFrame, weather_factor: float, ev_multiplier: float) -> float:
    ev_load = (
        areas["ev_count"] * ev_multiplier
        * areas["peak_charging_share"]
        * areas["avg_charger_power_kw"]
        * weather_factor
    )
    total_load = areas["base_load_kw"] + ev_load
    risk_ratios = total_load / areas["feeder_capacity_kw"].replace(0, float("nan"))
    return round(float(risk_ratios.mean(skipna=True)), 3)


def build_winter_report(areas: pd.DataFrame) -> dict:
    print("Fetching historical winter weather from Open-Meteo Archive...")
    history = []
    for w in WINTERS:
        key = w["start"][:7].replace("-", "")
        df = _fetch_winter_weather(w["start"], w["end"], key)
        result = analyse_winter(w["label"], df, areas)
        history.append(result)
        print(f"  {w['label']}: coldest {result['coldest_day']['temp_min_c']}°C, "
              f"{result['grid_impact']['overloaded_fsas_at_peak']} FSAs overloaded at peak")

    print("Projecting future winters...")
    future = project_future_winters(areas)

    report = {
        "history": history,
        "future_projections": future,
        "current_overloaded": int((areas["risk_ratio"] > 1.0).sum()),
    }

    out = PROCESSED / "winter_history.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Saved {out}")
    return report


if __name__ == "__main__":
    areas = pd.read_csv(PROCESSED / "feeder_sight_areas.csv")
    build_winter_report(areas)
