"""
Weather severity engine for FeederSight.

Fetches Open-Meteo 24h forecast for Toronto and computes:
  - severity score per hour (0-1)
  - event type (heat_wave, ice_storm, high_wind, heavy_snow, normal)
  - which FSAs are at elevated weather risk
  - simulated station outage probability per station
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# Toronto centre — weather is uniform enough across GTA for our purposes
GTA_LAT, GTA_LON = 43.6532, -79.3832

SEVERITY_THRESHOLDS = {
    "temp_high": 35,    # °C heat wave
    "temp_low": -15,    # °C ice/freeze risk
    "wind_high": 60,    # km/h high wind
    "precip_high": 10,  # mm/h heavy rain/snow
    "snow_high": 5,     # cm/h heavy snow
}


def fetch_forecast(cache: Path | None = None, force: bool = False) -> dict:
    cache = cache or RAW / "weather_forecast.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    params = {
        "latitude": GTA_LAT,
        "longitude": GTA_LON,
        "current": "temperature_2m,wind_speed_10m,precipitation,weather_code",
        "hourly": "temperature_2m,wind_speed_10m,precipitation,snowfall,weather_code",
        "forecast_days": 2,
        "timezone": "America/Toronto",
    }
    data = requests.get(OPEN_METEO, params=params, timeout=30).json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _event_type(temp: float, wind: float, precip: float, snow: float, code: int) -> str:
    if temp >= SEVERITY_THRESHOLDS["temp_high"]:
        return "heat_wave"
    if temp <= SEVERITY_THRESHOLDS["temp_low"] or snow >= SEVERITY_THRESHOLDS["snow_high"]:
        return "ice_storm"
    if wind >= SEVERITY_THRESHOLDS["wind_high"]:
        return "high_wind"
    if precip >= SEVERITY_THRESHOLDS["precip_high"]:
        return "heavy_rain"
    if code in (71, 73, 75, 77, 85, 86):   # WMO snow codes
        return "snowfall"
    return "normal"


def _severity(temp: float, wind: float, precip: float, snow: float) -> float:
    t = SEVERITY_THRESHOLDS
    score = 0.0
    if temp >= t["temp_high"]:
        score += min((temp - t["temp_high"]) / 10, 1.0) * 0.35
    if temp <= t["temp_low"]:
        score += min((t["temp_low"] - temp) / 15, 1.0) * 0.35
    score += min(wind / t["wind_high"], 1.0) * 0.30
    score += min(precip / t["precip_high"], 1.0) * 0.20
    score += min(snow / t["snow_high"], 1.0) * 0.15
    return round(min(score, 1.0), 3)


def analyse_forecast(data: dict) -> dict:
    """Return current conditions + next-24h peak severity summary."""
    cur = data.get("current", {})
    hourly = data.get("hourly", {})

    current_severity = _severity(
        cur.get("temperature_2m", 10),
        cur.get("wind_speed_10m", 0),
        cur.get("precipitation", 0),
        0,
    )
    current_event = _event_type(
        cur.get("temperature_2m", 10),
        cur.get("wind_speed_10m", 0),
        cur.get("precipitation", 0),
        0,
        cur.get("weather_code", 0),
    )

    hours = len(hourly.get("temperature_2m", []))
    severities = []
    events = []
    for i in range(min(24, hours)):
        t = hourly["temperature_2m"][i]
        w = hourly["wind_speed_10m"][i]
        p = hourly["precipitation"][i]
        s = hourly.get("snowfall", [0] * hours)[i]
        c = hourly.get("weather_code", [0] * hours)[i]
        severities.append(_severity(t, w, p, s))
        events.append(_event_type(t, w, p, s, c))

    peak_severity = max(severities) if severities else current_severity
    peak_event = events[severities.index(peak_severity)] if severities else current_event

    non_normal = [e for e in events if e != "normal"]
    alert_active = peak_severity >= 0.25

    return {
        "current_temp_c": cur.get("temperature_2m"),
        "current_wind_kmh": cur.get("wind_speed_10m"),
        "current_precip_mm": cur.get("precipitation"),
        "current_severity": current_severity,
        "current_event": current_event,
        "peak_severity_24h": peak_severity,
        "peak_event_24h": peak_event,
        "alert_active": alert_active,
        "alert_message": _alert_message(peak_event, peak_severity) if alert_active else None,
        "hourly_severity": severities[:24],
        "hourly_times": hourly.get("time", [])[:24],
    }


def _alert_message(event: str, severity: float) -> str:
    level = "WARNING" if severity < 0.6 else "SEVERE"
    messages = {
        "heat_wave":   f"{level}: Heat wave — increased AC load may strain feeders",
        "ice_storm":   f"{level}: Ice/freeze conditions — risk of equipment failure and line damage",
        "high_wind":   f"{level}: High winds — overhead line disruption risk",
        "heavy_rain":  f"{level}: Heavy rain — flooding risk at ground-level equipment",
        "snowfall":    f"{level}: Heavy snowfall — access and equipment risk",
        "normal":      "Conditions normal",
    }
    return messages.get(event, f"{level}: Adverse weather detected")


def station_outage_probability(stations: pd.DataFrame, weather: dict) -> pd.DataFrame:
    """
    Simulate per-station outage probability from weather severity + station type.
    Not real telemetry — this is a planning model.
    """
    sev = weather["peak_severity_24h"]
    event = weather["peak_event_24h"]

    df = stations.copy()

    # Base outage probability from weather severity
    base_p = sev * 0.4

    # DC Fast chargers are more vulnerable to power fluctuations
    type_multiplier = df["type"].apply(lambda t: 1.4 if t == "DC Fast" else 1.0)

    # Already "Not started" stations are flagged as offline regardless
    already_offline = df["status"].str.lower().str.contains("not started", na=False)

    # Ice/wind events hit outdoor equipment harder
    event_boost = {"ice_storm": 0.25, "high_wind": 0.20, "heavy_rain": 0.10,
                   "heat_wave": 0.15, "snowfall": 0.12, "normal": 0.0}
    boost = event_boost.get(event, 0.0)

    df["outage_probability"] = (base_p + boost) * type_multiplier
    df["outage_probability"] = df["outage_probability"].clip(0, 0.95).round(3)
    df.loc[already_offline, "outage_probability"] = 1.0

    def status_label(row):
        if row["outage_probability"] >= 1.0:
            return "Offline"
        if row["outage_probability"] >= 0.6:
            return "High Risk"
        if row["outage_probability"] >= 0.3:
            return "Degraded"
        return "Operational"

    df["weather_status"] = df.apply(status_label, axis=1)
    return df
