"""Claude-powered AI assistant for FeederSight."""

from __future__ import annotations

import json
import math
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearby_stations_context(
    stations_df,
    user_lat: float | None,
    user_lon: float | None,
    radius_km: float = 10,
    limit: int = 5,
) -> str:
    if user_lat is None or stations_df is None or stations_df.empty:
        return "No location data available."

    rows = []
    for _, s in stations_df.iterrows():
        if not s.get("latitude") or not s.get("longitude"):
            continue
        dist = _haversine(user_lat, user_lon, s["latitude"], s["longitude"])
        if dist <= radius_km:
            rows.append((dist, s))

    if not rows:
        return f"No charging stations found within {radius_km} km of your location."

    rows.sort(key=lambda x: x[0])
    lines = []
    for dist, s in rows[:limit]:
        status = s.get("weather_status", "Unknown")
        lines.append(
            f"- {s.get('name','Station')} ({dist:.1f} km away): "
            f"{s.get('machine_count', 1)} machines, "
            f"{s.get('type','—')}, {s.get('avg_power_kw', 7.2):.0f} kW avg, "
            f"Status: {status}"
        )
    return "\n".join(lines)


def _fsa_context(areas_df, user_lat: float | None, user_lon: float | None) -> str:
    if user_lat is None or areas_df is None or areas_df.empty:
        return ""
    best, best_dist = None, float("inf")
    for _, row in areas_df.iterrows():
        d = _haversine(user_lat, user_lon, row["latitude"], row["longitude"])
        if d < best_dist:
            best_dist, best = d, row
    if best is None:
        return ""
    return (
        f"User's nearest FSA: {best['fsa']} ({best.get('area','')}) — "
        f"Risk: {best.get('risk_level','?')}, "
        f"Outage probability: {int((best.get('outage_probability', 0) or 0) * 100)}%, "
        f"Rate limit: {best.get('rate_limit_label','—')}, "
        f"Forecast: {best.get('forecast_urgency','—')}"
    )


def ask(
    question: str,
    weather: dict,
    areas_df=None,
    stations_df=None,
    user_profile: dict | None = None,
    user_lat: float | None = None,
    user_lon: float | None = None,
    radius_km: float = 10,
) -> str:
    nearby = _nearby_stations_context(stations_df, user_lat, user_lon, radius_km)
    fsa_ctx = _fsa_context(areas_df, user_lat, user_lon)

    car_ctx = ""
    if user_profile and user_profile.get("car_model"):
        car_ctx = (
            f"User's car: {user_profile['car_model']}\n"
            f"Current battery: {user_profile.get('battery_pct', '?')}%\n"
            f"Charge target: {user_profile.get('charge_target', 80)}%\n"
            f"Reachable range: {user_profile.get('reachable_km', '?')} km"
        )

    system_prompt = f"""You are FeederSight, an AI assistant that helps EV drivers find safe charging stations and understand grid conditions in the Greater Toronto Area.

You have access to real-time data. Be concise, helpful, and always prioritise safety. Use plain English — no jargon. If a charger is risky or offline, say so clearly.

CURRENT CONDITIONS:
- Weather: {weather.get('current_temp_c')}°C, {(weather.get('current_event') or 'normal').replace('_',' ')}
- Weather alert: {weather.get('alert_message') or 'None'}
- Peak severity next 24h: {int((weather.get('peak_severity_24h') or 0) * 100)}%

{('USER PROFILE:\n' + car_ctx) if car_ctx else ''}

USER'S NEAREST GRID AREA:
{fsa_ctx}

NEARBY CHARGING STATIONS (within {radius_km} km):
{nearby}

Answer the user's question using this data. If they mention a car model and battery %, calculate how far they can go and suggest the best nearby charger. Keep responses under 4 sentences unless a detailed breakdown is genuinely needed."""

    client = _get_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text
