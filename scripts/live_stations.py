"""
Fetch live EV charger availability from Open Charge Map API.
Free tier works without a key but rate-limits. Set OCM_API_KEY in .env for production.
"""
from __future__ import annotations
import os
import time
import requests

OCM_BASE = "https://api.openchargemap.io/v3/poi/"

# OCM status type IDs
_STATUS = {
    0:  "Unknown",
    10: "Available",
    20: "Partially Available",
    30: "Unavailable",
    50: "In Use",
    75: "Temporarily Unavailable",
    100: "Planned",
    150: "Removed",
    200: "Operational",
}

# Simplified status for the UI
def _simple_status(status_id: int) -> str:
    if status_id == 10:   return "Available"
    if status_id == 50:   return "In Use"
    if status_id in (20,): return "Partial"
    if status_id in (30, 75): return "Unavailable"
    if status_id in (150,): return "Removed"
    return "Unknown"

# 5-minute in-process cache keyed by (lat, lon, radius)
_cache: dict = {}
_cache_ts: dict = {}
CACHE_TTL = 300


def fetch_live(lat: float = 43.75, lon: float = -79.45, radius_km: float = 60, max_results: int = 500) -> dict:
    key = (round(lat, 2), round(lon, 2), radius_km)
    now = time.time()
    if key in _cache and (now - _cache_ts.get(key, 0)) < CACHE_TTL:
        return _cache[key]

    api_key = os.getenv("OCM_API_KEY", "").strip()
    params: dict = {
        "latitude": lat,
        "longitude": lon,
        "distance": radius_km,
        "distanceunit": "km",
        "maxresults": max_results,
        "countrycode": "CA",
        "compact": False,
        "verbose": False,
        "output": "json",
    }
    if api_key:
        params["key"] = api_key

    try:
        r = requests.get(OCM_BASE, params=params, timeout=12)
        if r.status_code == 403:
            return {
                "error": "OCM_API_KEY_REQUIRED",
                "message": "Open Charge Map requires a free API key. Register at openchargemap.org/site/profile then add OCM_API_KEY=your_key to .env",
                "stations": [], "count": 0,
            }
        r.raise_for_status()
        raw = r.json()
    except Exception as exc:
        return {"error": str(exc), "stations": [], "count": 0}

    stations = []
    for poi in raw:
        try:
            addr = poi.get("AddressInfo") or {}
            lat_p = addr.get("Latitude")
            lon_p = addr.get("Longitude")
            if not lat_p or not lon_p:
                continue

            connections = poi.get("Connections") or []
            connector_types = []
            seen = set()
            max_kw = 0.0
            for c in connections:
                ct = (c.get("ConnectionType") or {}).get("Title", "")
                if ct and ct not in seen:
                    connector_types.append(ct)
                    seen.add(ct)
                kw = c.get("PowerKW") or 0
                if kw > max_kw:
                    max_kw = kw

            status_type = poi.get("StatusType") or {}
            status_id = status_type.get("ID", 0)

            op = poi.get("OperatorInfo") or {}
            network = op.get("Title") or "Unknown"

            stations.append({
                "id":              poi.get("ID"),
                "name":            addr.get("Title", ""),
                "lat":             lat_p,
                "lon":             lon_p,
                "network":         network,
                "connector_types": connector_types,
                "max_kw":          round(max_kw, 1) if max_kw else None,
                "num_points":      poi.get("NumberOfPoints") or len(connections) or 1,
                "status_id":       status_id,
                "status":          _simple_status(status_id),
                "last_verified":   (poi.get("DateLastVerified") or "")[:10],
                "usage_cost":      poi.get("UsageCost") or "—",
                "address":         addr.get("AddressLine1", ""),
                "city":            addr.get("Town", ""),
            })
        except Exception:
            continue

    result = {"stations": stations, "count": len(stations), "fetched_at": int(now)}
    _cache[key] = result
    _cache_ts[key] = now
    return result
