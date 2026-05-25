"""Download and cache real public datasets for FeederSight."""

from __future__ import annotations

import io
import json
import zipfile
from codecs import iterdecode
from csv import reader as csv_reader
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise ValueError(
        f"Could not find column. Tried {candidates}. Available: {list(df.columns)}"
    )

GEONAMES_CA_ZIP = "https://download.geonames.org/export/zip/CA.zip"
IESO_FSA_ZIP = (
    "https://reports-public.ieso.ca/public/HourlyConsumptionByFSA/"
    "PUB_HourlyConsumptionByFSA_202601_v1.zip"
)
TORONTO_FSA_API = (
    "https://gis.toronto.ca/arcgis/rest/services/cot_geospatial28/FeatureServer/14/query"
)
OCM_URL = "https://api.openchargemap.io/v3/poi/"
NRCAN_CHARGERS_API = (
    "https://maps-cartes.services.geo.ca/server_serveur/rest/services/"
    "NRCan/ZEVIP_EVAFIDI_CIB-30-06-2025_fr/MapServer/3/query"
)
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

GTA_FSA_PREFIXES = ("L", "M", "N")  # Toronto/GTA-ish Ontario FSAs


def _save(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def fetch_geonames_fsa_centroids(cache: Path | None = None) -> pd.DataFrame:
    """Real FSA lat/lon from GeoNames (open data)."""
    cache = cache or RAW / "fsa_centroids_geonames.csv"
    if cache.exists():
        return pd.read_csv(cache)

    r = requests.get(GEONAMES_CA_ZIP, timeout=90)
    r.raise_for_status()
    rows = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        with zf.open("CA.txt") as bf:
            for row in csv_reader(iterdecode(bf, "utf-8"), delimiter="\t"):
                code = row[1]
                if len(code) != 3:
                    continue
                rows.append(
                    {
                        "fsa": code.upper(),
                        "area": row[2],
                        "province": row[3],
                        "latitude": float(row[9]),
                        "longitude": float(row[10]),
                        "coord_source": "GeoNames CA.zip",
                    }
                )
    df = pd.DataFrame(rows).drop_duplicates("fsa")
    _save(df, cache)
    return df


def fetch_toronto_fsa_centroids(cache: Path | None = None) -> pd.DataFrame:
    """City of Toronto Open Data — FSA polygons with official lat/lon."""
    cache = cache or RAW / "fsa_centroids_toronto.csv"
    if cache.exists():
        return pd.read_csv(cache)

    rows = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "outFields": "AREA_SHORT_CODE,AREA_NAME,LATITUDE,LONGITUDE",
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": 100,
        }
        data = requests.get(TORONTO_FSA_API, params=params, timeout=60).json()
        features = data.get("features") or []
        if not features:
            break
        for f in features:
            a = f["attributes"]
            rows.append(
                {
                    "fsa": a["AREA_SHORT_CODE"],
                    "area": a["AREA_NAME"],
                    "latitude": a["LATITUDE"],
                    "longitude": a["LONGITUDE"],
                    "coord_source": "City of Toronto Open Data",
                }
            )
        offset += len(features)
        if len(features) < 100:
            break

    df = pd.DataFrame(rows)
    _save(df, cache)
    return df


def fetch_ieso_hourly_fsa(cache_csv: Path | None = None) -> pd.DataFrame:
    """IESO hourly consumption by FSA (real smart-meter aggregates)."""
    cache_csv = cache_csv or RAW / "ieso_hourly_fsa_sample.csv"
    cache_zip = RAW / "ieso_hourly_fsa.zip"

    if cache_csv.exists():
        return pd.read_csv(cache_csv)

    if not cache_zip.exists():
        print("Downloading IESO HourlyConsumptionByFSA (~14MB)...")
        r = requests.get(IESO_FSA_ZIP, timeout=180)
        r.raise_for_status()
        cache_zip.write_bytes(r.content)

    with zipfile.ZipFile(cache_zip) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as bf:
            df = pd.read_csv(bf, skiprows=3)
    df.columns = [c.strip() for c in df.columns]
    df["FSA"] = df["FSA"].astype(str).str.upper().str[:3]
    _save(df, cache_csv)
    return df


def aggregate_ieso_by_fsa(hourly: pd.DataFrame) -> pd.DataFrame:
    """Peak load and peak-hour fraction from real IESO hourly kWh."""
    h = hourly.copy()
    h["TOTAL_CONSUMPTION"] = pd.to_numeric(h["TOTAL_CONSUMPTION"], errors="coerce")
    h["PREMISE_COUNT"] = pd.to_numeric(h["PREMISE_COUNT"], errors="coerce")

    by_hour = (
        h.groupby(["FSA", "DATE", "HOUR"], as_index=False)
        .agg(total_kwh=("TOTAL_CONSUMPTION", "sum"), premises=("PREMISE_COUNT", "sum"))
    )
    daily = by_hour.groupby(["FSA", "DATE"], as_index=False).agg(daily_kwh=("total_kwh", "sum"))

    peak_rows = by_hour.loc[by_hour.groupby("FSA")["total_kwh"].idxmax()].copy()
    peak_rows = peak_rows.rename(
        columns={
            "FSA": "fsa",
            "total_kwh": "ieso_peak_hour_kwh",
            "HOUR": "ieso_peak_hour",
            "premises": "premise_count",
        }
    )

    daily_avg = daily.groupby("FSA")["daily_kwh"].mean()

    out = peak_rows[
        ["fsa", "ieso_peak_hour_kwh", "ieso_peak_hour", "premise_count"]
    ].copy()
    out["ieso_avg_daily_kwh"] = out["fsa"].map(daily_avg)
    out["ieso_peak_fraction"] = (
        out["ieso_peak_hour_kwh"] / out["ieso_avg_daily_kwh"].replace(0, pd.NA)
    ).clip(0.05, 0.45)
    # kWh in one hour ≈ average kW over that hour
    out["base_load_kw"] = out["ieso_peak_hour_kwh"]
    out["feeder_capacity_kw"] = out["ieso_peak_hour_kwh"] * 1.25
    out["base_load_source"] = "IESO Hourly Consumption by FSA"
    out["feeder_capacity_source"] = (
        "IESO observed monthly peak hour + 25% planning margin (proxy, not utility nameplate)"
    )
    return out


def _power_kw_from_technology(tech: str) -> float:
    t = (tech or "").lower()
    if "dc" in t or "fast" in t or "350" in t or "100" in t:
        return 50.0
    if "level 3" in t:
        return 50.0
    return 7.2


def fetch_nrcan_chargers_ontario(cache: Path | None = None) -> pd.DataFrame:
    """Natural Resources Canada ZEVIP/EVAFIDI — federal open data, no API key."""
    cache = cache or RAW / "existing_chargers_nrcan.csv"
    if cache.exists():
        return pd.read_csv(cache)

    rows = []
    offset = 0
    while True:
        params = {
            "where": "Province = 'Ontario'",
            "outFields": "Proponent_Name,Technology,City,Latitude,Longitude,Quantity,Status_EN",
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": 1000,
        }
        data = requests.get(NRCAN_CHARGERS_API, params=params, timeout=120).json()
        features = data.get("features") or []
        if not features:
            break
        for f in features:
            a = f["attributes"]
            qty = max(int(a.get("Quantity") or 1), 1)
            power = _power_kw_from_technology(a.get("Technology", ""))
            for _ in range(qty):
                rows.append(
                    {
                        "name": a.get("Proponent_Name") or "Charger",
                        "latitude": a.get("Latitude"),
                        "longitude": a.get("Longitude"),
                        "power_kw": power,
                        "type": "DC Fast" if power >= 50 else "Level 2",
                        "operator": a.get("Proponent_Name") or "",
                        "city": a.get("City"),
                        "status": a.get("Status_EN"),
                        "data_source": "NRCan ZEVIP/EVAFIDI open data (Canada.ca)",
                    }
                )
        offset += len(features)
        if len(features) < 1000:
            break

    df = pd.DataFrame(rows).dropna(subset=["latitude", "longitude"])
    _save(df, cache)
    return df


def fetch_open_charge_map_gta(cache: Path | None = None, api_key: str | None = None) -> pd.DataFrame:
    """Optional: Open Charge Map if you set OCM_API_KEY environment variable."""
    import os

    cache = cache or RAW / "existing_chargers_ocm.csv"
    api_key = api_key or os.environ.get("OCM_API_KEY")
    if not api_key:
        return pd.DataFrame()

    params = {
        "output": "json",
        "countrycode": "CA",
        "latitude": 43.6532,
        "longitude": -79.3832,
        "distance": 80,
        "distanceunit": "KM",
        "maxresults": 2000,
        "compact": "true",
        "verbose": "false",
        "key": api_key,
    }
    r = requests.get(OCM_URL, params=params, timeout=90)
    r.raise_for_status()
    rows = []
    for p in r.json():
        addr = p.get("AddressInfo") or {}
        conns = p.get("Connections") or []
        powers = [c.get("PowerKW") or 0 for c in conns]
        power = max(powers) if powers else 7.2
        rows.append(
            {
                "name": addr.get("Title") or "Charger",
                "latitude": addr.get("Latitude"),
                "longitude": addr.get("Longitude"),
                "power_kw": power,
                "type": "DC Fast" if power >= 50 else "Level 2",
                "operator": (p.get("OperatorInfo") or {}).get("Title", ""),
                "data_source": "Open Charge Map API",
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["latitude", "longitude"])
    if not df.empty:
        _save(df, cache)
    return df


def fetch_all_chargers(cache: Path | None = None) -> pd.DataFrame:
    """Merge federal NRCan Ontario chargers (+ optional OCM GTA supplement)."""
    cache = cache or RAW / "existing_chargers.csv"
    if cache.exists():
        return pd.read_csv(cache)

    nrcan = fetch_nrcan_chargers_ontario()
    ocm = fetch_open_charge_map_gta()
    df = pd.concat([nrcan, ocm], ignore_index=True) if not ocm.empty else nrcan
    df = df.drop_duplicates(subset=["latitude", "longitude", "name"])
    _save(df, cache)
    # backward-compatible alias
    _save(df, RAW / "existing_chargers_ocm.csv")
    return df


def fetch_current_weather(cache: Path | None = None) -> dict:
    cache = cache or RAW / "weather_openmeteo.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    params = {
        "latitude": 43.6532,
        "longitude": -79.3832,
        "current": "temperature_2m",
        "timezone": "America/Toronto",
    }
    data = requests.get(OPEN_METEO, params=params, timeout=30).json()
    temp = data["current"]["temperature_2m"]
    if temp < 0:
        factor = 1.25
    elif temp < 10:
        factor = 1.15
    else:
        factor = 1.0
    payload = {
        "temperature_c": temp,
        "weather_factor": factor,
        "data_source": "Open-Meteo API",
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def merge_fsa_centroids(geonames: pd.DataFrame, toronto: pd.DataFrame) -> pd.DataFrame:
    """Prefer Toronto official coords when available, else GeoNames."""
    g = geonames.rename(columns={"area": "area_geonames"})
    t = toronto.rename(columns={"area": "area_toronto"})
    t = t.rename(
        columns={
            "latitude": "latitude_to",
            "longitude": "longitude_to",
            "coord_source": "coord_source_to",
        }
    )
    merged = g.merge(
        t[["fsa", "latitude_to", "longitude_to", "area_toronto", "coord_source_to"]],
        on="fsa",
        how="left",
    )
    merged["latitude"] = merged["latitude_to"].combine_first(merged["latitude"])
    merged["longitude"] = merged["longitude_to"].combine_first(merged["longitude"])
    merged["area"] = merged["area_toronto"].combine_first(merged["area_geonames"])
    merged["coord_source"] = merged["coord_source_to"].combine_first(merged["coord_source"])
    return merged[["fsa", "area", "province", "latitude", "longitude", "coord_source"]]


def count_chargers_and_avg_power(
    chargers: pd.DataFrame, areas: pd.DataFrame, km: float = 8.0
) -> pd.DataFrame:
    import math

    def haversine(lat1, lon1, lat2, lon2):
        r = 6371
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    counts, avg_powers = [], []
    for _, row in areas.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        nearby = []
        for _, c in chargers.iterrows():
            if haversine(lat, lon, c["latitude"], c["longitude"]) <= km:
                nearby.append(c["power_kw"] or 7.2)
        counts.append(len(nearby))
        avg_powers.append(sum(nearby) / len(nearby) if nearby else 7.2)
    areas = areas.copy()
    areas["charger_count"] = counts
    areas["avg_charger_power_kw"] = avg_powers
    areas["charger_source"] = (
        "NRCan ZEVIP/EVAFIDI open data (count within 8 km of FSA centroid)"
    )
    return areas


def filter_gta(df: pd.DataFrame, fsa_col: str = "fsa") -> pd.DataFrame:
    return df[df[fsa_col].str[0].isin(GTA_FSA_PREFIXES)].copy()


def download_ontario_ev_fsa(
    quarter_label: str = "Q4 2025",
    cache: Path | None = None,
) -> Path:
    """Download real Ontario EV-by-FSA CSV from data.ontario.ca / open.canada.ca."""
    cache = cache or RAW / "ontario_ev_by_fsa.csv"
    if cache.exists():
        return cache

    api = (
        "https://open.canada.ca/data/api/3/action/package_show"
        "?id=b5696d0c-4b7a-4c64-b546-cad4117a1774"
    )
    package = requests.get(api, timeout=30).json()["result"]
    url = None
    for resource in package["resources"]:
        if resource.get("format", "").upper() == "CSV" and quarter_label in resource.get("name", ""):
            url = resource["url"]
            break
    if not url:
        raise ValueError(f"No CSV resource found for {quarter_label}")

    print(f"Downloading Ontario EV data ({quarter_label})...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    return cache
