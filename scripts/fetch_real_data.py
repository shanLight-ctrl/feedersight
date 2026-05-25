"""
Download ALL real public datasets and cache under data/raw/.

Run once (or when refreshing data):
  python scripts/fetch_real_data.py
  python scripts/fetch_real_data.py --gta-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from real_data_sources import (
    RAW,
    aggregate_ieso_by_fsa,
    download_ontario_ev_fsa,
    fetch_current_weather,
    fetch_geonames_fsa_centroids,
    fetch_ieso_hourly_fsa,
    fetch_all_chargers,
    fetch_toronto_fsa_centroids,
    filter_gta,
    merge_fsa_centroids,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gta-only", action="store_true", help="Also write GTA-filtered copies")
    args = parser.parse_args()

    print("1/5 GeoNames FSA centroids (Canada)...")
    geonames = fetch_geonames_fsa_centroids()

    print("2/5 Toronto Open Data FSA centroids...")
    toronto = fetch_toronto_fsa_centroids()
    centroids = merge_fsa_centroids(geonames, toronto)
    centroids.to_csv(RAW / "fsa_centroids_merged.csv", index=False)

    print("3/5 IESO hourly consumption by FSA...")
    hourly = fetch_ieso_hourly_fsa()
    ieso = aggregate_ieso_by_fsa(hourly)
    ieso.to_csv(RAW / "ieso_fsa_peak.csv", index=False)

    print("4/5 NRCan federal charger registry (Ontario)...")
    fetch_all_chargers()

    print("5/6 Open-Meteo weather...")
    weather = fetch_current_weather()

    print("6/6 Ontario EV registrations by FSA (Q4 2025)...")
    download_ontario_ev_fsa()

    if args.gta_only:
        filter_gta(centroids).to_csv(RAW / "fsa_centroids_gta.csv", index=False)
        filter_gta(ieso).to_csv(RAW / "ieso_fsa_peak_gta.csv", index=False)

    manifest = {
        "datasets": {
            "fsa_centroids_merged.csv": "GeoNames + City of Toronto Open Data",
            "ieso_fsa_peak.csv": "IESO Hourly Consumption by FSA",
            "existing_chargers.csv": "NRCan ZEVIP/EVAFIDI (+ optional Open Charge Map)",
            "weather_openmeteo.json": "Open-Meteo API",
            "ontario_ev_by_fsa.csv": "Ontario EV registrations Q4 2025 (data.ontario.ca)",
        },
        "weather": weather,
    }
    (RAW / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done. Cached files in data/raw/")
    print("Next: python scripts/build_feeder_sight.py")


if __name__ == "__main__":
    main()
