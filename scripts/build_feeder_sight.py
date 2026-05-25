"""
Build ArcGIS layers using REAL data only (no placeholders).

Prerequisites:
  python scripts/fetch_real_data.py
  Put Ontario EV registrations CSV in data/raw/

Usage:
  python scripts/build_feeder_sight.py --ev-csv data/raw/ontario_ev_fsa.csv
  python scripts/build_feeder_sight.py --ev-csv data/raw/ontario_ev_fsa.csv --all-ontario
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from real_data_sources import (
    RAW,
    count_chargers_and_avg_power,
    fetch_current_weather,
    filter_gta,
)

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "data" / "processed"

FSA_CANDIDATES = [
    "fsa",
    "FSA",
    "Forward Sortation Area",
    "forward_sortation_area",
    "Postal Code",
    "postal_code",
    "CFSA",
]
EV_COUNT_CANDIDATES = [
    "Total EV",
    "ev_count",
    "EV Count",
    "ev_registrations",
    "total_ev",
    "Total EVs",
    "count",
    "Count",
    "number_of_evs",
    "Registrations",
]


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


def load_ev_by_fsa(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    fsa_col = pick_column(df, FSA_CANDIDATES)

    # Some files split BEV + PHEV instead of one total column
    if any(c in df.columns for c in EV_COUNT_CANDIDATES):
        ev_col = pick_column(df, EV_COUNT_CANDIDATES)
        out = df[[fsa_col, ev_col]].copy()
        out.columns = ["fsa", "ev_count"]
    else:
        bev = next((c for c in df.columns if c.upper() in ("BEV", "BATTERY EV")), None)
        phev = next((c for c in df.columns if "PHEV" in c.upper()), None)
        if bev and phev:
            out = df[[fsa_col, bev, phev]].copy()
            out["ev_count"] = pd.to_numeric(out[bev], errors="coerce").fillna(0) + pd.to_numeric(
                out[phev], errors="coerce"
            ).fillna(0)
            out = out[[fsa_col, "ev_count"]]
            out.columns = ["fsa", "ev_count"]
        else:
            raise ValueError(f"No EV count column found. Columns: {list(df.columns)}")

    out["fsa"] = out["fsa"].astype(str).str.strip().str.upper().str[:3]
    out = out[out["fsa"].str.len() == 3]
    out["ev_count"] = pd.to_numeric(out["ev_count"], errors="coerce")
    out = out.dropna(subset=["ev_count"])
    out = out.groupby("fsa", as_index=False)["ev_count"].sum()
    out["ev_count_source"] = "Ontario EV registrations by FSA (user CSV)"
    return out


def require_cache() -> None:
    needed = [
        RAW / "fsa_centroids_merged.csv",
        RAW / "ieso_fsa_peak.csv",
        RAW / "existing_chargers.csv",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing real data cache. Run first:\n"
            "  python scripts/fetch_real_data.py\n"
            f"Missing: {[str(p) for p in missing]}"
        )


def compute_ev_load(areas: pd.DataFrame, weather: dict) -> pd.DataFrame:
    df = areas.copy()
    df["temperature_c"] = weather["temperature_c"]
    df["weather_factor"] = weather["weather_factor"]
    df["weather_source"] = weather["data_source"]

    # Peak charging share from real IESO hourly shape (not a fixed 20% guess)
    df["peak_charging_share"] = df["ieso_peak_fraction"]
    df["peak_charging_source"] = "IESO peak-hour / daily-average ratio by FSA"

    df["ev_load_kw"] = (
        df["ev_count"]
        * df["peak_charging_share"]
        * df["avg_charger_power_kw"]
        * df["weather_factor"]
    )
    df["ev_load_source"] = (
        "Ontario EV count × IESO peak fraction × NRCan avg charger kW × Open-Meteo weather factor"
    )

    df["total_load_kw"] = df["base_load_kw"] + df["ev_load_kw"]
    df["risk_ratio"] = df["total_load_kw"] / df["feeder_capacity_kw"]
    df["risk_level"] = pd.cut(
        df["risk_ratio"],
        bins=[-float("inf"), 0.8, 1.0, float("inf")],
        labels=["Safe", "Warning", "Overload"],
    ).astype(str)
    return df


def priority_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ev_norm = df["ev_count"] / df["ev_count"].max()
    gap = 1 - (df["charger_count"] / max(df["charger_count"].max(), 1))
    df["priority_score"] = 0.4 * ev_norm + 0.3 * df["risk_ratio"] + 0.3 * gap
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ev-csv",
        type=Path,
        default=RAW / "ontario_ev_by_fsa.csv",
        help="Ontario EV-by-FSA CSV (default: auto-downloaded Q4 2025)",
    )
    parser.add_argument("--all-ontario", action="store_true", help="Keep all FSAs (default: GTA L/M/N)")
    parser.add_argument("--refresh", action="store_true", help="Re-download public APIs before build")
    args = parser.parse_args()

    if args.refresh:
        import fetch_real_data

        fetch_real_data.main()
    else:
        cache_missing = not (RAW / "fsa_centroids_merged.csv").exists()
        if cache_missing:
            print("No cache found — fetching real public data...")
            import fetch_real_data

            fetch_real_data.main()

    require_cache()

    OUT.mkdir(parents=True, exist_ok=True)

    if not args.ev_csv.exists():
        from real_data_sources import download_ontario_ev_fsa

        print("Downloading Ontario EV-by-FSA (Q4 2025)...")
        download_ontario_ev_fsa()

    ev = load_ev_by_fsa(args.ev_csv)
    centroids = pd.read_csv(RAW / "fsa_centroids_merged.csv")
    ieso = pd.read_csv(RAW / "ieso_fsa_peak.csv")
    chargers = pd.read_csv(RAW / "existing_chargers.csv")
    weather = fetch_current_weather()

    areas = ev.merge(centroids, on="fsa", how="inner")
    areas = areas.merge(
        ieso[
            [
                "fsa",
                "base_load_kw",
                "feeder_capacity_kw",
                "ieso_peak_hour_kwh",
                "ieso_peak_hour",
                "premise_count",
                "ieso_peak_fraction",
                "base_load_source",
                "feeder_capacity_source",
            ]
        ],
        on="fsa",
        how="inner",
    )

    if areas.empty:
        raise SystemExit(
            "No rows after join. Your EV FSAs may not overlap IESO/centroids. "
            "Try --all-ontario or check FSA codes in your CSV."
        )

    if not args.all_ontario:
        areas = filter_gta(areas)

    areas = count_chargers_and_avg_power(chargers, areas)
    areas = compute_ev_load(areas, weather)
    areas = priority_score(areas)

    # ArcGIS-ready main layer (every column traceable)
    areas.to_csv(OUT / "feeder_sight_areas.csv", index=False)
    chargers.to_csv(OUT / "existing_chargers.csv", index=False)

    ranked = areas.sort_values("priority_score", ascending=False).head(10)
    ranked[
        [
            "area",
            "fsa",
            "latitude",
            "longitude",
            "ev_count",
            "charger_count",
            "risk_level",
            "priority_score",
            "ev_count_source",
        ]
    ].to_csv(OUT / "recommended_sites.csv", index=False)

    manifest = {
        "layers_for_arcgis": {
            "feeder_sight_areas.csv": {
                "xy": ["longitude", "latitude"],
                "style": "risk_level",
                "popup_fields": [
                    "fsa",
                    "area",
                    "ev_count",
                    "charger_count",
                    "base_load_kw",
                    "ev_load_kw",
                    "feeder_capacity_kw",
                    "risk_ratio",
                    "risk_level",
                    "ev_count_source",
                    "base_load_source",
                    "feeder_capacity_source",
                ],
            },
            "existing_chargers.csv": {
                "xy": ["longitude", "latitude"],
                "style": "type",
            },
            "recommended_sites.csv": {"xy": ["longitude", "latitude"]},
        },
        "rows": len(areas),
        "all_real_except": (
            "feeder_capacity_kw is a planning proxy from IESO peak + 25% margin, "
            "not utility nameplate capacity"
        ),
    }
    (OUT / "arcgis_layer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Built {len(areas)} areas (real data pipeline)")
    print(f"  {OUT / 'feeder_sight_areas.csv'}")
    print(f"  {OUT / 'existing_chargers.csv'} ({len(chargers)} chargers)")
    print(f"  {OUT / 'recommended_sites.csv'}")


if __name__ == "__main__":
    main()
