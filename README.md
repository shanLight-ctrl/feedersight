# FeederSight — 100% real public data for ArcGIS

Every layer field is backed by a real source. Nothing is a random placeholder.

## What is real on your map

| Field | Real source |
|-------|-------------|
| `ev_count` | **Your** Ontario EV registrations CSV |
| `latitude`, `longitude` | GeoNames + City of Toronto Open Data |
| `base_load_kw` | IESO hourly consumption by FSA (smart-meter aggregates) |
| `peak_charging_share` | Derived from IESO peak-hour / daily-average ratio |
| `charger_count`, `avg_charger_power_kw` | NRCan ZEVIP/EVAFIDI (Canada open data) |
| `weather_factor` | Open-Meteo current temperature |
| `feeder_capacity_kw` | **Proxy** from IESO observed peak + 25% margin (not utility nameplate) |

**Only honest assumption:** feeder nameplate capacity is not public — we use IESO measured peak as a planning envelope. Say that to judges.

## One-time setup

```powershell
cd "d:\Personal Project\ev adoption"
pip install -r requirements.txt
python scripts/fetch_real_data.py --gta-only
```

Downloads and caches (~15 MB IESO + chargers + coordinates).

## Build map layers (all real — one command)

```powershell
python scripts/fetch_real_data.py --gta-only
python scripts/build_feeder_sight.py
```

This auto-downloads Ontario EV Q4 2025, IESO load, NRCan chargers, coordinates, and weather.

Optional: use your own EV CSV with `--ev-csv data/raw/your_file.csv`

Outputs for ArcGIS:

- `data/processed/feeder_sight_areas.csv` — main risk layer + `*_source` columns for popups
- `data/processed/existing_chargers.csv` — real charger points
- `data/processed/recommended_sites.csv` — top priority FSAs
- `data/processed/arcgis_layer_manifest.json` — which fields to show

Use `--all-ontario` if you want every FSA in your EV file, not just GTA (L/M/N).

## ArcGIS — display everything

Upload **3 CSV layers**. In popups, show the `*_source` fields so judges see provenance.

| Layer | X | Y | Style |
|-------|---|---|-------|
| feeder_sight_areas | longitude | latitude | risk_level |
| existing_chargers | longitude | latitude | type |
| recommended_sites | longitude | latitude | priority_score |

Add **ArcGIS Living Atlas** in the same map for demographics (income, housing) — also real.

## Judge script

> FeederSight uses real Ontario EV registrations, real IESO hourly load by postal area, real public charger locations, and real weather. Feeder capacity is estimated from IESO observed peak demand because utilities do not publish feeder limits; Alectra would replace that with actual feeder data in production.

## Refresh data before demo

```powershell
python scripts/build_feeder_sight.py --ev-csv data/raw/ontario_ev_fsa.csv --refresh
```
