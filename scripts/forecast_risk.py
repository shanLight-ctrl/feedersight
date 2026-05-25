"""
Forecast EV grid overload risk using growth projections and scikit-learn.

Reads the already-built feeder_sight_areas.csv and produces:
  data/processed/risk_forecast.csv    — FSA-level forecast + urgency field for ArcGIS
  data/processed/model_summary.json   — model CV scores and feature importances

Usage:
  python scripts/forecast_risk.py
  python scripts/forecast_risk.py --max-years 15
  python scripts/forecast_risk.py --conservative 0.15 --moderate 0.30 --aggressive 0.45
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

FEATURE_COLS = [
    "ev_count",
    "base_load_kw",
    "feeder_capacity_kw",
    "charger_count",
    "avg_charger_power_kw",
    "peak_charging_share",
    "premise_count",
    # risk_ratio is excluded: it is a near-deterministic function of years_to_overload
    # and would dominate feature importances, masking which raw inputs matter most.
]


def project_risk_ratio(areas: pd.DataFrame, annual_growth: float, year: int) -> pd.Series:
    ev_projected = areas["ev_count"] * (1 + annual_growth) ** year
    ev_load = (
        ev_projected
        * areas["peak_charging_share"]
        * areas["avg_charger_power_kw"]
        * areas["weather_factor"]
    )
    total_load = areas["base_load_kw"] + ev_load
    return total_load / areas["feeder_capacity_kw"]


def years_until_overload(areas: pd.DataFrame, annual_growth: float, max_years: int) -> pd.Series:
    """Return the first year each FSA crosses risk_ratio > 1.0 under a given CAGR."""
    result = pd.Series(np.nan, index=areas.index)
    for y in range(1, max_years + 1):
        risk = project_risk_ratio(areas, annual_growth, y)
        newly_overloaded = result.isna() & (risk > 1.0)
        result[newly_overloaded] = y
    result[areas["risk_ratio"] > 1.0] = 0  # already overloaded
    return result


def build_risk_timeline(areas: pd.DataFrame, scenarios: dict[str, float], max_years: int) -> pd.DataFrame:
    """Build a per-FSA risk table with one column per (scenario, horizon) combination."""
    out = areas[["fsa", "area", "risk_ratio", "risk_level", "ev_count"]].copy()

    for label, cagr in scenarios.items():
        out[f"years_to_overload_{label}"] = years_until_overload(areas, cagr, max_years)
        for year in [1, 3, 5, 10]:
            if year <= max_years:
                out[f"risk_{label}_{year}yr"] = project_risk_ratio(areas, cagr, year).round(3)

    # Urgency label for ArcGIS symbology — uses moderate scenario
    moderate_years = out["years_to_overload_moderate"]

    def urgency(y: float) -> str:
        if y == 0:
            return "Overloaded Now"
        if pd.isna(y):
            return "Safe 10+ Years"
        if y <= 2:
            return "1-2 Years"
        if y <= 5:
            return "3-5 Years"
        return "6-10 Years"

    out["forecast_urgency"] = moderate_years.apply(urgency)
    return out


def train_overload_model(
    areas: pd.DataFrame, target: pd.Series, cv_folds: int = 5
) -> tuple[Pipeline, float, dict[str, float]]:
    """
    Fit a GradientBoostingRegressor to predict years_to_overload from current-state features.
    Returns (fitted pipeline, mean CV R², feature importances dict).
    """
    X = areas[FEATURE_COLS].fillna(0)
    # Replace NaN (never overloads within horizon) with max_years + 1 for regression
    sentinel = target.max() + 1 if target.notna().any() else 11
    y = target.fillna(sentinel)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gb",
                GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
                ),
            ),
        ]
    )
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")
    model.fit(X, y)

    importances = dict(
        zip(FEATURE_COLS, model.named_steps["gb"].feature_importances_.tolist())
    )
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    return model, float(cv_scores.mean()), importances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROCESSED / "feeder_sight_areas.csv")
    parser.add_argument("--max-years", type=int, default=10)
    parser.add_argument("--conservative", type=float, default=0.20, help="Annual EV growth rate")
    parser.add_argument("--moderate", type=float, default=0.35)
    parser.add_argument("--aggressive", type=float, default=0.50)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Input not found: {args.input}\n"
            "Run first: python scripts/build_feeder_sight.py"
        )

    areas = pd.read_csv(args.input)

    scenarios: dict[str, float] = {
        "conservative": args.conservative,
        "moderate": args.moderate,
        "aggressive": args.aggressive,
    }

    print(f"Forecasting risk for {len(areas)} FSAs over {args.max_years} years ...")
    print(f"  Scenarios: {', '.join(f'{k} ({v:.0%} CAGR)' for k, v in scenarios.items())}")

    forecast = build_risk_timeline(areas, scenarios, args.max_years)
    # Carry lat/lon forward so ArcGIS can geocode this layer directly
    forecast = forecast.merge(areas[["fsa", "latitude", "longitude"]], on="fsa", how="left")

    # Train model on moderate scenario target
    moderate_target = forecast["years_to_overload_moderate"]
    model, cv_r2, importances = train_overload_model(areas, moderate_target)

    forecast["predicted_years_to_overload"] = model.predict(areas[FEATURE_COLS].fillna(0)).round(1)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(PROCESSED / "risk_forecast.csv", index=False)
    print(f"  Wrote {PROCESSED / 'risk_forecast.csv'}")

    # Summary stats
    urgency_counts = forecast["forecast_urgency"].value_counts().to_dict()

    model_summary = {
        "model": "GradientBoostingRegressor",
        "target": "years_to_overload (moderate 35% CAGR scenario)",
        "cv_r2_mean": round(cv_r2, 3),
        "cv_folds": 5,
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "scenarios": scenarios,
        "max_forecast_years": args.max_years,
        "forecast_urgency_counts": urgency_counts,
        "fsa_count": len(areas),
    }
    (PROCESSED / "model_summary.json").write_text(
        json.dumps(model_summary, indent=2), encoding="utf-8"
    )
    print(f"  Wrote {PROCESSED / 'model_summary.json'}")

    print("\nForecast summary (moderate 35% CAGR):")
    order = ["Overloaded Now", "1-2 Years", "3-5 Years", "6-10 Years", "Safe 10+ Years"]
    for label in order:
        count = urgency_counts.get(label, 0)
        print(f"  {label:<20} {count:>4} FSAs")

    print(f"\nModel CV R²: {cv_r2:.3f}")
    print("Top risk drivers (feature importance):")
    for feat, imp in list(importances.items())[:5]:
        print(f"  {feat:<30} {imp:.4f}")

    _update_manifest(scenarios, args.max_years)


def _update_manifest(scenarios: dict[str, float], max_years: int) -> None:
    manifest_path = PROCESSED / "arcgis_layer_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["layers_for_arcgis"]["risk_forecast.csv"] = {
        "xy": ["longitude", "latitude"],
        "style": "forecast_urgency",
        "popup_fields": [
            "fsa",
            "area",
            "ev_count",
            "risk_level",
            "forecast_urgency",
            "years_to_overload_conservative",
            "years_to_overload_moderate",
            "years_to_overload_aggressive",
            "risk_moderate_3yr",
            "risk_moderate_5yr",
            "predicted_years_to_overload",
        ],
        "note": (
            f"years_to_overload = first year risk_ratio exceeds 1.0 under that CAGR scenario. "
            f"Horizon: {max_years} years. "
            f"Scenarios: {', '.join(f'{k}={v:.0%}' for k, v in scenarios.items())}."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Updated {manifest_path.name}")


if __name__ == "__main__":
    main()
