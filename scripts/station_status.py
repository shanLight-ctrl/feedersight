"""
ML models for FeederSight station intelligence:

1. OutageProbabilityModel  — Logistic Regression: P(outage in 24h) per FSA
2. RateLimitOptimizer      — Gradient Boosting: recommended kW cap per station
3. AnomalyDetector         — Isolation Forest: flag unusual station behaviour

All models are trained on synthetic-but-realistic scenarios derived from real data.
No real telemetry exists; these are planning/decision-support tools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── 1. Outage Probability Model ───────────────────────────────────────────────

def _make_outage_training_data(areas: pd.DataFrame, n_scenarios: int = 2000):
    """
    Synthesise training scenarios by combining real FSA features with
    random weather severity draws. Label = 1 if conditions imply outage risk.
    """
    rng = np.random.default_rng(42)
    rows = areas.sample(n=n_scenarios, replace=True, random_state=42).reset_index(drop=True)

    weather_sev = rng.uniform(0, 1, n_scenarios)
    event_boost = rng.choice([0.0, 0.10, 0.20, 0.25], n_scenarios, p=[0.5, 0.2, 0.2, 0.1])

    # Outage probability: driven by grid stress + weather severity
    raw_p = (
        0.35 * (rows["risk_ratio"].clip(0, 5) / 5)
        + 0.40 * weather_sev
        + 0.25 * event_boost
    )
    labels = (raw_p + rng.normal(0, 0.05, n_scenarios) > 0.55).astype(int)

    X = pd.DataFrame({
        "risk_ratio":     rows["risk_ratio"].values,
        "base_load_kw":   rows["base_load_kw"].values,
        "feeder_capacity_kw": rows["feeder_capacity_kw"].values,
        "ev_count":       rows["ev_count"].values,
        "charger_count":  rows["charger_count"].values,
        "weather_severity": weather_sev,
        "event_boost":    event_boost,
    })
    return X, labels


class OutageProbabilityModel:
    """Logistic Regression: P(outage) per FSA given current weather + grid state."""

    FEATURES = [
        "risk_ratio", "base_load_kw", "feeder_capacity_kw",
        "ev_count", "charger_count", "weather_severity", "event_boost",
    ]

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ])
        self._trained = False

    def fit(self, areas: pd.DataFrame) -> "OutageProbabilityModel":
        X, y = _make_outage_training_data(areas)
        self.model.fit(X, y)
        self._trained = True
        return self

    def predict_proba(self, areas: pd.DataFrame, weather: dict) -> pd.Series:
        if not self._trained:
            raise RuntimeError("Call fit() first")
        sev = weather.get("peak_severity_24h", 0)
        boost = {"ice_storm": 0.25, "high_wind": 0.20, "heavy_rain": 0.10,
                 "heat_wave": 0.15, "snowfall": 0.12}.get(weather.get("peak_event_24h", "normal"), 0.0)

        X = areas[["risk_ratio", "base_load_kw", "feeder_capacity_kw",
                    "ev_count", "charger_count"]].copy()
        X["weather_severity"] = sev
        X["event_boost"] = boost
        X = X.fillna(0)
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=areas.index).round(3)


# ── 2. Rate Limit Optimizer ───────────────────────────────────────────────────

def _make_rate_limit_training_data(areas: pd.DataFrame, n_scenarios: int = 3000):
    """
    Target: the highest kW rate that keeps total_load_kw <= feeder_capacity_kw.

    rate_limit = max kW such that:
      base_load + (ev_count * peak_share * rate_limit * weather_factor) <= feeder_capacity
    => rate_limit = (feeder_capacity - base_load) / (ev_count * peak_share * weather_factor)
    """
    rng = np.random.default_rng(7)
    rows = areas.sample(n=n_scenarios, replace=True, random_state=7).reset_index(drop=True)
    weather_factor = rng.uniform(0.9, 1.35, n_scenarios)

    denom = (
        rows["ev_count"].values
        * rows["peak_charging_share"].values
        * weather_factor
    )
    headroom = rows["feeder_capacity_kw"].values - rows["base_load_kw"].values
    safe_rate = np.where(denom > 0, headroom / denom, rows["avg_charger_power_kw"].values)
    safe_rate = np.clip(safe_rate, 3.3, 350.0)

    X = pd.DataFrame({
        "risk_ratio":         rows["risk_ratio"].values,
        "ev_count":           rows["ev_count"].values,
        "base_load_kw":       rows["base_load_kw"].values,
        "feeder_capacity_kw": rows["feeder_capacity_kw"].values,
        "peak_charging_share":rows["peak_charging_share"].values,
        "avg_charger_power_kw":rows["avg_charger_power_kw"].values,
        "charger_count":      rows["charger_count"].values,
        "weather_factor":     weather_factor,
    })
    return X, safe_rate


class RateLimitOptimizer:
    """
    Gradient Boosting: recommends per-FSA kW cap to keep grid within safe limits.
    Output: recommended_rate_kw — clip chargers to this value during stress periods.
    """

    FEATURES = [
        "risk_ratio", "ev_count", "base_load_kw", "feeder_capacity_kw",
        "peak_charging_share", "avg_charger_power_kw", "charger_count", "weather_factor",
    ]

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42
            )),
        ])
        self._trained = False

    def fit(self, areas: pd.DataFrame) -> "RateLimitOptimizer":
        X, y = _make_rate_limit_training_data(areas)
        self.model.fit(X, y)
        self._trained = True
        return self

    def recommend(self, areas: pd.DataFrame, weather: dict) -> pd.Series:
        if not self._trained:
            raise RuntimeError("Call fit() first")
        wf = weather.get("current_severity", 0) * 0.35 + 1.0   # 1.0–1.35 range
        X = areas[["risk_ratio", "ev_count", "base_load_kw", "feeder_capacity_kw",
                    "peak_charging_share", "avg_charger_power_kw", "charger_count"]].copy()
        X["weather_factor"] = wf
        X = X.fillna(0)
        limits = self.model.predict(X)
        return pd.Series(np.clip(limits, 3.3, 350.0).round(1), index=areas.index)


# ── 3. Anomaly Detector ───────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Isolation Forest: flags stations with unusual feature combinations.
    Anomalies = statistical outliers in machine_count × power_kw × report_count space.
    """

    FEATURES = ["machine_count", "avg_power_kw"]

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            n_estimators=100, contamination=contamination, random_state=42
        )
        self._trained = False

    def fit(self, stations: pd.DataFrame) -> "AnomalyDetector":
        X = stations[self.FEATURES].fillna(0)
        self.model.fit(X)
        self._trained = True
        return self

    def flag(self, stations: pd.DataFrame, reports: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self._trained:
            raise RuntimeError("Call fit() first")

        df = stations.copy()

        # Add report count per station if reports exist
        if reports is not None and not reports.empty and "station_name" in reports.columns:
            report_counts = reports.groupby("station_name").size().rename("report_count")
            df = df.merge(report_counts, left_on="name", right_index=True, how="left")
            df["report_count"] = df["report_count"].fillna(0)
        else:
            df["report_count"] = 0

        X = df[self.FEATURES].fillna(0)
        scores = self.model.decision_function(X)   # lower = more anomalous
        preds = self.model.predict(X)              # -1 = anomaly, 1 = normal

        df["anomaly_score"] = scores.round(4)
        df["is_anomaly"] = preds == -1
        df["anomaly_reason"] = df.apply(_anomaly_reason, axis=1)
        return df


def _anomaly_reason(row: pd.Series) -> str:
    if not row.get("is_anomaly"):
        return ""
    reasons = []
    if row.get("machine_count", 0) > 50:
        reasons.append("unusually large station")
    if row.get("avg_power_kw", 0) > 100:
        reasons.append("very high power rating")
    if row.get("report_count", 0) >= 3:
        reasons.append(f"{int(row['report_count'])} user reports")
    return "; ".join(reasons) if reasons else "statistical outlier"


# ── Convenience: train all three at once ─────────────────────────────────────

def train_all(areas: pd.DataFrame, stations: pd.DataFrame):
    outage_model = OutageProbabilityModel().fit(areas)
    rate_model = RateLimitOptimizer().fit(areas)
    anomaly_model = AnomalyDetector().fit(stations)
    return outage_model, rate_model, anomaly_model
