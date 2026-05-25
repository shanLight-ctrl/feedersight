"""FeederSight Flask web server — start with: python app.py"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from ai_assistant import ask as claude_ask
from station_status import train_all
from weather_alerts import analyse_forecast, fetch_forecast, station_outage_probability
from winter_history import build_winter_report

PROCESSED = ROOT / "data" / "processed"
REPORTS_FILE = ROOT / "data" / "reports.json"
USERS_FILE = ROOT / "data" / "users.json"

app = Flask(__name__, template_folder="templates")
app.secret_key = "feedersight-secret-2026"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _keyword_answer(question: str, error: str = "") -> str:
    q = question.lower()

    # Grid / overload questions
    if any(w in q for w in ["overload", "worst", "critical", "most at risk", "dangerous", "bad", "problem area"]):
        top = areas[areas["risk_level"] == "Overload"].sort_values("risk_ratio", ascending=False).head(3)
        names = ", ".join(f"{r.get('area', r['fsa'])}" for _, r in top.iterrows())
        return f"The most stressed areas right now are {names} — their grids are already carrying more EV load than they were designed for."

    # Near-term risk
    if any(w in q for w in ["soon", "1 year", "2 year", "upcoming", "next year", "about to"]):
        soon = forecast_df[forecast_df["forecast_urgency"] == "1-2 Years"]
        names = ", ".join(soon.head(3)["area"].fillna(soon.head(3)["fsa"]).tolist())
        return f"{len(soon)} areas will hit their grid limit within 1–2 years at current EV growth rates — including {names}."

    # Weather
    if any(w in q for w in ["weather", "storm", "temperature", "cold", "hot", "wind", "rain", "snow", "tonight", "today"]):
        t = weather["current_temp_c"]
        evt = (weather["current_event"] or "normal").replace("_", " ")
        alert = weather.get("alert_message")
        base = f"Right now in Toronto it's {t}°C with {evt} conditions."
        return (base + f" ⚠️ {alert}") if alert else (base + " No weather alerts active.")

    # Station status
    if any(w in q for w in ["offline", "broken", "down", "not working", "closed", "available", "open"]):
        offline = stations_enhanced[stations_enhanced["weather_status"].isin(["Offline", "High Risk"])]
        operational = stations_enhanced[stations_enhanced["weather_status"] == "Operational"]
        return f"{len(operational):,} charging stations are currently operational. {len(offline)} are offline or high risk due to weather conditions."

    # Charger recommendations / where to build
    if any(w in q for w in ["recommend", "build", "where", "new charger", "invest", "priority", "should go"]):
        top = recommended.head(3)
        names = ", ".join(f"{r.get('area', r['fsa'])}" for _, r in top.iterrows())
        return f"The top 3 areas that need new chargers most urgently are {names} — high EV demand but not enough charging infrastructure."

    # How many / counts
    if any(w in q for w in ["how many", "total", "count", "number of"]):
        total_machines = int(stations_enhanced["machine_count"].sum())
        return f"FeederSight is monitoring {len(areas)} areas across the GTA, {len(stations_enhanced):,} charging stations, and {total_machines:,} individual charging machines."

    # Safe to charge / can I charge
    if any(w in q for w in ["safe", "can i charge", "charge here", "charge tonight", "is it ok", "good time"]):
        alert = weather.get("alert_message")
        overloaded = len(areas[areas["risk_level"] == "Overload"])
        if alert:
            return f"Use caution tonight — {alert}. {overloaded} areas are currently overloaded. Turn on the Charging Stations layer and press 'Use my location' to find a safe nearby charger."
        return f"Conditions look fine for charging right now — no weather alerts and the grid is stable in most areas. Turn on the Charging Stations layer to find one near you."

    # Rate limit / speed / slow
    if any(w in q for w in ["rate", "limit", "slow", "speed", "fast", "kw", "power"]):
        limited = areas[areas["recommended_rate_kw"] < areas["avg_charger_power_kw"] * 0.9]
        return f"{len(limited)} areas currently have a recommended rate limit — chargers there may run slower than normal to protect the grid. Full speed is available everywhere else."

    # ML / models / AI / predictions
    if any(w in q for w in ["ml", "model", "ai", "predict", "machine learning", "algorithm"]):
        return "FeederSight uses 3 ML models: Logistic Regression predicts outage probability per area, Gradient Boosting optimises charging rate limits, and Isolation Forest flags unusual stations. See the ML tab for details."

    # Markham / specific area questions
    for _, row in areas.iterrows():
        area_name = str(row.get("area", "")).lower()
        if area_name and len(area_name) > 2 and area_name in q:
            risk = row.get("risk_level", "Unknown")
            urgency = row.get("forecast_urgency", "Unknown")
            prob = int((row.get("outage_probability", 0) or 0) * 100)
            return (f"{row.get('area', row['fsa'])} is currently at {risk} level. "
                    f"Forecast: {urgency}. Tonight's outage risk: {prob}%. "
                    f"Rate limit: {row.get('rate_limit_label', 'Full power')}.")

    # Default — actually helpful, not a list of commands
    overloaded = len(areas[areas["risk_level"] == "Overload"])
    t = weather["current_temp_c"]
    return (f"Right now: {overloaded} areas in the GTA have overloaded grids, "
            f"weather is {t}°C with no major alerts. "
            f"Try asking about a specific area, or press 'Use my location' in the Near Me tab to find chargers close to you.")


def _load_reports() -> pd.DataFrame:
    data = _load_json(REPORTS_FILE, [])
    return pd.DataFrame(data) if data else pd.DataFrame()


def _save_report(report: dict) -> None:
    existing = _load_json(REPORTS_FILE, [])
    existing.append(report)
    _save_json(REPORTS_FILE, existing)


# ── Startup: load data + train models ────────────────────────────────────────

print("Loading data...")
areas = pd.read_csv(PROCESSED / "feeder_sight_areas.csv")
forecast_df = pd.read_csv(PROCESSED / "risk_forecast.csv")
stations = pd.read_csv(PROCESSED / "stations.csv")
machines_df = pd.read_csv(PROCESSED / "machines.csv")
recommended = pd.read_csv(PROCESSED / "recommended_sites.csv")

print("Fetching weather...")
raw_weather = fetch_forecast(force=True)
weather = analyse_forecast(raw_weather)

print("Training ML models...")
outage_model, rate_model, anomaly_model = train_all(areas, stations)

areas["outage_probability"] = outage_model.predict_proba(areas, weather)
areas["recommended_rate_kw"] = rate_model.recommend(areas, weather)


def rate_label(row):
    cap = row["avg_charger_power_kw"]
    limit = row["recommended_rate_kw"]
    if limit >= cap * 0.9:
        return f"Full power ({cap:.0f} kW)"
    pct = int(limit / cap * 100) if cap > 0 else 100
    return f"Limited to {limit:.0f} kW ({pct}% of normal)"


areas["rate_limit_label"] = areas.apply(rate_label, axis=1)

stations_enhanced = station_outage_probability(stations, weather)
stations_enhanced = anomaly_model.flag(stations_enhanced, _load_reports())

areas = areas.merge(
    forecast_df[["fsa", "forecast_urgency", "years_to_overload_moderate",
                 "risk_moderate_3yr", "risk_moderate_5yr"]],
    on="fsa", how="left",
)

print("Building winter history...")
winter_report = build_winter_report(areas)

print("Ready — http://localhost:5000")


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    if not username or len(password) < 6:
        return jsonify({"error": "Username required and password must be 6+ characters"}), 400
    users = _load_json(USERS_FILE, {})
    if username in users:
        return jsonify({"error": "Username already taken"}), 409
    users[username] = {
        "password_hash": _hash(password),
        "created": datetime.utcnow().isoformat(),
        "car_model": None,
        "battery_pct": 80,
        "charge_target": 80,
    }
    _save_json(USERS_FILE, users)
    session["username"] = username
    return jsonify({"status": "ok", "username": username}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    users = _load_json(USERS_FILE, {})
    user = users.get(username)
    if not user or user["password_hash"] != _hash(password):
        return jsonify({"error": "Invalid username or password"}), 401
    session["username"] = username
    profile = {k: v for k, v in user.items() if k != "password_hash"}
    return jsonify({"status": "ok", "username": username, "profile": profile})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "ok"})


@app.route("/api/profile", methods=["GET", "POST"])
def profile():
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    users = _load_json(USERS_FILE, {})
    if request.method == "POST":
        data = request.get_json(force=True)
        for field in ["car_model", "battery_pct", "charge_target"]:
            if field in data:
                users[username][field] = data[field]
        _save_json(USERS_FILE, users)
    profile = {k: v for k, v in users[username].items() if k != "password_hash"}
    return jsonify(profile)


# ── Station + machine routes ──────────────────────────────────────────────────

@app.route("/api/stations/nearby")
def stations_nearby():
    try:
        lat = float(request.args.get("lat", 43.6532))
        lon = float(request.args.get("lon", -79.3832))
        radius = float(request.args.get("radius", 10))
    except ValueError:
        return jsonify({"error": "Invalid coordinates"}), 400

    import math

    def haversine(lat1, lon1, lat2, lon2):
        r = 6371
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    result = []
    for _, s in stations_enhanced.iterrows():
        if not s.get("latitude") or not s.get("longitude"):
            continue
        dist = haversine(lat, lon, s["latitude"], s["longitude"])
        if dist <= radius:
            row = s.fillna("").to_dict()
            row["distance_km"] = round(dist, 2)
            result.append(row)

    result.sort(key=lambda x: x["distance_km"])
    return jsonify(result)


@app.route("/api/stations/<station_id>/machines")
def station_machines(station_id: str):
    mach = machines_df[machines_df["station_id"] == station_id].copy()
    if mach.empty:
        return jsonify([])
    reports = _load_reports()
    station_name = mach.iloc[0]["name"] if not mach.empty else ""
    report_count = 0
    if not reports.empty and "station_name" in reports.columns:
        report_count = int((reports["station_name"] == station_name).sum())
    rows = mach.fillna("").to_dict(orient="records")
    for r in rows:
        r["report_count"] = report_count
    return jsonify(rows)


# ── AI assistant ──────────────────────────────────────────────────────────────

@app.route("/api/ask", methods=["POST"])
def ask_question():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "Please type a question."})

    user_lat = data.get("lat")
    user_lon = data.get("lon")
    radius = float(data.get("radius", 10))

    # Build user profile from session if logged in
    user_profile = None
    username = session.get("username")
    if username:
        users = _load_json(USERS_FILE, {})
        if username in users:
            user_profile = {k: v for k, v in users[username].items() if k != "password_hash"}
            # Add reachable range if battery known
            if user_profile.get("car_model") and user_profile.get("battery_pct"):
                from car_models import get_model, reachable_range_km
                car = get_model(user_profile["car_model"])
                if car:
                    user_profile["reachable_km"] = reachable_range_km(car, float(user_profile["battery_pct"]))

    limited_mode = False
    try:
        answer = claude_ask(
            question=question,
            weather=weather,
            areas_df=areas,
            stations_df=stations_enhanced,
            user_profile=user_profile,
            user_lat=user_lat,
            user_lon=user_lon,
            radius_km=radius,
        )
    except Exception as e:
        answer = _keyword_answer(question, str(e))
        limited_mode = "credit" in str(e).lower() or "balance" in str(e).lower()

    # Find nearest operational station for navigation button
    nav_target = None
    if user_lat and user_lon:
        import math
        def _hav(la1, lo1, la2, lo2):
            r = 6371
            p1, p2 = math.radians(la1), math.radians(la2)
            a = math.sin(math.radians(la2-la1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lo2-lo1)/2)**2
            return 2*r*math.asin(math.sqrt(a))
        operational = stations_enhanced[stations_enhanced["weather_status"] == "Operational"]
        best, best_d = None, float("inf")
        for _, s in operational.iterrows():
            if not s.get("latitude") or not s.get("longitude"):
                continue
            d = _hav(user_lat, user_lon, s["latitude"], s["longitude"])
            if d < best_d and d <= radius:
                best_d, best = d, s
        if best is not None:
            nav_target = {
                "name": best.get("name", "Charging Station"),
                "lat": best["latitude"],
                "lon": best["longitude"],
                "distance_km": round(best_d, 1),
                "type": best.get("type", ""),
                "machines": int(best.get("machine_count", 1)),
            }

    return jsonify({"answer": answer, "navigation_target": nav_target, "limited_mode": limited_mode})


# ── Charge time estimate ──────────────────────────────────────────────────────

@app.route("/api/chargetime", methods=["POST"])
def charge_time():
    data = request.get_json(force=True)
    from car_models import estimate_charge_time, get_model, reachable_range_km
    car = get_model(data.get("car_model", ""))
    if not car:
        return jsonify({"error": "Unknown car model"}), 400
    current_pct = float(data.get("current_pct", 20))
    target_pct = float(data.get("target_pct", 80))
    charger_kw = float(data.get("charger_kw", 7.2))
    result = estimate_charge_time(car, current_pct, target_pct, charger_kw)
    result["reachable_km"] = reachable_range_km(car, current_pct)
    result["car_model"] = data["car_model"]
    return jsonify(result)


# ── Other API routes ──────────────────────────────────────────────────────────

@app.route("/api/report", methods=["POST"])
def submit_report():
    data = request.get_json(force=True)
    required = {"station_name", "issue_type", "description"}
    if not required.issubset(data):
        return jsonify({"error": f"Missing: {required - data.keys()}"}), 400
    report = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "station_name": str(data["station_name"])[:200],
        "issue_type": str(data["issue_type"])[:100],
        "description": str(data["description"])[:500],
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "reported_by": session.get("username", "anonymous"),
    }
    _save_report(report)
    return jsonify({"status": "saved", "id": report["id"]}), 201


@app.route("/api/reports")
def get_reports():
    reports = _load_reports()
    return jsonify([] if reports.empty else reports.to_dict(orient="records"))


@app.route("/api/weather")
def get_weather():
    return jsonify(weather)


@app.route("/api/cars")
def get_cars():
    from car_models import CAR_MODELS
    return jsonify(list(CAR_MODELS.keys()))


@app.route("/api/winter-history")
def get_winter_history():
    return jsonify(winter_report)


@app.route("/")
def index():
    from car_models import CAR_MODELS
    urgency_counts = forecast_df["forecast_urgency"].value_counts().to_dict()
    return render_template(
        "index.html",
        weather=weather,
        weather_alert=weather["alert_message"] if weather["alert_active"] else None,
        overloaded_now=urgency_counts.get("Overloaded Now", 0),
        at_risk_soon=urgency_counts.get("1-2 Years", 0),
        total_fsas=len(areas),
        areas_json=areas.to_json(orient="records"),
        stations_json=stations_enhanced.to_json(orient="records"),
        machines_json=machines_df.to_json(orient="records"),
        recommended_json=recommended.merge(
            areas[["fsa", "latitude", "longitude"]], on="fsa", how="left"
        ).to_json(orient="records"),
        weather_json=json.dumps(weather),
        car_models=list(CAR_MODELS.keys()),
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
