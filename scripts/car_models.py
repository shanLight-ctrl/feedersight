"""EV car model database with real specs for charge time estimation."""

from __future__ import annotations

# battery_kwh: usable capacity
# range_km: real-world range (not EPA/WLTP, adjusted ~80%)
# max_ac_kw: max Level 2 acceptance rate
# max_dc_kw: max DC fast charge acceptance rate
CAR_MODELS: dict[str, dict] = {
    "Tesla Model 3 Standard Range":  {"battery_kwh": 57.5,  "range_km": 350, "max_ac_kw": 11.0, "max_dc_kw": 170},
    "Tesla Model 3 Long Range":      {"battery_kwh": 75.0,  "range_km": 570, "max_ac_kw": 11.0, "max_dc_kw": 250},
    "Tesla Model 3 Performance":     {"battery_kwh": 75.0,  "range_km": 530, "max_ac_kw": 11.0, "max_dc_kw": 250},
    "Tesla Model Y Long Range":      {"battery_kwh": 75.0,  "range_km": 530, "max_ac_kw": 11.0, "max_dc_kw": 250},
    "Tesla Model Y Performance":     {"battery_kwh": 75.0,  "range_km": 480, "max_ac_kw": 11.0, "max_dc_kw": 250},
    "Tesla Model S":                 {"battery_kwh": 100.0, "range_km": 600, "max_ac_kw": 11.0, "max_dc_kw": 250},
    "Tesla Model X":                 {"battery_kwh": 100.0, "range_km": 560, "max_ac_kw": 11.0, "max_dc_kw": 250},
    "Nissan Leaf (40 kWh)":          {"battery_kwh": 36.0,  "range_km": 240, "max_ac_kw": 6.6,  "max_dc_kw": 50},
    "Nissan Leaf (62 kWh)":          {"battery_kwh": 56.0,  "range_km": 340, "max_ac_kw": 6.6,  "max_dc_kw": 100},
    "Chevrolet Bolt EV":             {"battery_kwh": 60.0,  "range_km": 415, "max_ac_kw": 7.2,  "max_dc_kw": 55},
    "Chevrolet Bolt EUV":            {"battery_kwh": 60.0,  "range_km": 397, "max_ac_kw": 7.2,  "max_dc_kw": 55},
    "Hyundai IONIQ 5 Standard":      {"battery_kwh": 58.0,  "range_km": 360, "max_ac_kw": 11.0, "max_dc_kw": 220},
    "Hyundai IONIQ 5 Long Range":    {"battery_kwh": 72.6,  "range_km": 480, "max_ac_kw": 11.0, "max_dc_kw": 230},
    "Hyundai IONIQ 6 Standard":      {"battery_kwh": 53.0,  "range_km": 385, "max_ac_kw": 11.0, "max_dc_kw": 150},
    "Hyundai IONIQ 6 Long Range":    {"battery_kwh": 77.4,  "range_km": 570, "max_ac_kw": 11.0, "max_dc_kw": 233},
    "Kia EV6 Standard":              {"battery_kwh": 58.0,  "range_km": 370, "max_ac_kw": 11.0, "max_dc_kw": 180},
    "Kia EV6 Long Range":            {"battery_kwh": 77.4,  "range_km": 490, "max_ac_kw": 11.0, "max_dc_kw": 233},
    "Kia EV9 Long Range":            {"battery_kwh": 99.8,  "range_km": 505, "max_ac_kw": 11.0, "max_dc_kw": 233},
    "Ford F-150 Lightning Standard": {"battery_kwh": 98.0,  "range_km": 370, "max_ac_kw": 19.2, "max_dc_kw": 150},
    "Ford F-150 Lightning Extended": {"battery_kwh": 131.0, "range_km": 480, "max_ac_kw": 19.2, "max_dc_kw": 150},
    "Ford Mustang Mach-E Standard":  {"battery_kwh": 68.0,  "range_km": 400, "max_ac_kw": 11.0, "max_dc_kw": 115},
    "Ford Mustang Mach-E Extended":  {"battery_kwh": 91.0,  "range_km": 490, "max_ac_kw": 11.0, "max_dc_kw": 150},
    "Volkswagen ID.4 Standard":      {"battery_kwh": 52.0,  "range_km": 340, "max_ac_kw": 7.2,  "max_dc_kw": 110},
    "Volkswagen ID.4 Long Range":    {"battery_kwh": 77.0,  "range_km": 480, "max_ac_kw": 11.0, "max_dc_kw": 135},
    "BMW i4 eDrive35":               {"battery_kwh": 70.2,  "range_km": 480, "max_ac_kw": 11.0, "max_dc_kw": 180},
    "BMW i4 M50":                    {"battery_kwh": 80.7,  "range_km": 455, "max_ac_kw": 11.0, "max_dc_kw": 205},
    "BMW iX xDrive50":               {"battery_kwh": 105.2, "range_km": 620, "max_ac_kw": 11.0, "max_dc_kw": 195},
    "Mercedes EQB":                  {"battery_kwh": 66.5,  "range_km": 419, "max_ac_kw": 11.0, "max_dc_kw": 100},
    "Mercedes EQS":                  {"battery_kwh": 107.8, "range_km": 700, "max_ac_kw": 22.0, "max_dc_kw": 200},
    "Audi Q4 e-tron":                {"battery_kwh": 76.6,  "range_km": 488, "max_ac_kw": 11.0, "max_dc_kw": 135},
    "Rivian R1T":                    {"battery_kwh": 135.0, "range_km": 483, "max_ac_kw": 11.5, "max_dc_kw": 200},
    "Rivian R1S":                    {"battery_kwh": 135.0, "range_km": 516, "max_ac_kw": 11.5, "max_dc_kw": 200},
    "Polestar 2 Standard":           {"battery_kwh": 64.0,  "range_km": 440, "max_ac_kw": 11.0, "max_dc_kw": 130},
    "Polestar 2 Long Range":         {"battery_kwh": 78.0,  "range_km": 540, "max_ac_kw": 11.0, "max_dc_kw": 205},
    "Volvo XC40 Recharge":           {"battery_kwh": 69.0,  "range_km": 418, "max_ac_kw": 11.0, "max_dc_kw": 150},
    "Mini Cooper SE":                {"battery_kwh": 28.9,  "range_km": 193, "max_ac_kw": 11.0, "max_dc_kw": 50},
    "Mazda MX-30":                   {"battery_kwh": 30.0,  "range_km": 200, "max_ac_kw": 6.6,  "max_dc_kw": 50},
    "Subaru Solterra":               {"battery_kwh": 71.4,  "range_km": 422, "max_ac_kw": 6.6,  "max_dc_kw": 150},
    "Toyota bZ4X":                   {"battery_kwh": 71.4,  "range_km": 406, "max_ac_kw": 6.6,  "max_dc_kw": 150},
    "GMC Hummer EV":                 {"battery_kwh": 200.0, "range_km": 507, "max_ac_kw": 19.2, "max_dc_kw": 350},
}


def get_model(name: str) -> dict | None:
    return CAR_MODELS.get(name)


def estimate_charge_time(
    car: dict,
    current_pct: float,
    target_pct: float,
    charger_kw: float,
) -> dict:
    """
    Estimate charge time in minutes.
    Effective rate = min(charger_kw, car max rate).
    Uses a simplified two-phase model: linear to 80%, tapered 80-100%.
    """
    battery_kwh = car["battery_kwh"]
    is_dc = charger_kw >= 20
    max_rate = car["max_dc_kw"] if is_dc else car["max_ac_kw"]
    effective_kw = min(charger_kw, max_rate)

    needed_kwh = battery_kwh * (target_pct - current_pct) / 100

    if needed_kwh <= 0:
        return {"minutes": 0, "effective_kw": effective_kw, "note": "Already at target"}

    # Taper above 80%: average rate drops to ~60% of peak in 80-100% range
    if target_pct > 80 and current_pct < 80:
        kwh_to_80 = battery_kwh * (80 - current_pct) / 100
        kwh_80_to_target = battery_kwh * (target_pct - 80) / 100
        mins = (kwh_to_80 / effective_kw + kwh_80_to_target / (effective_kw * 0.60)) * 60
    elif current_pct >= 80:
        mins = (needed_kwh / (effective_kw * 0.60)) * 60
    else:
        mins = (needed_kwh / effective_kw) * 60

    note = ""
    if charger_kw > max_rate:
        note = f"Your car caps at {max_rate} kW — charger is faster but car limits the rate"

    return {
        "minutes": round(mins),
        "hours_mins": f"{int(mins//60)}h {int(mins%60)}m" if mins >= 60 else f"{int(mins)}m",
        "effective_kw": round(effective_kw, 1),
        "kwh_added": round(needed_kwh, 1),
        "note": note,
    }


def reachable_range_km(car: dict, current_pct: float) -> float:
    """How far the car can go from current battery %."""
    return round(car["range_km"] * current_pct / 100, 1)
