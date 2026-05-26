"""
Demand Response Optimizer — peak-shaving LP for EV charging schedules.
Scales from single driver → fleet → neighbourhood (FSA-wide).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from typing import Optional

# Typical Ontario hourly load shape (index 0=midnight, 23=11pm)
# Relative multipliers — scaled against FSA base load
ONTARIO_LOAD_SHAPE = np.array([
    0.72, 0.68, 0.65, 0.63, 0.62, 0.64,   # 0–5 am   (overnight low)
    0.72, 0.83, 0.92, 0.96, 0.99, 1.00,   # 6–11 am  (morning ramp)
    0.98, 0.97, 0.96, 0.95, 0.96, 1.00,   # 12–17 pm (midday plateau)
    1.02, 1.05, 1.03, 0.97, 0.90, 0.80,   # 18–23 pm (evening peak → decline)
])

HOUR_LABELS = [
    "12am","1am","2am","3am","4am","5am",
    "6am","7am","8am","9am","10am","11am",
    "12pm","1pm","2pm","3pm","4pm","5pm",
    "6pm","7pm","8pm","9pm","10pm","11pm",
]


def _base_load_curve(base_kw: float, weather_factor: float = 1.0) -> np.ndarray:
    return ONTARIO_LOAD_SHAPE * base_kw * weather_factor


def optimize(
    base_load_kw: float,
    feeder_capacity_kw: float,
    num_evs: int,
    avg_charge_kw: float = 7.2,
    avg_energy_needed_kwh: float = 30.0,
    weather_factor: float = 1.0,
    charge_window_start: int = 0,    # earliest hour EVs can charge
    charge_window_end: int = 23,     # latest hour EVs can charge
) -> dict:
    """
    LP peak-shaving: schedule num_evs EVs to minimise peak grid load.

    Returns schedule dict with per-hour load, recommended windows,
    peak reduction %, and a plain-English recommendation.
    """
    T = 24
    base = _base_load_curve(base_load_kw, weather_factor)

    # Hours available for charging
    avail = np.zeros(T)
    for h in range(charge_window_start, charge_window_end + 1):
        avail[h] = 1.0

    # --- LP formulation ---
    # Variables: [x_0, x_1, ..., x_23, z]
    # x_h = total kW of EV charging scheduled in hour h
    # z   = peak total load (to minimise)
    #
    # Objective: minimise z
    n_vars = T + 1
    c = np.zeros(n_vars)
    c[-1] = 1.0  # minimise z

    A_ub = []
    b_ub = []

    # 1) total load[h] <= z  for all h  →  base[h] + x_h - z <= 0
    for h in range(T):
        row = np.zeros(n_vars)
        row[h]  =  1.0
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(-base[h])

    # 2) charging only in allowed window  →  x_h <= 0 for unavailable hours
    for h in range(T):
        if avail[h] == 0:
            row = np.zeros(n_vars)
            row[h] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # 3) feeder capacity hard constraint  →  base[h] + x_h <= feeder_capacity
    for h in range(T):
        row = np.zeros(n_vars)
        row[h] = 1.0
        A_ub.append(row)
        b_ub.append(feeder_capacity_kw - base[h])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Equality: total energy delivered == num_evs * avg_energy_needed_kwh
    total_energy = num_evs * avg_energy_needed_kwh
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :T] = 1.0            # sum of hourly kW == total_energy (1h intervals)
    b_eq = np.array([total_energy])

    # Bounds: x_h in [0, num_evs * avg_charge_kw], z unbounded below
    max_ev_load = num_evs * avg_charge_kw
    bounds = [(0, max_ev_load)] * T + [(0, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.status != 0:
        # Fallback: spread evenly across off-peak hours
        x = np.zeros(T)
        off_peak = [h for h in range(T) if avail[h] and base[h] < base.mean()]
        if not off_peak:
            off_peak = [h for h in range(T) if avail[h]]
        per_hour = total_energy / max(len(off_peak), 1)
        for h in off_peak:
            x[h] = min(per_hour, max_ev_load)
    else:
        x = result.x[:T]

    total_load = base + x

    # --- Analysis ---
    unmanaged = base + np.full(T, total_energy / max(sum(avail), 1)) * avail
    peak_unmanaged = float(unmanaged.max())
    peak_managed   = float(total_load.max())
    peak_reduction = max(0.0, (peak_unmanaged - peak_managed) / peak_unmanaged * 100)

    headroom = feeder_capacity_kw - total_load
    safest_hours = sorted(range(T), key=lambda h: total_load[h])
    charge_hours = [h for h in range(T) if x[h] > 0.5]

    # Best consecutive 4-hour window
    best_window_start = min(
        (h for h in range(T - 3) if avail[h]),
        key=lambda h: total_load[h:h+4].mean(),
        default=22,
    )

    # Plain-English recommendation
    avoid = [HOUR_LABELS[h] for h in range(T)
             if total_load[h] > feeder_capacity_kw * 0.92]
    best_label = f"{HOUR_LABELS[best_window_start]}–{HOUR_LABELS[min(best_window_start+4,23)]}"

    if peak_reduction >= 10:
        summary = (f"Shifting {num_evs} EV{'s' if num_evs>1 else ''} to off-peak hours "
                   f"reduces grid peak by {peak_reduction:.0f}%. "
                   f"Best charging window: {best_label}.")
    else:
        summary = (f"Grid has enough headroom for {num_evs} EV{'s' if num_evs>1 else ''} "
                   f"tonight. Preferred window: {best_label}.")

    if avoid:
        summary += f" Avoid: {', '.join(avoid[:3])}."

    overcapacity_hours = [HOUR_LABELS[h] for h in range(T)
                          if total_load[h] > feeder_capacity_kw]

    return {
        "hours":            HOUR_LABELS,
        "base_load":        base.tolist(),
        "ev_schedule":      x.tolist(),
        "total_load":       total_load.tolist(),
        "unmanaged_load":   unmanaged.tolist(),
        "feeder_capacity":  feeder_capacity_kw,
        "peak_managed":     round(peak_managed, 1),
        "peak_unmanaged":   round(peak_unmanaged, 1),
        "peak_reduction_pct": round(peak_reduction, 1),
        "best_window":      best_label,
        "avoid_hours":      avoid[:4],
        "overcapacity_hours": overcapacity_hours,
        "summary":          summary,
        "num_evs":          num_evs,
        "total_energy_kwh": round(total_energy, 1),
        "feasible":         result.status == 0,
    }


def neighbourhood_schedule(areas_df, fsa: str, num_evs: Optional[int] = None,
                            weather_factor: float = 1.0) -> dict:
    """Wrapper that pulls FSA data from the areas DataFrame."""
    row = areas_df[areas_df["fsa"] == fsa]
    if row.empty:
        return {"error": f"FSA {fsa} not found"}
    r = row.iloc[0]
    evs = num_evs or int(r.get("ev_count", 100))
    return optimize(
        base_load_kw=float(r.get("base_load_kw", 5000)),
        feeder_capacity_kw=float(r.get("feeder_capacity_kw", 8000)),
        num_evs=evs,
        avg_charge_kw=float(r.get("avg_charger_power_kw", 7.2)),
        weather_factor=weather_factor,
    )
