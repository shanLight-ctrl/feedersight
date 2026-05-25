"""
Generate an interactive HTML map of FeederSight results.

Usage:
  python scripts/view_map.py              # opens map.html in your browser
  python scripts/view_map.py --no-open   # just writes the file
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "map.html"

URGENCY_COLOR = {
    "Overloaded Now":  "#d7191c",   # red
    "1-2 Years":       "#f4730a",   # orange
    "3-5 Years":       "#fecc02",   # yellow
    "6-10 Years":      "#92c5de",   # light blue
    "Safe 10+ Years":  "#2ca02c",   # green
}

URGENCY_ORDER = [
    "Overloaded Now",
    "1-2 Years",
    "3-5 Years",
    "6-10 Years",
    "Safe 10+ Years",
]


def _na(val, fmt: str = "") -> str:
    if pd.isna(val):
        return "—"
    if fmt:
        return format(val, fmt)
    return str(val)


def fsa_popup(row: pd.Series) -> str:
    urgency_html = (
        f'<span style="background:{URGENCY_COLOR.get(row["forecast_urgency"], "#888")}; '
        f'color:#fff; padding:2px 6px; border-radius:3px; font-weight:bold;">'
        f'{row["forecast_urgency"]}</span>'
    )
    return f"""
<div style="font-family:sans-serif; min-width:240px; font-size:13px">
  <b style="font-size:15px">{row['fsa']} — {row['area']}</b><br>
  {urgency_html}<br><br>
  <table style="width:100%; border-collapse:collapse">
    <tr><td style="color:#555">EV registrations</td><td><b>{int(row['ev_count']):,}</b></td></tr>
    <tr><td style="color:#555">Current risk ratio</td><td><b>{_na(row['risk_ratio'],'.2f')} × capacity</b></td></tr>
    <tr><td style="color:#555">Current status</td><td><b>{row['risk_level']}</b></td></tr>
    <tr><td colspan="2"><hr style="margin:4px 0"></td></tr>
    <tr><td colspan="2" style="color:#555; font-size:11px"><b>Years to overload (35% annual EV growth)</b></td></tr>
    <tr><td style="color:#555">Conservative (20%)</td><td>{_na(row.get('years_to_overload_conservative'),'g')} yr</td></tr>
    <tr><td style="color:#555">Moderate (35%)</td><td>{_na(row.get('years_to_overload_moderate'),'g')} yr</td></tr>
    <tr><td style="color:#555">Aggressive (50%)</td><td>{_na(row.get('years_to_overload_aggressive'),'g')} yr</td></tr>
    <tr><td colspan="2"><hr style="margin:4px 0"></td></tr>
    <tr><td style="color:#555">Risk in 3 yrs (moderate)</td><td>{_na(row.get('risk_moderate_3yr'),'.2f')} ×</td></tr>
    <tr><td style="color:#555">Risk in 5 yrs (moderate)</td><td>{_na(row.get('risk_moderate_5yr'),'.2f')} ×</td></tr>
  </table>
</div>
"""


def charger_popup(row: pd.Series) -> str:
    return (
        f"<b>{row.get('name','Charger')}</b><br>"
        f"Type: {row.get('type','—')}<br>"
        f"Power: {row.get('power_kw','—')} kW<br>"
        f"Operator: {row.get('operator','—')}<br>"
        f"Source: {row.get('data_source','—')}"
    )


def recommended_popup(row: pd.Series) -> str:
    return (
        f"<b>Recommended Site: {row['fsa']}</b><br>"
        f"{row.get('area','')}<br>"
        f"Priority score: {row.get('priority_score', 0):.3f}<br>"
        f"EV count: {int(row.get('ev_count', 0)):,}<br>"
        f"Chargers today: {int(row.get('charger_count', 0))}<br>"
        f"Risk: {row.get('risk_level','—')}"
    )


def legend_html() -> str:
    rows = "".join(
        f'<tr><td><span style="background:{URGENCY_COLOR[u]};display:inline-block;'
        f'width:14px;height:14px;border-radius:50%;margin-right:6px"></span></td>'
        f'<td style="font-size:12px">{u}</td></tr>'
        for u in URGENCY_ORDER
    )
    return f"""
    <div style="position:fixed; bottom:30px; left:10px; z-index:9999;
                background:white; padding:10px 14px; border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:sans-serif">
      <b style="font-size:13px">Forecast urgency<br><span style="font-weight:normal;font-size:11px">(moderate 35% CAGR)</span></b>
      <table style="margin-top:6px; border-collapse:collapse">{rows}</table>
      <hr style="margin:6px 0">
      <span style="font-size:10px;color:#666">Circle size ∝ EV count</span>
    </div>
    """


def build_map() -> folium.Map:
    forecast = pd.read_csv(PROCESSED / "risk_forecast.csv")
    areas = pd.read_csv(PROCESSED / "feeder_sight_areas.csv")
    chargers = pd.read_csv(PROCESSED / "existing_chargers.csv")
    recommended = pd.read_csv(PROCESSED / "recommended_sites.csv")

    # Merge forecast urgency + risk back onto areas for popups
    merged = areas.merge(
        forecast[["fsa", "forecast_urgency", "years_to_overload_conservative",
                  "years_to_overload_moderate", "years_to_overload_aggressive",
                  "risk_moderate_3yr", "risk_moderate_5yr"]],
        on="fsa", how="left",
    )

    center_lat = merged["latitude"].median()
    center_lon = merged["longitude"].median()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="CartoDB positron",
    )

    # ── Layer 1: FSA risk circles ─────────────────────────────────────────
    fsa_layer = folium.FeatureGroup(name="FSA risk forecast", show=True)
    ev_max = merged["ev_count"].max()

    for _, row in merged.iterrows():
        urgency = row.get("forecast_urgency", "Safe 10+ Years")
        color = URGENCY_COLOR.get(urgency, "#888")
        radius = max(300, (row["ev_count"] / ev_max) * 4000)

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            weight=1,
            tooltip=f"{row['fsa']} — {row.get('area','')} | {urgency}",
            popup=folium.Popup(fsa_popup(row), max_width=300),
        ).add_to(fsa_layer)

    fsa_layer.add_to(m)

    # ── Layer 2: Existing chargers (clustered) ────────────────────────────
    charger_cluster = MarkerCluster(name="Existing chargers", show=False)
    for _, row in chargers.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue
        icon_color = "red" if row.get("type") == "DC Fast" else "blue"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.Icon(color=icon_color, icon="bolt", prefix="fa"),
            tooltip=f"{row.get('name','Charger')} ({row.get('type','—')})",
            popup=folium.Popup(charger_popup(row), max_width=260),
        ).add_to(charger_cluster)
    charger_cluster.add_to(m)

    # ── Layer 3: Recommended new sites ───────────────────────────────────
    sites_layer = folium.FeatureGroup(name="Top 10 recommended sites", show=True)
    rec_merged = recommended.merge(areas[["fsa", "latitude", "longitude"]], on="fsa", how="left")
    for rank, (_, row) in enumerate(rec_merged.iterrows(), start=1):
        if pd.isna(row.get("latitude")):
            continue
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:14px; font-weight:bold; color:#fff; '
                     f'background:#5d3fd3; border-radius:50%; width:26px; height:26px; '
                     f'display:flex; align-items:center; justify-content:center; '
                     f'border:2px solid white; box-shadow:0 1px 4px rgba(0,0,0,0.4)">{rank}</div>',
                icon_size=(26, 26),
                icon_anchor=(13, 13),
            ),
            tooltip=f"#{rank} Priority: {row['fsa']} — {row.get('area','')}",
            popup=folium.Popup(recommended_popup(row), max_width=260),
        ).add_to(sites_layer)
    sites_layer.add_to(m)

    # ── Legend + layer control ────────────────────────────────────────────
    m.get_root().html.add_child(folium.Element(legend_html()))
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true", help="Write map.html without opening it")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    for f in ["feeder_sight_areas.csv", "risk_forecast.csv", "existing_chargers.csv", "recommended_sites.csv"]:
        if not (PROCESSED / f).exists():
            raise SystemExit(
                f"Missing {f}. Run first:\n"
                "  python scripts/build_feeder_sight.py\n"
                "  python scripts/forecast_risk.py"
            )

    print("Building map...")
    m = build_map()
    m.save(str(args.out))
    print(f"Saved: {args.out}")

    if not args.no_open:
        webbrowser.open(args.out.as_uri())
        print("Opening in browser...")


if __name__ == "__main__":
    main()
