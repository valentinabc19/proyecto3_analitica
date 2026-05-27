"""
components/map_utils.py
Utilidades de mapa para GeoVision-CLIP Cali.
Genera mapas Folium con: estaciones DAGMA, BBox del proyecto,
overlays de gradiente de contaminación e incertidumbre Kriging.

BBox del proyecto: [-76.75, 3.20, -76.30, 3.75]  (lon_min, lat_min, lon_max, lat_max)
"""

from __future__ import annotations

from typing import Optional
import numpy as np

# ── Constantes geográficas ────────────────────────────────────────────────────
CALI_CENTER    = [3.4516, -76.5320]   # lat, lon centro de Cali
BBOX           = {
    "lon_min": -76.75, "lat_min": 3.20,
    "lon_max": -76.30, "lat_max": 3.75,
}

# ── Estaciones DAGMA ──────────────────────────────────────────────────────────
DAGMA_STATIONS = [
    {"id": "E01", "name": "Univalle",         "lat": 3.3750,  "lon": -76.5350, "zona": "Sur"},
    {"id": "E02", "name": "Compartir",        "lat": 3.3860,  "lon": -76.5260, "zona": "Sur"},
    {"id": "E03", "name": "Fátima",           "lat": 3.4320,  "lon": -76.5100, "zona": "Centro"},
    {"id": "E04", "name": "Pance",            "lat": 3.3400,  "lon": -76.5700, "zona": "Sur"},
    {"id": "E05", "name": "Aguablanca",       "lat": 3.4200,  "lon": -76.4700, "zona": "Oriente"},
    {"id": "E06", "name": "Meléndez",         "lat": 3.3600,  "lon": -76.5500, "zona": "Sur"},
    {"id": "E07", "name": "Yumbo (Acopi)",    "lat": 3.5800,  "lon": -76.5000, "zona": "Norte-Industrial"},
    {"id": "E08", "name": "San Antonio",      "lat": 3.4480,  "lon": -76.5450, "zona": "Centro"},
    {"id": "E09", "name": "Siloé",            "lat": 3.4350,  "lon": -76.5620, "zona": "Ladera"},
]

# ── Paletas de colores por contaminante ───────────────────────────────────────
COLORMAP = {
    "NO2": ["#0a2342", "#1e4d6b", "#2e8b8b", "#4af0b0", "#f0e94a", "#f0a94a", "#f05a4a"],
    "SO2": ["#0a1a2e", "#1a3a5c", "#2060a0", "#4a9af0", "#a0d0f0", "#f0e0a0", "#f0a040"],
    "O3":  ["#0a2010", "#1a4020", "#2a7040", "#4ab870", "#a0e080", "#f0f040", "#f07000"],
}

CONTAMINANTE_UNITS = {"NO2": "µg/m³", "SO2": "µg/m³", "O3": "µg/m³"}
CONTAMINANTE_THRESHOLDS = {
    "NO2": {"bueno": 40, "moderado": 80, "malo": 200},
    "SO2": {"bueno": 20, "moderado": 50, "malo": 100},
    "O3":  {"bueno": 60, "moderado": 100, "malo": 180},
}


def build_base_map(zoom: int = 12, show_bbox: bool = True) -> "folium.Map":
    """
    Construye el mapa base de Cali con las 9 estaciones DAGMA.
    Usa tiles de CartoDB Dark Matter para coherencia con el tema oscuro.
    """
    import folium  # type: ignore

    m = folium.Map(
        location=CALI_CENTER,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        attr="© OpenStreetMap, © CartoDB",
        prefer_canvas=True,
    )

    # ── BBox del proyecto ────────────────────────────────────────────────────
    if show_bbox:
        folium.Rectangle(
            bounds=[
                [BBOX["lat_min"], BBOX["lon_min"]],
                [BBOX["lat_max"], BBOX["lon_max"]],
            ],
            color="#4af0b0",
            weight=1.5,
            fill=False,
            dash_array="6 4",
            tooltip="Área de estudio: −76.75 / 3.20 → −76.30 / 3.75",
        ).add_to(m)

    # ── Estaciones DAGMA ─────────────────────────────────────────────────────
    station_group = folium.FeatureGroup(name="Estaciones DAGMA", show=True)
    for s in DAGMA_STATIONS:
        popup_html = f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                    background:#10141c;color:#e8ecf4;padding:0.6rem 0.85rem;
                    border:1px solid #252d3d;border-radius:5px;min-width:160px;">
            <div style="color:#4af0b0;font-weight:600;margin-bottom:0.3rem;">
                {s['id']} · {s['name']}
            </div>
            <div style="color:#8a96a8;font-size:0.7rem;">
                Zona: {s['zona']}<br>
                Lat: {s['lat']:.4f} · Lon: {s['lon']:.4f}
            </div>
        </div>
        """
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=7,
            color="#4af0b0",
            fill=True,
            fill_color="#4af0b0",
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{s['id']}: {s['name']}",
        ).add_to(station_group)

    station_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def add_gradient_overlay(
    m: "folium.Map",
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    values: np.ndarray,
    contaminante: str = "NO2",
    opacity: float = 0.65,
    layer_name: str = "Predicción",
) -> "folium.Map":
    """
    Agrega un overlay de gradiente de contaminación al mapa.

    Parameters
    ----------
    grid_lats, grid_lons : 1-D arrays de coordenadas de la grilla.
    values               : 2-D array (len(lats) × len(lons)) con concentraciones.
    contaminante         : "NO2" | "SO2" | "O3"
    """
    import folium  # type: ignore
    import branca.colormap as cm  # type: ignore

    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    colors = COLORMAP.get(contaminante, COLORMAP["NO2"])

    colormap = cm.LinearColormap(
        colors=colors,
        vmin=vmin,
        vmax=vmax,
        caption=f"{contaminante} [{CONTAMINANTE_UNITS[contaminante]}]",
    )

    # Construir heatmap como matriz de puntos (ImageOverlay si se tiene raster)
    from folium.plugins import HeatMap  # type: ignore

    heat_data = []
    for i, lat in enumerate(grid_lats):
        for j, lon in enumerate(grid_lons):
            v = values[i, j]
            if not np.isnan(v):
                heat_data.append([lat, lon, float(v)])

    HeatMap(
        heat_data,
        name=layer_name,
        min_opacity=0.3,
        max_opacity=opacity,
        radius=18,
        blur=12,
        gradient={
            "0.0": colors[0],
            "0.3": colors[2],
            "0.6": colors[4],
            "0.85": colors[5],
            "1.0": colors[6],
        },
    ).add_to(m)

    colormap.add_to(m)
    return m


def add_uncertainty_overlay(
    m: "folium.Map",
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    variance: np.ndarray,
    layer_name: str = "Incertidumbre (σ²)",
) -> "folium.Map":
    """
    Agrega capa de incertidumbre Kriging (σ²) con opacidad proporcional.
    Zonas de mayor varianza aparecen más oscuras/opacas.
    """
    import folium  # type: ignore
    from folium.plugins import HeatMap  # type: ignore

    heat_data = []
    var_max = float(np.nanmax(variance)) + 1e-9
    for i, lat in enumerate(grid_lats):
        for j, lon in enumerate(grid_lons):
            v = variance[i, j]
            if not np.isnan(v):
                heat_data.append([lat, lon, float(v) / var_max])

    HeatMap(
        heat_data,
        name=layer_name,
        min_opacity=0.15,
        max_opacity=0.5,
        radius=20,
        blur=15,
        gradient={
            "0.0": "#0a0c10",
            "0.5": "#f0a94a",
            "1.0": "#f05a4a",
        },
    ).add_to(m)
    return m


def add_click_marker(
    m: "folium.Map",
    lat: float,
    lon: float,
    predictions: dict,
    horizon: str,
) -> "folium.Map":
    """
    Agrega un marcador en el punto consultado con los valores predichos.

    predictions: {"NO2": (val, sigma), "SO2": (val, sigma), "O3": (val, sigma)}
    """
    import folium  # type: ignore

    rows = ""
    for gas, (val, sigma) in predictions.items():
        rows += (
            f"<tr><td style='color:#8a96a8;padding-right:0.5rem;'>{gas}</td>"
            f"<td style='color:#e8ecf4;font-weight:500;'>{val:.2f}"
            f"<span style='color:#4a5568;font-size:0.7rem;'> ±{sigma:.2f}</span></td>"
            f"<td style='color:#4a5568;'>{CONTAMINANTE_UNITS[gas]}</td></tr>"
        )

    popup_html = f"""
    <div style="font-family:'DM Mono',monospace;background:#10141c;
                color:#e8ecf4;padding:0.75rem 1rem;border:1px solid #4af0b0;
                border-radius:6px;min-width:200px;">
        <div style="color:#4af0b0;font-size:0.75rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">
            Horizonte {horizon}
        </div>
        <div style="color:#8a96a8;font-size:0.68rem;margin-bottom:0.5rem;">
            {lat:.5f}, {lon:.5f}
        </div>
        <table style="border-collapse:collapse;font-size:0.8rem;width:100%;">
            {rows}
        </table>
    </div>
    """
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"Consulta: {lat:.4f}, {lon:.4f}",
        icon=folium.DivIcon(
            html=(
                '<div style="width:14px;height:14px;background:#4af0b0;'
                "border:2px solid #10141c;border-radius:50%;"
                'box-shadow:0 0 8px #4af0b0;"></div>'
            ),
            icon_size=(14, 14),
            icon_anchor=(7, 7),
        ),
    ).add_to(m)
    return m


def synthetic_prediction_grid(
    center_lat: float,
    center_lon: float,
    radius_deg: float = 0.05,
    n: int = 20,
    contaminante: str = "NO2",
    horizon_idx: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera una grilla sintética de predicciones para modo demo
    (cuando los modelos reales no están cargados).

    Retorna: (lats, lons, values, variance)
    """
    rng = np.random.default_rng(seed + horizon_idx)
    lats = np.linspace(center_lat - radius_deg, center_lat + radius_deg, n)
    lons = np.linspace(center_lon - radius_deg, center_lon + radius_deg, n)

    base = {"NO2": 35.0, "SO2": 12.0, "O3": 55.0}[contaminante]
    scale = {"NO2": 20.0, "SO2": 8.0, "O3": 25.0}[contaminante]
    decay = 1.0 + 0.15 * horizon_idx  # degradación por horizonte

    x, y = np.meshgrid(
        np.linspace(-1, 1, n), np.linspace(-1, 1, n)
    )
    values = (
        base * decay
        + scale * np.exp(-((x**2 + y**2) / 0.5))
        + scale * 0.3 * rng.standard_normal((n, n))
    )
    variance = (
        (scale * 0.15 * decay) ** 2
        * (0.5 + 0.5 * (x**2 + y**2))
        * rng.uniform(0.8, 1.2, (n, n))
    )
    return lats, lons, np.clip(values, 0, None), np.abs(variance)