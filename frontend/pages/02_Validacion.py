"""
pages/02_Validacion.py
Página de validación geoestadística.
Muestra: LOO-CV por estación DAGMA, variogramas, Moran Global, LISA,
y tabla de KPIs con indicadores de cumplimiento.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Validación · GeoVision-CLIP",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent.parent / "assets" / "styles.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from components.map_utils import DAGMA_STATIONS, CONTAMINANTE_UNITS
from components.model_loader import sidebar_model_status

DARK = "#0a0c10"
SURFACE = "#10141c"
CARD = "#1a2030"
BORDER = "#252d3d"
ACCENT = "#4af0b0"
WARN = "#f0a94a"
DANGER = "#f05a4a"
TEXT = "#e8ecf4"
DIM = "#4a5568"
SECONDARY = "#8a96a8"

FONT_MONO = "DM Mono, monospace"
FONT_DISP = "Syne, sans-serif"

# ── Config Plotly ─────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=DARK,
    font=dict(family=FONT_MONO, color=SECONDARY, size=10),
    margin=dict(l=40, r=20, t=36, b=40),
)

PLOTLY_XAXIS = dict(
    gridcolor=BORDER,
    zerolinecolor=BORDER,
    color=SECONDARY,
)

PLOTLY_YAXIS = dict(
    gridcolor=BORDER,
    zerolinecolor=BORDER,
    color=SECONDARY,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand-title">GeoVision<span class="accent">-CLIP</span></div>'
        '<div class="sidebar-brand-sub">Cali · UAO · 2026</div></div>',
        unsafe_allow_html=True,
    )

    contaminante = st.selectbox(
        "Contaminante",
        ["NO2", "SO2", "O3"],
        format_func=lambda x: {
            "NO2": "NO₂",
            "SO2": "SO₂",
            "O3": "O₃",
        }[x],
    )

    horizon = st.selectbox("Horizonte", ["T+1", "T+3", "T+7"])


    sidebar_model_status()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="page-header">
        <div class="eyebrow">GeoVision-CLIP Cali · Módulo de validación</div>
        <h2>Validación Geoestadística</h2>
        <div class="subtitle">
            LOO-CV sobre estaciones DAGMA · Variogramas · Moran Global · LISA
            · {contaminante} · {horizon}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Datos sintéticos ───────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n_stations = len(DAGMA_STATIONS)

rmse_targets = {"NO2": 6.0, "SO2": 4.0, "O3": 10.0}

h_penalty = {
    "T+1": 1.0,
    "T+3": 1.25,
    "T+7": 1.5,
}[horizon]

observed = rng.uniform(20, 80, n_stations)
noise = rng.normal(
    0,
    rmse_targets[contaminante] * h_penalty,
    n_stations,
)

predicted = observed + noise
residuals = observed - predicted

rmse_val = float(np.sqrt(np.mean(residuals**2)))
mae_val = float(np.mean(np.abs(residuals)))
r2_val = float(1 - np.var(residuals) / np.var(observed))

# ── KPI Cards ──────────────────────────────────────────────────────────────────
kpi_thresholds = {
    "NO2": {"min": 8, "exc": 4},
    "SO2": {"min": 6, "exc": 3},
    "O3": {"min": 12, "exc": 6},
}

thr = kpi_thresholds[contaminante]


def kpi_color(val, min_thr, exc_thr, lower_better=True):
    if lower_better:
        if val <= exc_thr:
            return ACCENT
        elif val <= min_thr:
            return WARN
        else:
            return DANGER
    else:
        if val >= exc_thr:
            return ACCENT
        elif val >= min_thr:
            return WARN
        else:
            return DANGER


rmse_col = kpi_color(rmse_val, thr["min"], thr["exc"])
r2_col = kpi_color(r2_val, 0.55, 0.75, lower_better=False)

c1, c2, c3, c4 = st.columns(4)

for col, label, val, color, unit in [
    (
        c1,
        "RMSE LOO-CV",
        f"{rmse_val:.2f}",
        rmse_col,
        CONTAMINANTE_UNITS[contaminante],
    ),
    (
        c2,
        "MAE LOO-CV",
        f"{mae_val:.2f}",
        WARN,
        CONTAMINANTE_UNITS[contaminante],
    ),
    (
        c3,
        "R² LOO-CV",
        f"{r2_val:.3f}",
        r2_col,
        "",
    ),
    (
        c4,
        "N estaciones",
        "9",
        ACCENT,
        "DAGMA",
    ),
]:
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color:{color};">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "LOO-CV por estación",
    "Variograma",
    "Moran Global",
    "LISA",
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:

    col_l, col_r = st.columns([3, 2], gap="medium")

    with col_l:

        station_names = [s["name"] for s in DAGMA_STATIONS]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Observado",
            x=station_names,
            y=observed,
            marker_color=ACCENT,
            marker_opacity=0.8,
            width=0.35,
            offset=-0.18,
        ))

        fig.add_trace(go.Bar(
            name="Predicho",
            x=station_names,
            y=predicted,
            marker_color="#4a9af0",
            marker_opacity=0.8,
            width=0.35,
            offset=0.18,
        ))

        fig.add_trace(go.Scatter(
            name="Residuo",
            x=station_names,
            y=residuals,
            mode="markers+lines",
            marker=dict(
                color=WARN,
                size=6,
                symbol="diamond",
            ),
            line=dict(
                color=WARN,
                width=1,
                dash="dot",
            ),
            yaxis="y2",
        ))

        fig.update_layout(
            **PLOTLY_BASE,
            title=dict(
                text=f"LOO-CV · {contaminante} · {horizon}",
                font=dict(
                    family=FONT_DISP,
                    size=13,
                    color=TEXT,
                ),
            ),
            barmode="overlay",
            height=360,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=9),
            ),
            xaxis=dict(
                tickangle=-35,
                **PLOTLY_XAXIS,
            ),
            yaxis=dict(
                title=CONTAMINANTE_UNITS[contaminante],
                **PLOTLY_YAXIS,
            ),
            yaxis2=dict(
                title="Residuo",
                overlaying="y",
                side="right",
                color=WARN,
                gridcolor="rgba(0,0,0,0)",
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )