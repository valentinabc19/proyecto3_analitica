"""
GeoVision-CLIP Cali — Sistema de Estimación de Contaminación Atmosférica
Universidad Autónoma de Occidente · Analítica de Datos I · 2026
"""

import streamlit as st
from pathlib import Path

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoVision-CLIP · Cali",
    page_icon="assets/favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "GeoVision-CLIP Cali — UAO Ingeniería de Datos e IA, 2026",
    },
)

# ── CSS global ───────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Hero de la página de inicio ──────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-block">
        <div class="hero-eyebrow">Universidad Autónoma de Occidente · Analítica de Datos I</div>
        <h1 class="hero-title">GeoVision<span class="accent">-CLIP</span> Cali</h1>
        <p class="hero-sub">
            Estimación de contaminación atmosférica en puntos no muestreados<br>
            mediante Deep Learning y Estadística Geoespacial Avanzada
        </p>
        <div class="hero-tags">
            <span class="tag">Sentinel-5P TROPOMI</span>
            <span class="tag">Sentinel-2 MSI</span>
            <span class="tag">CLIP + SAE</span>
            <span class="tag">ConvLSTM</span>
            <span class="tag">ST-Kriging</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Panel de métricas KPI ────────────────────────────────────────────────────
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

kpi_data = [
    ("9", "Estaciones DAGMA", "Puntos de validación ground-truth"),
    ("3", "Contaminantes", "NO\u2082 · SO\u2082 · O\u2083"),
    ("3", "Horizontes", "T+1 · T+3 · T+7 días"),
    ("5", "Años de datos", "2020 – 2024"),
]

for col, (val, label, sub) in zip([col1, col2, col3, col4], kpi_data):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-val">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Descripción del sistema ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c1, c2 = st.columns([3, 2])

with c1:
    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-title">Arquitectura del Sistema</h3>
            <p class="section-body">
                El pipeline integra cuatro capas tecnológicas para producir mapas continuos
                de calidad del aire sobre el área metropolitana de Santiago de Cali
                (BBox: −76.75, 3.20, −76.30, 3.75).
            </p>
            <div class="pipeline-steps">
                <div class="pipe-step"><span class="step-num">01</span><span class="step-text">Panel espacio-temporal <span class="dim">&gt; 50 GB en Zarr/Parquet</span></span></div>
                <div class="pipe-step"><span class="step-num">02</span><span class="step-text">GeoVision-CLIP + SAE <span class="dim">embeddings ℝ²⁵⁶</span></span></div>
                <div class="pipe-step"><span class="step-num">03</span><span class="step-text">ConvLSTM bidireccional <span class="dim">secuencias 8 fechas</span></span></div>
                <div class="pipe-step"><span class="step-num">04</span><span class="step-text">ST-Kriging <span class="dim">superficie continua + σ²</span></span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-title">Navegación</h3>
            <p class="section-body">Selecciona una sección en la barra lateral:</p>
            <div class="nav-list">
                <div class="nav-item">
                    <span class="nav-num">01</span>
                    <div>
                        <strong>Prediccion</strong>
                        <div class="nav-desc">Mapas de gradiente interactivos por contaminante y horizonte</div>
                    </div>
                </div>
                <div class="nav-item">
                    <span class="nav-num">02</span>
                    <div>
                        <strong>Validacion</strong>
                        <div class="nav-desc">LOO-CV sobre estaciones DAGMA · Moran · LISA · Variogramas</div>
                    </div>
                </div>
                <div class="nav-item">
                    <span class="nav-num">03</span>
                    <div>
                        <strong>Panel EDA</strong>
                        <div class="nav-desc">Distribuciones, cobertura espacial y series temporales del panel</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        Facultad de Ingenierías · Ingeniería de Datos e IA · Cali, Colombia · 2026
    </div>
    """,
    unsafe_allow_html=True,
)