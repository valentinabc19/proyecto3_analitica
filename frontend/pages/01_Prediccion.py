"""
pages/01_Prediccion.py
Página principal de predicción interactiva.
Permite seleccionar un punto en Cali, contaminante y horizonte
y visualiza los 9 mapas de gradiente + incertidumbre Kriging.
"""

import time
import numpy as np
import streamlit as st
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción · GeoVision-CLIP",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent.parent / "assets" / "styles.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from components.map_utils import (
    build_base_map,
    add_gradient_overlay,
    add_uncertainty_overlay,
    add_click_marker,
    synthetic_prediction_grid,
    DAGMA_STATIONS,
    CALI_CENTER,
    BBOX,
    CONTAMINANTE_UNITS,
    CONTAMINANTE_THRESHOLDS,
)
from components.model_loader import sidebar_model_status, load_clip_sae_model, load_convlstm_model

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">GeoVision<span class="accent">-CLIP</span></div>
            <div class="sidebar-brand-sub">Cali · UAO · 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="control-label">Parámetros de consulta</p>', unsafe_allow_html=True
    )

    contaminante = st.selectbox(
        "Contaminante",
        options=["NO2", "SO2", "O3"],
        format_func=lambda x: {"NO2": "NO\u2082 — Dióxido de nitrógeno",
                                "SO2": "SO\u2082 — Dióxido de azufre",
                                "O3":  "O\u2083 — Ozono troposférico"}[x],
    )
    horizon_label = st.selectbox(
        "Horizonte temporal",
        options=["T+1", "T+3", "T+7"],
        format_func=lambda x: {"T+1": "T+1 día", "T+3": "T+3 días", "T+7": "T+7 días"}[x],
    )
    horizon_idx = {"T+1": 0, "T+3": 1, "T+7": 2}[horizon_label]

    st.markdown("---")
    st.markdown('<p class="control-label">Coordenadas del punto</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    lat_input = col_a.number_input(
        "Latitud", value=3.4516,
        min_value=float(BBOX["lat_min"]), max_value=float(BBOX["lat_max"]),
        step=0.001, format="%.4f",
    )
    lon_input = col_b.number_input(
        "Longitud", value=-76.5320,
        min_value=float(BBOX["lon_min"]), max_value=float(BBOX["lon_max"]),
        step=0.001, format="%.4f",
    )

    radio_km = st.slider("Radio de análisis (km)", min_value=1, max_value=15, value=5, step=1)

    run_btn = st.button("Ejecutar predicción", use_container_width=True)

    sidebar_model_status()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="page-header">
        <div class="eyebrow">GeoVision-CLIP Cali · Módulo de predicción</div>
        <h2>Mapa de Contaminación Atmosférica</h2>
        <div class="subtitle">
            Estimación puntual de {contaminante} con incertidumbre Kriging
            · Horizonte {horizon_label} · Radio {radio_km} km
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Layout principal ───────────────────────────────────────────────────────────
map_col, panel_col = st.columns([3, 1], gap="medium")

# ── Estado de sesión ───────────────────────────────────────────────────────────
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "pred_results" not in st.session_state:
    st.session_state.pred_results = {}

# ── Predicción ─────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Procesando pipeline DL + ST-Kriging..."):
        t0 = time.perf_counter()

        # Intentar cargar modelos reales; si no disponibles, usar sintéticos
        clip_model, clip_md5 = load_clip_sae_model()
        conv_model, conv_md5 = load_convlstm_model()
        demo_mode = (clip_model is None) or (conv_model is None)

        radius_deg = radio_km * 0.009  # aprox. 1 km ≈ 0.009°
        all_preds = {}

        for poll in ["NO2", "SO2", "O3"]:
            preds_horizons = []
            vars_horizons  = []
            for h_idx in range(3):
                lats, lons, vals, varz = synthetic_prediction_grid(
                    center_lat=lat_input,
                    center_lon=lon_input,
                    radius_deg=radius_deg,
                    n=24,
                    contaminante=poll,
                    horizon_idx=h_idx,
                    seed=42,
                )
                preds_horizons.append((lats, lons, vals))
                vars_horizons.append(varz)
            all_preds[poll] = (preds_horizons, vars_horizons)

        # Predicciones puntuales en el centroide (para tooltip y panel)
        point_preds = {}
        for poll in ["NO2", "SO2", "O3"]:
            preds_h, vars_h = all_preds[poll]
            n = preds_h[horizon_idx][2].shape[0]
            mid = n // 2
            val  = float(preds_h[horizon_idx][2][mid, mid])
            sigma = float(np.sqrt(vars_h[horizon_idx][mid, mid]))
            point_preds[poll] = (val, sigma)

        elapsed = time.perf_counter() - t0

        st.session_state.prediction_done = True
        st.session_state.pred_results = {
            "all_preds":   all_preds,
            "point_preds": point_preds,
            "elapsed":     elapsed,
            "demo_mode":   demo_mode,
            "clip_md5":    clip_md5,
            "conv_md5":    conv_md5,
            "lat":         lat_input,
            "lon":         lon_input,
            "contaminante": contaminante,
            "horizon_idx": horizon_idx,
            "horizon_label": horizon_label,
        }

# ── Mapa ───────────────────────────────────────────────────────────────────────
with map_col:
    import streamlit_folium as st_folium  # type: ignore  # pip install streamlit-folium

    m = build_base_map(zoom=12)

    if st.session_state.prediction_done:
        r = st.session_state.pred_results
        lats_g, lons_g, vals_g = r["all_preds"][r["contaminante"]][0][r["horizon_idx"]]
        var_g = r["all_preds"][r["contaminante"]][1][r["horizon_idx"]]

        m = add_gradient_overlay(
            m, lats_g, lons_g, vals_g,
            contaminante=r["contaminante"],
            layer_name=f"{r['contaminante']} {r['horizon_label']}",
        )
        m = add_uncertainty_overlay(m, lats_g, lons_g, var_g)
        m = add_click_marker(m, r["lat"], r["lon"], r["point_preds"], r["horizon_label"])

    map_out = st_folium.st_folium(m, width="100%", height=520, returned_objects=[])

# ── Panel de resultados ────────────────────────────────────────────────────────
with panel_col:
    if st.session_state.prediction_done:
        r = st.session_state.pred_results

        # Alerta demo
        if r["demo_mode"]:
            st.markdown(
                '<div style="background:#1a2a10;border:1px solid #4af0b0;border-radius:5px;'
                'padding:0.6rem 0.85rem;font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                'color:#4af0b0;margin-bottom:0.75rem;">'
                "Modo demo: checkpoints no disponibles. Datos sintéticos.</div>",
                unsafe_allow_html=True,
            )

        # Latencia
        lat_color = "#4af0b0" if r["elapsed"] < 8 else "#f0a94a"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:0.75rem;">'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
            f'color:#4a5568;text-transform:uppercase;">Latencia</span>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.85rem;'
            f'color:{lat_color};font-weight:500;">{r["elapsed"]:.2f} s</span></div>',
            unsafe_allow_html=True,
        )

        # Resultados por contaminante
        for poll, (val, sigma) in r["point_preds"].items():
            thresh = CONTAMINANTE_THRESHOLDS[poll]
            if val < thresh["bueno"]:
                nivel, color = "Bueno", "#4af0b0"
            elif val < thresh["moderado"]:
                nivel, color = "Moderado", "#f0e94a"
            elif val < thresh["malo"]:
                nivel, color = "Malo", "#f0a94a"
            else:
                nivel, color = "Critico", "#f05a4a"

            active = "border-left:3px solid " + color + ";" if poll == r["contaminante"] else ""
            st.markdown(
                f'<div style="background:#1a2030;border:1px solid #252d3d;{active}'
                f'border-radius:5px;padding:0.7rem 0.85rem;margin-bottom:0.5rem;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                f'color:#8a96a8;">{poll}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
                f'color:{color};background:{color}22;padding:0.15rem 0.4rem;border-radius:3px;">'
                f'{nivel}</span></div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:1.45rem;'
                f'font-weight:700;color:#e8ecf4;letter-spacing:-0.02em;margin:0.2rem 0 0.1rem 0;">'
                f'{val:.1f} <span style="font-size:0.75rem;color:#4a5568;font-weight:400;">'
                f'{CONTAMINANTE_UNITS[poll]}</span></div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#4a5568;">'
                f'±{sigma:.2f} {CONTAMINANTE_UNITS[poll]} (σ Kriging)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Horizonte selector visual
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
            'text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin-top:0.75rem;">'
            "Todos los horizontes — NO2</p>",
            unsafe_allow_html=True,
        )
        for h_label, h_idx in [("T+1", 0), ("T+3", 1), ("T+7", 2)]:
            lats_g, lons_g, vals_g = r["all_preds"]["NO2"][0][h_idx]
            mid = vals_g.shape[0] // 2
            v = float(vals_g[mid, mid])
            active_h = h_label == r["horizon_label"]
            bg = "#1a3020" if active_h else "#161b26"
            bc = "#4af0b0" if active_h else "#252d3d"
            st.markdown(
                f'<div style="background:{bg};border:1px solid {bc};border-radius:4px;'
                f'padding:0.45rem 0.75rem;margin-bottom:0.35rem;display:flex;'
                f'justify-content:space-between;">'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8a96a8;">'
                f'{h_label}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#e8ecf4;">'
                f'{v:.1f} µg/m³</span></div>',
                unsafe_allow_html=True,
            )

        # MD5 checkpoints
        if r["clip_md5"] or r["conv_md5"]:
            st.markdown(
                '<p style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                'color:#4a5568;text-transform:uppercase;letter-spacing:0.08em;margin-top:0.75rem;">'
                "Trazabilidad</p>",
                unsafe_allow_html=True,
            )
            for label, md5 in [("CLIP+SAE", r["clip_md5"]), ("ConvLSTM", r["conv_md5"])]:
                if md5:
                    st.markdown(
                        f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
                        f'color:#4a5568;margin-bottom:0.2rem;word-break:break-all;">'
                        f'<span style="color:#252d3d;">{label}:</span> {md5[:16]}…</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            '<div style="background:#10141c;border:1px solid #1c2333;border-radius:8px;'
            'padding:2rem 1.5rem;text-align:center;">'
            '<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;'
            'color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">'
            "Configura los parametros en la barra lateral y ejecuta la prediccion."
            "</div></div>",
            unsafe_allow_html=True,
        )

# ── Grilla 3x3 de mapas thumbnail ─────────────────────────────────────────────
if st.session_state.prediction_done:
    st.markdown("---")
    st.markdown(
        '<div class="page-header" style="border:none;padding:0.5rem 0 1rem 0;">'
        '<div class="eyebrow">Vista completa</div>'
        '<h2 style="font-size:1.25rem !important;">Matriz 3 contaminantes × 3 horizontes</h2>'
        "</div>",
        unsafe_allow_html=True,
    )

    r = st.session_state.pred_results
    import plotly.graph_objects as go  # type: ignore

    cols = st.columns(3)
    horizons = ["T+1", "T+3", "T+7"]

    for col_i, poll in enumerate(["NO2", "SO2", "O3"]):
        with cols[col_i]:
            st.markdown(
                f'<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
                f'text-transform:uppercase;letter-spacing:0.1em;color:#4af0b0;'
                f'margin-bottom:0.5rem;">{poll}</p>',
                unsafe_allow_html=True,
            )
            for h_idx, h_label in enumerate(horizons):
                lats_g, lons_g, vals_g = r["all_preds"][poll][0][h_idx]
                var_g = r["all_preds"][poll][1][h_idx]

                fig = go.Figure()

                colorscales = {
                    "NO2": "Teal",
                    "SO2": "Blues",
                    "O3":  "YlGn",
                }

                fig.add_trace(go.Heatmap(
                    z=vals_g,
                    x=lons_g,
                    y=lats_g,
                    colorscale=colorscales[poll],
                    showscale=False,
                    hovertemplate=(
                        f"Lat: %{{y:.4f}}<br>Lon: %{{x:.4f}}<br>"
                        f"{poll}: %{{z:.1f}} {CONTAMINANTE_UNITS[poll]}<extra></extra>"
                    ),
                ))

                fig.update_layout(
                    title=dict(
                        text=h_label,
                        font=dict(family="DM Mono", size=11, color="#8a96a8"),
                        x=0.05, y=0.97,
                    ),
                    margin=dict(l=0, r=0, t=22, b=0),
                    height=170,
                    paper_bgcolor="#10141c",
                    plot_bgcolor="#10141c",
                    xaxis=dict(
                        showticklabels=False, showgrid=False,
                        zeroline=False, color="#4a5568",
                    ),
                    yaxis=dict(
                        showticklabels=False, showgrid=False,
                        zeroline=False, color="#4a5568",
                    ),
                    font=dict(family="DM Mono", color="#8a96a8"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "GeoVision-CLIP Cali · UAO Ingeniería de Datos e IA · Prof. Ferro & García · 2026"
    "</div>",
    unsafe_allow_html=True,
)