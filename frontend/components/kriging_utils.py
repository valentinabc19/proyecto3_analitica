"""
components/kriging_utils.py
Wrappers de estadística geoespacial para la UI de GeoVision-CLIP Cali.
Envuelve PyKrige (OrdinaryKriging3D) y esda (Moran) para su uso en Streamlit.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ── ST-Kriging ────────────────────────────────────────────────────────────────

def run_st_kriging(
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    query_lats: np.ndarray,
    query_lons: np.ndarray,
    query_t: float,
    variogram_model: str = "exponential",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta Kriging Ordinario 3D (lat, lon, t) sobre los valores de entrada.

    Parameters
    ----------
    lats, lons, times : arrays 1-D de coordenadas de los puntos de entrenamiento.
    values            : array 1-D de concentraciones en esos puntos.
    query_lats, query_lons : grilla 1-D de puntos a predecir.
    query_t           : tiempo de consulta (normalizado igual que times).
    variogram_model   : "exponential" | "gaussian" | "spherical"

    Returns
    -------
    z    : array 1-D de predicciones en los puntos de consulta.
    var  : array 1-D de varianzas Kriging.
    """
    try:
        from pykrige.ok3d import OrdinaryKriging3D  # type: ignore
    except ImportError:
        raise ImportError("pykrige no instalado. Ejecuta: pip install pykrige")

    # Normalización (evita anisotropía espuria)
    def _norm(arr: np.ndarray) -> np.ndarray:
        std = arr.std()
        return (arr - arr.mean()) / (std if std > 1e-9 else 1.0)

    lat_n  = _norm(lats)
    lon_n  = _norm(lons)
    t_n    = _norm(times)

    lat_mean, lat_std = lats.mean(), (lats.std() or 1.0)
    lon_mean, lon_std = lons.mean(), (lons.std() or 1.0)
    t_mean,   t_std   = times.mean(), (times.std() or 1.0)

    ok = OrdinaryKriging3D(
        lat_n, lon_n, t_n, values,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
    )

    ql_n = (query_lats - lat_mean) / lat_std
    qln_n = (query_lons - lon_mean) / lon_std
    qt_n  = np.full_like(ql_n, (query_t - t_mean) / t_std)

    z, var = ok.execute("points", ql_n, qln_n, qt_n)
    return np.asarray(z), np.asarray(var)


def variogram_params(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    model: str = "exponential",
) -> dict:
    """
    Ajusta un variograma experimental 2D y retorna nugget, sill y range.
    Usa PyKrige OrdinaryKriging (2D) como proxy para inspección rápida.
    """
    try:
        from pykrige.ok import OrdinaryKriging  # type: ignore
    except ImportError:
        return {"nugget": None, "sill": None, "range": None, "model": model}

    ok = OrdinaryKriging(
        lons, lats, values,
        variogram_model=model,
        verbose=False,
        enable_plotting=False,
    )
    p = ok.variogram_model_parameters
    # PyKrige retorna [psill, range, nugget] para modelos estándar
    if len(p) == 3:
        return {
            "nugget": float(p[2]),
            "sill":   float(p[0] + p[2]),
            "range":  float(p[1]),
            "model":  model,
        }
    return {"nugget": None, "sill": None, "range": None, "model": model}


# ── Moran Global ─────────────────────────────────────────────────────────────

def compute_moran(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    k: int = 5,
    permutations: int = 999,
) -> dict:
    """
    Calcula el Índice de Moran Global I sobre un conjunto de puntos irregulares.
    Requiere: pip install esda libpysal

    Returns
    -------
    dict con keys: I, EI, p_value, z_score, significativo
    """
    try:
        import libpysal  # type: ignore
        from esda.moran import Moran  # type: ignore
    except ImportError:
        raise ImportError(
            "esda / libpysal no instalados. "
            "Ejecuta: pip install esda libpysal"
        )

    coords = np.column_stack([lons, lats])
    w = libpysal.weights.KNN.from_array(coords, k=k)
    w.transform = "r"

    mi = Moran(values, w, permutations=permutations)
    return {
        "I":           float(mi.I),
        "EI":          float(mi.EI),
        "p_value":     float(mi.p_sim),
        "z_score":     float(mi.z_sim),
        "significativo": bool(mi.p_sim < 0.05),
    }


def compute_lisa(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    k: int = 5,
    permutations: int = 999,
    significance: float = 0.05,
) -> dict:
    """
    Calcula LISA (Local Indicators of Spatial Association).
    Retorna clasificación por punto: HH, LL, HL, LH o NS (no significativo).
    """
    try:
        import libpysal  # type: ignore
        from esda.moran import Moran_Local  # type: ignore
    except ImportError:
        raise ImportError("esda / libpysal no instalados.")

    coords = np.column_stack([lons, lats])
    w = libpysal.weights.KNN.from_array(coords, k=k)
    w.transform = "r"

    ml = Moran_Local(values, w, permutations=permutations)
    sig = ml.p_sim < significance

    quad_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    labels = np.where(sig, [quad_labels.get(q, "NS") for q in ml.q], "NS")

    return {
        "Is":     ml.Is.tolist(),
        "p_sim":  ml.p_sim.tolist(),
        "labels": labels.tolist(),
        "lats":   lats.tolist(),
        "lons":   lons.tolist(),
    }


# ── Residuos y nugget puro ────────────────────────────────────────────────────

def check_nugget_pure(
    residuals: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    model: str = "exponential",
    nugget_ratio_threshold: float = 0.90,
) -> dict:
    """
    Verifica si el variograma de residuos es nugget puro.
    Un variograma es "nugget puro" si nugget / sill > threshold (~0.90).

    Returns
    -------
    dict con is_nugget_pure, nugget_ratio, params
    """
    params = variogram_params(lats, lons, residuals, model)
    if params["sill"] and params["sill"] > 0:
        ratio = params["nugget"] / params["sill"]
        return {
            "is_nugget_pure":   ratio >= nugget_ratio_threshold,
            "nugget_ratio":     ratio,
            "params":           params,
        }
    return {"is_nugget_pure": False, "nugget_ratio": None, "params": params}