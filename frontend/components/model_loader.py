"""
components/model_loader.py
Carga de checkpoints GeoVision-CLIP y ConvLSTM desde HuggingFace Hub.
Los modelos se cachean en sesión de Streamlit para evitar recargas.
"""

from __future__ import annotations

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

# ── Configuración HuggingFace ────────────────────────────────────────────────
HF_REPO_ID   = os.getenv('HF_REPO_ID')
HF_TOKEN     = os.getenv('HF_TOKEN')          
CACHE_DIR    = Path(os.getenv("MODEL_CACHE_DIR", "./model_cache"))

CHECKPOINT_FILES = {
    "clip_sae":   "geovision_clip.pt",
    "convlstm":   "convlstm_model.pt",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    """Calcula hash MD5 de un archivo local."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_download(filename: str) -> Path:
    """
    Descarga un archivo del repositorio HuggingFace si no está en caché.
    Requiere: pip install huggingface_hub
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub no instalado. Ejecuta: pip install huggingface_hub"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CACHE_DIR / filename

    if local_path.exists():
        logger.info("Checkpoint en caché local: %s", local_path)
        return local_path

    logger.info("Descargando %s desde HF Hub (%s)...", filename, HF_REPO_ID)
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=HF_TOKEN or None,
        local_dir=CACHE_DIR,
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


# ── Carga de modelos con caché de sesión ─────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_clip_sae_model():
    """
    Carga el modelo GeoVision-CLIP + Sparse Autoencoders.
    Retorna (model, md5_hash) o (None, None) si no están disponibles los pesos.
    """
    import torch  # type: ignore

    try:
        ckpt_path = _hf_download(CHECKPOINT_FILES["clip_sae"])
        md5 = _md5(ckpt_path)

        # ── Importar arquitectura desde el módulo de modelos ────────────────
        # Ajusta el import según la ubicación real de tu definición de modelo
        try:
            from components.architectures import GeoVisionCLIP  # type: ignore
            model = GeoVisionCLIP()
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            logger.info("CLIP+SAE cargado. MD5: %s", md5)
            return model, md5
        except ImportError:
            # Fallback: retorna solo la metadata (modo demo)
            logger.warning(
                "Arquitectura GeoVisionCLIP no encontrada. "
                "Modo demo activo — se retornan embeddings sintéticos."
            )
            return None, md5

    except Exception as exc:
        logger.error("Error cargando CLIP+SAE: %s", exc)
        return None, None


@st.cache_resource(show_spinner=False)
def load_convlstm_model():
    """
    Carga el modelo ConvLSTM espacio-temporal.
    Retorna (model, md5_hash) o (None, None) si no están disponibles.
    """
    import torch  # type: ignore

    try:
        ckpt_path = _hf_download(CHECKPOINT_FILES["convlstm"])
        md5 = _md5(ckpt_path)

        try:
            from components.architectures import ConvLSTMForecaster  # type: ignore
            model = ConvLSTMForecaster()
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            logger.info("ConvLSTM cargado. MD5: %s", md5)
            return model, md5
        except ImportError:
            logger.warning("ConvLSTMForecaster no encontrado. Modo demo activo.")
            return None, md5

    except Exception as exc:
        logger.error("Error cargando ConvLSTM: %s", exc)
        return None, None


def get_model_status() -> dict:
    """
    Retorna un dict con el estado de cada checkpoint para mostrarlo en la UI.
    No carga los modelos completos, solo verifica presencia y MD5.
    """
    status = {}
    for key, filename in CHECKPOINT_FILES.items():
        local = CACHE_DIR / filename
        if local.exists():
            status[key] = {"available": True, "md5": _md5(local), "path": str(local)}
        else:
            status[key] = {"available": False, "md5": None, "path": None}
    return status


def sidebar_model_status():
    """
    Muestra en la sidebar el estado de los checkpoints.
    Llamar desde cada página que use modelos.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
        'text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;">'
        "Estado de modelos</p>",
        unsafe_allow_html=True,
    )

    status = get_model_status()
    labels = {
        "clip_sae":  "CLIP + SAE",
        "convlstm":  "ConvLSTM",
        "tokenizer": "Tokenizer",
    }
    for key, info in status.items():
        icon  = "●" if info["available"] else "○"
        color = "#4af0b0" if info["available"] else "#4a5568"
        md5_short = info["md5"][:8] + "..." if info["md5"] else "—"
        st.sidebar.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;padding:0.3rem 0;">'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
            f'color:{color};">{icon} {labels.get(key, key)}</span>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
            f'color:#4a5568;">{md5_short}</span></div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(
        f'<p style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
        f'color:#4a5568;margin-top:0.3rem;">HF: {HF_REPO_ID}</p>',
        unsafe_allow_html=True,
    )