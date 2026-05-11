import os
import json
import pandas as pd
from pathlib import Path
import s3fs
import xarray as xr
from dask.distributed import Client
from dotenv import load_dotenv

# ─── CARGAR VARIABLES DE ENTORNO ─────────────────────────────────────────────
load_dotenv()
WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
WASABI_REGION     = os.getenv('WASABI_REGION')
WASABI_BUCKET     = os.getenv('WASABI_BUCKET')
ENDPOINT          = f'https://s3.{WASABI_REGION}.wasabisys.com'

# Inyectar credenciales al entorno para que las librerías C/C++ subyacentes tengan acceso
os.environ['AWS_ACCESS_KEY_ID'] = WASABI_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = WASABI_SECRET_KEY
os.environ['AWS_S3_ENDPOINT'] = ENDPOINT.replace('https://', '')
os.environ['AWS_VIRTUAL_HOSTING'] = 'FALSE'
os.environ['AWS_HTTPS'] = 'YES'

# ─── RUTAS ───────────────────────────────────────────────────────────────────
RUTA_ORIGEN_NC = f'{WASABI_BUCKET}/GeoVision/ERA5'
RUTA_DESTINO_ZARR = f'{WASABI_BUCKET}/GeoVision_Panel/ERA5.zarr'
RUTA_MANIFEST_LOCAL = 'manifest_era5.json'
BBOX_CALI = [3.75, -76.75, 3.20, -76.30] # [N, W, S, E]

def main():
    # 1. Iniciar Dask 
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
    print("Dashboard de Dask disponible en:", client.dashboard_link)

    # 2. Conectarse a Wasabi
    fs = s3fs.S3FileSystem(
        key=WASABI_ACCESS_KEY,
        secret=WASABI_SECRET_KEY,
        client_kwargs={'endpoint_url': ENDPOINT}
    )

    print("\nBuscando archivos NetCDF en Wasabi...")
    archivos_nc = fs.glob(f'{RUTA_ORIGEN_NC}/*.nc')
    
    if not archivos_nc:
        raise ValueError("No se encontraron archivos .nc. Verifica la ruta origen.")
        
    rutas_s3 = ['s3://' + archivo for archivo in archivos_nc]
    print(f"Se encontraron {len(rutas_s3)} archivos. Generando Manifest...")

    # ─── 4. ABRIR DATASET MASIVO ─────────────────────────────────────────────
    print("\nAbriendo cubo multidimensional distribuido...")
    
    ds_masivo = xr.open_mfdataset(
        rutas_s3,
        engine='h5netcdf', 
        combine='by_coords', 
        parallel=True,       # ¡Clave! Obliga a Dask a abrir los metadatos en paralelo
        storage_options={'key': WASABI_ACCESS_KEY, 'secret': WASABI_SECRET_KEY, 'client_kwargs': {'endpoint_url': ENDPOINT}}
    )

    # ─── 5. CHUNKING ESPACIO-TEMPORAL ────────────────────────────────────────
    # time: 720 horas = ~30 días (1 mes por bloque)
    # lat/lon: -1 = Todo el espacio en un solo bloque (ideal para áreas pequeñas como Cali)
    ds_chunked = ds_masivo.chunk({
        'time': 720,       
        'latitude': -1,    
        'longitude': -1
    })

    print("\nDataset preparado con la siguiente estructura:")
    print(ds_chunked)

    # ─── 6. GUARDAR ZARR EN WASABI ───────────────────────────────────────────
    print(f"\nIniciando conversión y subida a Zarr en: s3://{RUTA_DESTINO_ZARR}")
    print("¡Revisa el panel de Dask para ver el progreso!")

    store = s3fs.S3Map(root=RUTA_DESTINO_ZARR, s3=fs, check=False)

    ds_chunked.to_zarr(
        store=store,
        consolidated=True, 
        mode='w'           
    )

    print(f"\n Panel Zarr de ERA5 guardado exitosamente en la nube.")

if __name__ == '__main__':
    main()