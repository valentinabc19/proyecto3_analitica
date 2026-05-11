import os
import re
import json
import hashlib
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
import s3fs
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# ─── 1. CONFIGURACIÓN DE WASABI ────────────────────────────────────────────────
load_dotenv()
ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
REGION     = os.getenv('WASABI_REGION')
BUCKET     = os.getenv('WASABI_BUCKET')
ENDPOINT   = f'https://s3.{REGION}.wasabisys.com'

# Variables de entorno OBLIGATORIAS para que rioxarray pueda leer desde Wasabi
os.environ['AWS_ACCESS_KEY_ID'] = ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = SECRET_KEY
os.environ['AWS_S3_ENDPOINT'] = ENDPOINT.replace('https://', '')
os.environ['AWS_VIRTUAL_HOSTING'] = 'FALSE'
os.environ['AWS_HTTPS'] = 'YES'

# ─── 2. RUTAS ──────────────────────────────────────────────────────────────────
# Rutas en Wasabi (Asegúrate de que el prefijo de los TIFFs sea correcto)
RUTA_TIFS_S3 = f'{BUCKET}/GeoVision/Sentinel5P' 
RUTA_ZARR_S3 = f'{BUCKET}/GeoVision_Panel/sentinel5p.zarr'

# El manifest DEBE ser local para que puedas subirlo a GitHub (Pág 5 del PDF)
RUTA_MANIFEST_LOCAL = 'manifest_s5p.json'

# Inicializar sistema de archivos S3
fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={'endpoint_url': ENDPOINT})

# ─── PARSEAR NOMBRE DE ARCHIVO ─────────────────────────────────────────────────
def parsear_nombre(nombre):
    # Extrae el nombre del archivo de la ruta completa de S3
    nombre_archivo = nombre.split('/')[-1]
    match = re.match(r'S5P_(\w+)_(\d{4}-\d{2}-\d{2})\.tif', nombre_archivo)
    if not match:
        return None, None
    return match.group(1), datetime.strptime(match.group(2), '%Y-%m-%d')

# ─── CARGAR TIFs DESDE WASABI ──────────────────────────────────────────────────
def cargar_tifs_s3(ruta_base_s3):
    print(f"Explorando bucket: s3://{ruta_base_s3}")
    # Buscar recursivamente todos los .tif
    archivos = fs.glob(f'{ruta_base_s3}/*/*/*.tif')
    
    if not archivos:
        raise ValueError(f"No se encontraron archivos en s3://{ruta_base_s3}")
        
    print(f'Archivos encontrados en la nube: {len(archivos)}')

    por_contaminante = {'NO2': [], 'SO2': [], 'O3': []}

    # Leer archivos directamente desde Wasabi
    for ruta in tqdm(archivos, desc='Descargando y armando Xarray'):
        contaminante, fecha = parsear_nombre(ruta)
        if contaminante not in por_contaminante:
            continue

        # Leer desde S3 usando URI
        da = rioxarray.open_rasterio(f"s3://{ruta}", masked=True)
        da = da.squeeze('band', drop=True)
        da = da.rename({'x': 'lon', 'y': 'lat'})
        da = da.assign_coords(time=fecha).expand_dims('time')
        por_contaminante[contaminante].append(da)

    datasets = {}
    for cont, lista in por_contaminante.items():
        if lista:
            # Ordenar por tiempo para evitar errores cronológicos
            lista_ordenada = sorted(lista, key=lambda x: x.time.values[0])
            concatenado = xr.concat(lista_ordenada, dim='time')
            concatenado.name = cont
            datasets[cont] = concatenado
            print(f'{cont}: {len(lista)} fechas → shape {concatenado.shape}')

    ds = xr.Dataset(datasets)
    ds.attrs = {
        'fuente':      'Sentinel-5P TROPOMI L3 OFFL',
        'area':        'Cali metropolitana (-76.60, 3.30, -76.40, 3.55)',
        'resolucion':  '1000m',
        'unidades':    'mol/m²',
        'creado':      pd.Timestamp.now().isoformat(),
    }
    return ds

# ─── GUARDAR ZARR EN WASABI ────────────────────────────────────────────────────
def guardar_zarr_s3(ds, ruta_zarr):
    chunks = {
        'time': min(30, ds.sizes['time']),
        'lat':  ds.sizes['lat'],
        'lon':  ds.sizes['lon'],
    }

    print(f'\nSubiendo Zarr multidimensional a: s3://{ruta_zarr}')
    
    # Mapear el almacenamiento Zarr a Wasabi S3
    s3_store = s3fs.S3Map(root=ruta_zarr, s3=fs, check=False)
    
    ds.chunk(chunks).to_zarr(store=s3_store, mode='w', consolidated=True)

    # Calcular peso directamente en Wasabi
    tamanio_bytes = fs.du(ruta_zarr)
    tamanio_mb = tamanio_bytes / 1e6

    print(f'Tamaño en la nube: {tamanio_mb:.2f} MB')
    return tamanio_mb

# ─── EJECUCIÓN PRINCIPAL ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Conversión S5P GeoTIFF → Zarr (Wasabi Cloud) ===\n')

    # 1. Leer de Wasabi
    ds = cargar_tifs_s3(RUTA_TIFS_S3)

    print('\n=== Verificación de Integridad ===')
    for var in ds.data_vars:
        vals = ds[var].values
        nans = np.isnan(vals).sum() / vals.size * 100
        print(f'  {var}: min={np.nanmin(vals):.6f}  max={np.nanmax(vals):.6f}  NaN={nans:.1f}%')

    # 2. Escribir en Wasabi
    tamanio_mb = guardar_zarr_s3(ds, RUTA_ZARR_S3)

    print('\n Proceso completado con éxito en la nube')