import os
import re
import json
import hashlib
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ── Rutas ──────────────────────────────────────────────────────
RUTA_TIFS = r'C:\Users\valen\Desktop\proyecto3_analitica\data\raw\sentinel5p'
RUTA_ZARR = r'C:\Users\valen\Desktop\proyecto3_analitica\data\processed\zarr\sentinel5p.zarr'
RUTA_MANIFEST = r'C:\Users\valen\Desktop\proyecto3_analitica\data\manifest_local.json'

# ── Parsear nombre de archivo ──────────────────────────────────
def parsear_nombre(nombre):
    match = re.match(r'S5P_(\w+)_(\d{4}-\d{2}-\d{2})\.tif', nombre)
    if not match:
        return None, None
    return match.group(1), datetime.strptime(match.group(2), '%Y-%m-%d')

# ── Cargar TIFs ────────────────────────────────────────────────
def cargar_tifs(ruta_tifs):
    archivos = sorted(Path(ruta_tifs).glob('*.tif'))
    print(f'Archivos encontrados: {len(archivos)}')

    por_contaminante = {'NO2': [], 'SO2': [], 'O3': []}

    for archivo in tqdm(archivos, desc='Leyendo GeoTIFFs'):
        contaminante, fecha = parsear_nombre(archivo.name)
        if contaminante not in por_contaminante:
            print(f'  ⚠ Ignorado: {archivo.name}')
            continue

        da = rioxarray.open_rasterio(archivo, masked=True)
        da = da.squeeze('band', drop=True)
        da = da.rename({'x': 'lon', 'y': 'lat'})
        da = da.assign_coords(time=fecha).expand_dims('time')
        por_contaminante[contaminante].append(da)

    datasets = {}
    for cont, lista in por_contaminante.items():
        if lista:
            concatenado = xr.concat(lista, dim='time')
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

# ── Guardar Zarr ───────────────────────────────────────────────
def guardar_zarr(ds, ruta_zarr):
    os.makedirs(os.path.dirname(ruta_zarr), exist_ok=True)

    chunks = {
        'time': min(30, ds.sizes['time']),
        'lat':  ds.sizes['lat'],
        'lon':  ds.sizes['lon'],
    }

    print(f'\nGuardando Zarr en: {ruta_zarr}')
    ds.chunk(chunks).to_zarr(ruta_zarr, mode='w', consolidated=True)

    tamanio_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(ruta_zarr)
        for f in files
    ) / 1e6

    print(f'Tamaño en disco: {tamanio_mb:.2f} MB')
    return tamanio_mb

# ── Actualizar manifest ────────────────────────────────────────
def actualizar_manifest(ds, ruta_zarr, tamanio_mb, ruta_manifest):
    # Zarr v3 usa zarr.json, v2 usaba .zmetadata
    for nombre_meta in ['zarr.json', '.zmetadata']:
        ruta_meta = os.path.join(ruta_zarr, nombre_meta)
        if os.path.exists(ruta_meta):
            md5 = hashlib.md5(open(ruta_meta, 'rb').read()).hexdigest()
            break
    else:
        # Si no encuentra ninguno, calcular MD5 de todo el directorio
        h = hashlib.md5()
        for dp, _, files in sorted(os.walk(ruta_zarr)):
            for f in sorted(files):
                h.update(open(os.path.join(dp, f), 'rb').read())
        md5 = h.hexdigest()

    entrada = {
        'archivo':       'sentinel5p.zarr',
        'ruta':          ruta_zarr,
        'fuente':        'Sentinel-5P TROPOMI L3 OFFL',
        'variables':     list(ds.data_vars),
        'periodo':       f'{str(ds.time.values[0])[:10]} / {str(ds.time.values[-1])[:10]}',
        'dims':          {k: int(v) for k, v in ds.sizes.items()},
        'tamanio_mb':    round(tamanio_mb, 2),
        'md5':           md5,
        'etapa':         'zarr',
        'creado':        pd.Timestamp.now().isoformat(),
    }

    manifest = []
    if os.path.exists(ruta_manifest):
        with open(ruta_manifest) as f:
            manifest = json.load(f)

    manifest = [m for m in manifest if m.get('archivo') != 'sentinel5p.zarr']
    manifest.append(entrada)

    with open(ruta_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nManifest actualizado: {ruta_manifest}')
    print(f'MD5: {md5}')

# ── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Conversión S5P GeoTIFF → Zarr ===\n')

    ds = cargar_tifs(RUTA_TIFS)

    print('\n=== Verificación ===')
    for var in ds.data_vars:
        vals = ds[var].values
        nans = np.isnan(vals).sum() / vals.size * 100
        print(f'  {var}: min={np.nanmin(vals):.6f}  max={np.nanmax(vals):.6f}  NaN={nans:.1f}%')

    tamanio_mb = guardar_zarr(ds, RUTA_ZARR)
    actualizar_manifest(ds, RUTA_ZARR, tamanio_mb, RUTA_MANIFEST)

    print('\n✓ Proceso completado')