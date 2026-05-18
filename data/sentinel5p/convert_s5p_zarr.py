import os
import re
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import s3fs
import zarr

from tqdm import tqdm
from dotenv import load_dotenv
from zarr.codecs import BloscCodec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# VARIABLES DE ENTORNO
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

ACCESS_KEY  = os.getenv('WASABI_ACCESS_KEY')
SECRET_KEY  = os.getenv('WASABI_SECRET_KEY')
REGION      = os.getenv('WASABI_REGION')
BUCKET      = os.getenv('WASABI_BUCKET')
DEST_BUCKET = os.getenv('WASABI_DEST_BUCKET')
ENDPOINT    = f'https://s3.{REGION}.wasabisys.com'

os.environ['AWS_ACCESS_KEY_ID']     = ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = SECRET_KEY
os.environ['AWS_S3_ENDPOINT']       = ENDPOINT.replace('https://', '')
os.environ['AWS_VIRTUAL_HOSTING']   = 'FALSE'
os.environ['AWS_HTTPS']             = 'YES'

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
RUTA_TIFS_S3 = f'{BUCKET}/GeoVision/Sentinel5P'

# Un Zarr por contaminante — cada uno tiene su propia dimensión time
# independiente porque NO2/SO2/O3 no tienen el mismo número de granulos.
RUTA_ZARR_S3 = {
    'NO2': f'{DEST_BUCKET}/GeoVision_Panel/sentinel5p_NO2.zarr',
    'SO2': f'{DEST_BUCKET}/GeoVision_Panel/sentinel5p_SO2.zarr',
    'O3':  f'{DEST_BUCKET}/GeoVision_Panel/sentinel5p_O3.zarr',
}

EXPORT_CRS     = 'EPSG:4326'
BBOX           = [-76.75, 3.20, -76.30, 3.75]
BAND_POLLUTANT = 1   # Band 1 → contaminante
BAND_CLOUD     = 2   # Band 2 → cloud_fraction
UNITS          = {'NO2': 'mol m-2', 'SO2': 'mol m-2', 'O3': 'mol m-2'}
CHUNKS         = {'time': 16, 'lat': 64, 'lon': 64}
BATCH_SIZE     = 200

# ─────────────────────────────────────────────────────────────────────────────
# S3 FILESYSTEM
# ─────────────────────────────────────────────────────────────────────────────
fs = s3fs.S3FileSystem(
    key=ACCESS_KEY,
    secret=SECRET_KEY,
    client_kwargs={'endpoint_url': ENDPOINT}
)

# ─────────────────────────────────────────────────────────────────────────────
# PARSEAR NOMBRE DE ARCHIVO
# Espera: S5P_NO2_2020-01-01_13-45-11.tif
# ─────────────────────────────────────────────────────────────────────────────
def parsear_nombre(ruta: str):
    nombre = ruta.split('/')[-1]
    match  = re.match(
        r'S5P_(\w+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.tif',
        nombre
    )
    if not match:
        return None, None, None
    contaminante = match.group(1)
    fecha        = datetime.strptime(match.group(2), '%Y-%m-%d_%H-%M-%S')
    orbit_id     = match.group(2)
    return contaminante, fecha, orbit_id

# ─────────────────────────────────────────────────────────────────────────────
# VALIDAR GRID ESPACIAL
# ─────────────────────────────────────────────────────────────────────────────
def validar_grid(da, referencia: dict) -> bool:
    if referencia is None:
        return True
    return (
        da.shape[-2:] == referencia['shape'] and
        str(da.rio.crs) == referencia['crs']
    )

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAR UN TIFF
# Retorna un xr.Dataset con:
#   - variables: {contaminante}, {contaminante}_cloud_fraction
#   - coordenadas extra en time: orbit_id, valid_pixel_fraction,
#     cf_mean, cf_p25, cf_p75
# ─────────────────────────────────────────────────────────────────────────────
def procesar_tiff(ruta: str, referencia_grid: dict, timestamps_vistos: set):

    contaminante, fecha, orbit_id = parsear_nombre(ruta)
    if contaminante is None:
        return None, referencia_grid

    try:
        da = rioxarray.open_rasterio(
            f's3://{ruta}',
            masked=True,
            chunks={'band': -1, 'x': CHUNKS['lon'], 'y': CHUNKS['lat']}
        )

        # ── Validaciones básicas ──────────────────────────────────────────────
        if da.sizes.get('band', 0) < 2:
            print(f'⚠️  Solo {da.sizes.get("band", 0)} banda(s): {ruta}')
            return None, referencia_grid

        if da.rio.crs is None:
            print(f'⚠️  Sin CRS: {ruta}')
            return None, referencia_grid

        if not validar_grid(da, referencia_grid):
            print(f'⚠️  Grid inconsistente: {ruta}')
            return None, referencia_grid

        # ── Fijar grid de referencia en el primer TIFF válido ─────────────────
        if referencia_grid is None:
            referencia_grid = {
                'shape': da.shape[-2:],
                'crs': str(da.rio.crs)
            }
            print(f'\n✅ Grid de referencia fijado:')
            print(f'   Shape : {referencia_grid["shape"]}')
            print(f'   CRS   : {referencia_grid["crs"]}\n')

        # ── Extraer bandas ────────────────────────────────────────────────────
        poll_raw  = da.sel(band=BAND_POLLUTANT).drop_vars('band')
        cloud_raw = da.sel(band=BAND_CLOUD).drop_vars('band')

        poll_raw  = poll_raw.rename({'x': 'lon', 'y': 'lat'})
        cloud_raw = cloud_raw.rename({'x': 'lon', 'y': 'lat'})

        poll_raw  = poll_raw.astype(np.float32)
        cloud_raw = cloud_raw.astype(np.float32)

        # ── Estadísticas de calidad sobre cloud_fraction ──────────────────────
        cf_vals  = cloud_raw.values.ravel()
        cf_valid = cf_vals[~np.isnan(cf_vals)]

        if cf_valid.size == 0:
            print(f'⚠️  cloud_fraction todo NaN: {ruta}')
            return None, referencia_grid

        total_px   = cf_vals.size
        valid_frac = np.float32(cf_valid.size / total_px)
        cf_mean    = np.float32(cf_valid.mean())
        cf_p25     = np.float32(np.percentile(cf_valid, 25))
        cf_p75     = np.float32(np.percentile(cf_valid, 75))

        # ── Resolver timestamps duplicados ────────────────────────────────────
        t = np.datetime64(fecha, 'us')
        while t in timestamps_vistos:
            t += np.timedelta64(1, 'us')
        timestamps_vistos.add(t)

        # ── Asignar dimensión time ────────────────────────────────────────────
        poll_da  = poll_raw.assign_coords(time=t).expand_dims('time')
        cloud_da = cloud_raw.assign_coords(time=t).expand_dims('time')

        # ── Atributos ─────────────────────────────────────────────────────────
        poll_da.attrs = {
            'long_name':  contaminante,
            'units':      UNITS.get(contaminante, 'mol m-2'),
            'source':     'Sentinel-5P TROPOMI L3 OFFL',
            'crs':        str(da.rio.crs),
            'band_index': BAND_POLLUTANT,
        }
        cf_var_name = f'{contaminante}_cloud_fraction'
        cloud_da.attrs = {
            'long_name':   f'{contaminante} cloud fraction (QA proxy)',
            'units':       'fraction [0-1]',
            'valid_range': [0.0, 1.0],
            'source':      'Sentinel-5P TROPOMI L3 OFFL',
            'band_index':  BAND_CLOUD,
            'note': (
                f'Cloud fraction specific to the {contaminante} retrieval algorithm. '
                'Used as QA proxy (qa_value not available at L3 level in GEE). '
                'Kept separate per contaminant — may differ across products '
                'due to different UV-Vis retrieval wavelengths. '
                'Recommended threshold for GEOClip: cf < 0.5'
            ),
        }

        # ── Dataset con coordenadas de calidad escalares en time ──────────────
        ds = xr.Dataset(
            {
                contaminante: poll_da,
                cf_var_name:  cloud_da,
            },
            coords={
                'orbit_id': ('time', [orbit_id]),
                'valid_pixel_fraction': (
                    'time', [valid_frac],
                    {'long_name': 'Fraction of non-NaN pixels over bbox',
                     'units': 'fraction [0-1]'}
                ),
                'cf_mean': (
                    'time', [cf_mean],
                    {'long_name': 'Mean cloud fraction over bbox',
                     'units': 'fraction [0-1]'}
                ),
                'cf_p25': (
                    'time', [cf_p25],
                    {'long_name': '25th percentile cloud fraction',
                     'units': 'fraction [0-1]'}
                ),
                'cf_p75': (
                    'time', [cf_p75],
                    {'long_name': '75th percentile cloud fraction',
                     'units': 'fraction [0-1]'}
                ),
            }
        )

        return ds, referencia_grid

    except Exception as e:
        print(f'❌ Error en {ruta}: {e}')
        return None, referencia_grid

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUIR ENCODING
# Se llama una sola vez por contaminante al conocer las variables del primer lote.
# ─────────────────────────────────────────────────────────────────────────────
def construir_encoding(ds_batch: xr.Dataset, compressor) -> dict:
    encoding = {
        var: {
            'compressor': compressor,
            'dtype': 'float32',
        }
        for var in ds_batch.data_vars
    }
    # Epoch fija en microsegundos — evita overflow al hacer append
    # cuando el rango temporal es grande (ej. 2020–2024).
    encoding['time'] = {
        'units':    'microseconds since 1970-01-01',
        'calendar': 'proleptic_gregorian',
        'dtype':    'int64',
    }
    return encoding

# ─────────────────────────────────────────────────────────────────────────────
# ESCRITURA INCREMENTAL POR LOTES
# ─────────────────────────────────────────────────────────────────────────────
def guardar_lote(ds_lote: xr.Dataset, store, primer_lote: bool, encoding: dict):
    # Ordenar y rechunkear para alinear Dask con Zarr antes de escribir.
    ds_lote = ds_lote.sortby('time').chunk({
        'time': CHUNKS['time'],
        'lat':  CHUNKS['lat'],
        'lon':  CHUNKS['lon'],
    })

    if primer_lote:
        ds_lote.to_zarr(
            store=store,
            mode='w',
            consolidated=False,
            encoding=encoding,
            safe_chunks=False,
        )
    else:
        ds_lote.to_zarr(
            store=store,
            mode='a',
            append_dim='time',
            consolidated=False,
            safe_chunks=False,
        )

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAR UN CONTAMINANTE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────
def procesar_contaminante(contaminante: str, archivos: list, compressor):

    print(f'\n{"="*50}')
    print(f'  {contaminante} — {len(archivos)} archivos')
    print(f'{"="*50}')

    store = s3fs.S3Map(
        root=RUTA_ZARR_S3[contaminante],
        s3=fs,
        check=False
    )

    encoding         = None
    primer_lote      = True
    referencia_grid  = None
    timestamps_vistos = set()
    lote             = []
    total_ok         = 0
    total_skip       = 0

    for ruta in tqdm(archivos, desc=contaminante, unit='archivo'):

        ds_granulo, referencia_grid = procesar_tiff(
            ruta, referencia_grid, timestamps_vistos
        )

        if ds_granulo is None:
            total_skip += 1
            continue

        lote.append(ds_granulo)
        total_ok += 1

        if len(lote) >= BATCH_SIZE:
            ds_batch = xr.concat(lote, dim='time')

            if encoding is None:
                encoding = construir_encoding(ds_batch, compressor)

            guardar_lote(ds_batch, store, primer_lote, encoding)
            primer_lote = False
            lote = []
            print(f'   💾 Lote guardado — acumulados: {total_ok}')

    # ── Último lote (residuo) ─────────────────────────────────────────────────
    if lote:
        ds_batch = xr.concat(lote, dim='time')
        if encoding is None:
            encoding = construir_encoding(ds_batch, compressor)
        guardar_lote(ds_batch, store, primer_lote, encoding)
        print(f'   💾 Lote final guardado — acumulados: {total_ok}')

    # ── Consolidar metadata ───────────────────────────────────────────────────
    print(f'\n🔗 Consolidando metadata {contaminante}...')
    zarr.consolidate_metadata(store)

    # ── Metadata global del cubo ──────────────────────────────────────────────
    ds_final = xr.open_zarr(store, consolidated=True)
    ds_final.attrs.update({
        'title':             f'Sentinel-5P {contaminante} — Cali bbox',
        'source':            'Sentinel-5P TROPOMI L3 OFFL via Google Earth Engine',
        'institution':       'ESA / Copernicus',
        'contaminant':       contaminante,
        'crs':               EXPORT_CRS,
        'bounding_box':      str(BBOX),
        'scale_meters':      7000,
        'created':           pd.Timestamp.now().isoformat(),
        'processing_level':  'L3 RAW (no QA filter applied)',
        'qa_proxy':          (
            f'{contaminante}_cloud_fraction '
            '(recommended threshold < 0.5 for GEOClip)'
        ),
        'cloud_fraction_note': (
            'Kept separate per contaminant — may differ across products '
            'due to different UV-Vis retrieval wavelengths'
        ),
        'orbit_id_note':     'orbit_id coordinate allows granule-level filtering in EDA',
        'bands_per_tiff':    'band1=pollutant, band2=cloud_fraction',
    })

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f'\n--- {contaminante} ---')
    print(ds_final)
    size_gb = fs.du(RUTA_ZARR_S3[contaminante]) / 1e9
    print(f'✅ Granulos procesados : {total_ok}')
    print(f'⚠️  Granulos omitidos   : {total_skip}')
    print(f'📦 Tamaño Zarr         : {size_gb:.2f} GB')

    return total_ok, total_skip

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print('\n======================================')
    print('SENTINEL-5P → ZARR PIPELINE')
    print('======================================\n')

    # ── Listar y agrupar TIFFs por contaminante ───────────────────────────────
    print('🔎 Buscando GeoTIFFs en Wasabi...')
    todos = fs.glob(f'{RUTA_TIFS_S3}/*/*/*.tif')
    if not todos:
        raise ValueError('No se encontraron TIFFs en la ruta especificada.')
    print(f'✅ TIFFs encontrados: {len(todos)}\n')

    archivos_por_cont = defaultdict(list)
    for ruta in todos:
        cont, _, _ = parsear_nombre(ruta)
        if cont in RUTA_ZARR_S3:
            archivos_por_cont[cont].append(ruta)
        else:
            print(f'⚠️  Contaminante desconocido, ignorado: {ruta}')

    for cont, n in archivos_por_cont.items():
        print(f'   {cont}: {len(n)} archivos')

    # ── Compresión ────────────────────────────────────────────────────────────
    compressor = BloscCodec(cname='zstd', clevel=5, shuffle='bitshuffle')

    # ── Procesar cada contaminante de forma independiente ─────────────────────
    resumen = {}
    for contaminante in RUTA_ZARR_S3.keys():
        if contaminante not in archivos_por_cont:
            print(f'\n⚠️  Sin archivos para {contaminante}, se omite.')
            continue
        ok, skip = procesar_contaminante(
            contaminante,
            archivos_por_cont[contaminante],
            compressor
        )
        resumen[contaminante] = {'ok': ok, 'skip': skip}

    # ── Resumen global ────────────────────────────────────────────────────────
    print('\n======================================')
    print('PIPELINE FINALIZADO')
    print('======================================')
    total_ok_global   = sum(v['ok']   for v in resumen.values())
    total_skip_global = sum(v['skip'] for v in resumen.values())
    for cont, vals in resumen.items():
        print(f'  {cont}: {vals["ok"]} ok  |  {vals["skip"]} omitidos')
    print(f'  ─────────────────────────────')
    print(f'  Total procesados : {total_ok_global}')
    print(f'  Total omitidos   : {total_skip_global}')
    print('======================================')