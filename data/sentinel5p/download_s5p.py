import os
#import io
import json
import hashlib
import requests
#import rasterio
import boto3
import ee
import dask

from rasterio.io import MemoryFile
from dask.distributed import Client
from datetime import date, timedelta
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────────────────────────────────────
# VARIABLES DE ENTORNO
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
WASABI_REGION     = os.getenv('WASABI_REGION')
WASABI_BUCKET     = os.getenv('WASABI_BUCKET')
GEE_PROJECT       = os.getenv('GEE_PROJECT_ID')

# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN GEE
# ─────────────────────────────────────────────────────────────────────────────
ee.Initialize(project=GEE_PROJECT)

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────
CALI_BBOX = ee.Geometry.Rectangle(
    [-76.75, 3.20, -76.30, 3.75]
)

# Resolución coherente con Sentinel-5P (~5.5-7 km)
SCALE = 7000

EXPORT_CRS = 'EPSG:4326'

CONTAMINANTES = {
    'NO2': {
        'collection': 'COPERNICUS/S5P/OFFL/L3_NO2',
        'band': 'tropospheric_NO2_column_number_density',
        'qa': 0.75
    },
    'SO2': {
        'collection': 'COPERNICUS/S5P/OFFL/L3_SO2',
        'band': 'SO2_column_number_density',
        'qa': 0.50
    },
    'O3': {
        'collection': 'COPERNICUS/S5P/OFFL/L3_O3',
        'band': 'O3_column_number_density',
        'qa': 0.70
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# SESIÓN HTTP ROBUSTA
# ─────────────────────────────────────────────────────────────────────────────
session = requests.Session()

retries = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504]
)

session.mount('https://', HTTPAdapter(max_retries=retries))

# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
manifest_entries = []

# ─────────────────────────────────────────────────────────────────────────────
# HASH MD5
# ─────────────────────────────────────────────────────────────────────────────
def calcular_md5(binary_data):
    return hashlib.md5(binary_data).hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────────────────────────────────────
def procesar_s5p(fecha_str, contaminante):

    try:

        s3_client = boto3.client(
            's3',
            endpoint_url=f'https://s3.{WASABI_REGION}.wasabisys.com',
            aws_access_key_id=WASABI_ACCESS_KEY,
            aws_secret_access_key=WASABI_SECRET_KEY
        )

        start = ee.Date(fecha_str)
        end   = start.advance(1, 'day')

        config = CONTAMINANTES[contaminante]

        # ---------------------------------------------------------------------
        # FILTRADO
        # ---------------------------------------------------------------------
        collection = (
            ee.ImageCollection(config['collection'])
            .filterBounds(CALI_BBOX)
            .filterDate(start, end)
        )

        size = collection.size().getInfo()

        if size == 0:
            return f"⚠️ [{fecha_str} | {contaminante}] Sin órbitas."

        image_list = collection.toList(size)

        resultados_locales = []

        # ---------------------------------------------------------------------
        # DESCARGAR CADA ÓRBITA INDIVIDUAL
        # ---------------------------------------------------------------------
        for i in range(size):

            image = ee.Image(image_list.get(i))

            # QA FILTER
            image = image.updateMask(
                image.select('cloud_fraction').gte(config['qa'])
            )

            # Selección banda
            image = image.select(config['band'])

            # Validar que aún tenga pixeles válidos
            stats = image.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=CALI_BBOX,
                scale=SCALE,
                maxPixels=1e9
            ).getInfo()

            pixel_count = list(stats.values())[0]

            if pixel_count is None or pixel_count == 0:
                continue

            acquisition_time = (
                ee.Date(image.get('system:time_start'))
                .format('YYYY-MM-dd_HH-mm-ss')
                .getInfo()
            )

            año = fecha_str.split('-')[0]

            nombre_archivo = (
                f'S5P_{contaminante}_{acquisition_time}.tif'
            )

            key = (
                f'GeoVision/Sentinel5Pv2/'
                f'{contaminante}/{año}/{nombre_archivo}'
            )

            # -----------------------------------------------------------------
            # URL DESCARGA
            # -----------------------------------------------------------------
            url = image.clip(CALI_BBOX).getDownloadURL({
                'scale': SCALE,
                'crs': EXPORT_CRS,
                'region': CALI_BBOX,
                'format': 'GEO_TIFF'
            })

            response = session.get(url, timeout=300)

            if response.status_code != 200:
                continue

            binary_data = response.content

            # -----------------------------------------------------------------
            # VALIDACIÓN TIFF
            # -----------------------------------------------------------------
            with MemoryFile(binary_data) as memfile:

                with memfile.open() as dataset:

                    width = dataset.width
                    height = dataset.height
                    bounds = dataset.bounds

                    if width <= 0 or height <= 0:
                        continue

            # -----------------------------------------------------------------
            # MD5
            # -----------------------------------------------------------------
            md5_hash = calcular_md5(binary_data)

            # -----------------------------------------------------------------
            # SUBIR A WASABI
            # -----------------------------------------------------------------
            s3_client.put_object(
                Bucket=WASABI_BUCKET,
                Key=key,
                Body=binary_data,
                ContentType='image/tiff'
            )

            # -----------------------------------------------------------------
            # MANIFEST ENTRY
            # -----------------------------------------------------------------
            manifest_entry = {
                "file_path": key,
                "md5": md5_hash,
                "dimensions": [width, height],
                "acquisition_date": acquisition_time,
                "source": config['collection'],
                "bounding_box": [
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top
                ],
                "pollutant": contaminante,
                "scale_meters": SCALE,
                "crs": EXPORT_CRS
            }

            resultados_locales.append(manifest_entry)

        return resultados_locales

    except Exception as e:

        return f"❌ [{fecha_str} | {contaminante}] {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# FECHAS
# ─────────────────────────────────────────────────────────────────────────────
def generar_fechas(start_date, end_date):

    delta = end_date - start_date

    return [
        (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(delta.days + 1)
    ]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print("Iniciando Dask...")

    client = Client(
        n_workers=4,
        threads_per_worker=2,
        memory_limit='4GB'
    )

    print(client.dashboard_link)

    fecha_inicio = date(2020, 1, 1)
    fecha_fin    = date(2024, 12, 31)

    lista_fechas = generar_fechas(
        fecha_inicio,
        fecha_fin
    )

    futures = []

    for fecha in lista_fechas:

        for contaminante in CONTAMINANTES.keys():

            futures.append(
                dask.delayed(procesar_s5p)(
                    fecha,
                    contaminante
                )
            )

    print(f"Tareas: {len(futures)}")

    resultados = dask.compute(*futures)

    manifest = []

    exitos = 0
    errores = 0

    for r in resultados:

        if isinstance(r, list):

            manifest.extend(r)
            exitos += len(r)

        else:

            errores += 1
            print(r)

    # -------------------------------------------------------------------------
    # GUARDAR MANIFEST
    # -------------------------------------------------------------------------
    with open('manifest.json', 'w', encoding='utf-8') as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n====================================")
    print("DESCARGA FINALIZADA")
    print("====================================")
    print(f"Archivos válidos : {exitos}")
    print(f"Errores          : {errores}")
    print(f"Manifest entries : {len(manifest)}")