import os
import json
import hashlib
import requests
import boto3
import ee
from rasterio.io import MemoryFile
from dask.distributed import Client, as_completed
from datetime import date, timedelta
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

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

SCALE       = 7000          # Resolución coherente con Sentinel-5P
EXPORT_CRS  = 'EPSG:4326'
BANDA_CLOUD = 'cloud_fraction'

CONTAMINANTES = {
    'NO2': {'collection': 'COPERNICUS/S5P/OFFL/L3_NO2', 'band': 'tropospheric_NO2_column_number_density'},
    'SO2': {'collection': 'COPERNICUS/S5P/OFFL/L3_SO2', 'band': 'SO2_column_number_density'},
    'O3':  {'collection': 'COPERNICUS/S5P/OFFL/L3_O3',  'band': 'O3_column_number_density'},
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
# HASH MD5
# ─────────────────────────────────────────────────────────────────────────────
def calcular_md5(binary_data: bytes) -> str:
    return hashlib.md5(binary_data).hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
# WORKER — libre de cualquier estado global no serializable
# ─────────────────────────────────────────────────────────────────────────────
def procesar_s5p(fecha_str: str, contaminante: str) -> list | str:
    """
    Descarga la banda cloud_fraction de TODAS las órbitas disponibles
    para (fecha_str, contaminante) sin aplicar filtro de QA.

    Retorna:
        list  -> entradas del manifest para las órbitas descargadas con éxito.
        str   -> mensaje de aviso/error si no hay datos o falla algo grave.
    """
    tarea_id = f'{fecha_str} | {contaminante}'
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=f'https://s3.{WASABI_REGION}.wasabisys.com',
            aws_access_key_id=WASABI_ACCESS_KEY,
            aws_secret_access_key=WASABI_SECRET_KEY
        )

        start  = ee.Date(fecha_str)
        end    = start.advance(1, 'day')
        config = CONTAMINANTES[contaminante]
        pollutant_band = config['band']

        # Filtrado: solo bounds + fecha, SIN filtro QA
        collection = (
            ee.ImageCollection(config['collection'])
            .filterBounds(CALI_BBOX)
            .filterDate(start, end)
        )

        size = collection.size().getInfo()
        if size == 0:
            return f"⚠️ [{tarea_id}] Sin órbitas."

        image_list = collection.toList(size)
        resultados_locales = []

        for i in range(size):
            image = ee.Image(image_list.get(i))

            # Seleccionar cloud_fraction
            try:
                image = image.select([pollutant_band, BANDA_CLOUD])
            except Exception:
                continue

            # Verificar que haya píxeles con valor en la región
            stats = image.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=CALI_BBOX,
                scale=SCALE,
                maxPixels=1e9
            ).getInfo()

            pixel_count = list(stats.values())[0] if stats else 0
            if not pixel_count:
                continue

            acquisition_time = (
                ee.Date(image.get('system:time_start'))
                .format('YYYY-MM-dd_HH-mm-ss')
                .getInfo()
            )

            año            = fecha_str.split('-')[0]
            nombre_archivo = f'S5P_{contaminante}_{acquisition_time}.tif'
            key = f'GeoVision/Sentinel5P/{contaminante}/{año}/{nombre_archivo}'

            # Descarga
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

            # Validación TIFF
            try:
                with MemoryFile(binary_data) as memfile:
                    with memfile.open() as dataset:
                        width  = dataset.width
                        height = dataset.height
                        bounds = dataset.bounds
                        if width <= 0 or height <= 0:
                            continue
            except Exception:
                continue

            md5_hash = calcular_md5(binary_data)

            # Subida a Wasabi
            try:
                s3_client.put_object(
                    Bucket=WASABI_BUCKET,
                    Key=key,
                    Body=binary_data,
                    ContentType='image/tiff'
                )
                print(f"✅ Subido: {key}")
            except Exception as e:
                print(f"❌ Error subiendo {key}: {type(e).__name__}: {e}")
                continue  

            resultados_locales.append({
                "file_path": key,
                "md5": md5_hash,
                "dimensions": [width, height],
                "acquisition_date": acquisition_time,
                "source": config['collection'],
                "band": BANDA_CLOUD,
                "bounding_box": [
                    bounds.left, bounds.bottom,
                    bounds.right, bounds.top
                ],
                "pollutant": contaminante,
                "scale_meters": SCALE,
                "crs": EXPORT_CRS
            })

        return resultados_locales

    except Exception as e:
        return f"❌ [{tarea_id}] {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# FECHAS
# ─────────────────────────────────────────────────────────────────────────────
def generar_fechas(start_date: date, end_date: date) -> list[str]:
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
    print(f"Dashboard: {client.dashboard_link}\n")

    fecha_inicio = date(2020, 1, 1)
    fecha_fin    = date(2024, 12, 31)
    lista_fechas = generar_fechas(fecha_inicio, fecha_fin)

    futures = [
        client.submit(procesar_s5p, fecha, contaminante)
        for fecha in lista_fechas
        for contaminante in CONTAMINANTES.keys()
    ]

    total_tareas = len(futures)
    print(f"Tareas programadas  : {total_tareas}")
    print(f"  · Fechas          : {len(lista_fechas)}")
    print(f"  · Contaminantes   : {list(CONTAMINANTES.keys())}\n")

    # ── Progreso en tiempo real ───────────────────────────
    manifest     = []
    exitos       = 0
    advertencias = 0
    errores      = 0

    with tqdm(total=total_tareas, desc='Descargando', unit='tarea',
              dynamic_ncols=True, colour='cyan') as pbar:

        for future in as_completed(futures):
            try:
                resultado = future.result()
            except Exception as e:
                errores += 1
                pbar.set_postfix_str(f'❌ {str(e)[:60]}', refresh=True)
                pbar.update(1)
                continue

            if isinstance(resultado, list):
                manifest.extend(resultado)
                exitos += len(resultado)
                pbar.set_postfix_str(
                    f'✅ +{len(resultado)} archivo(s)', refresh=True
                )
            elif isinstance(resultado, str):
                if resultado.startswith('⚠️'):
                    advertencias += 1
                    pbar.set_postfix_str(
                        f'⚠️ {resultado[3:60]}', refresh=True
                    )
                else:
                    errores += 1
                    pbar.set_postfix_str(
                        f'❌ {resultado[:60]}', refresh=True
                    )

            pbar.update(1)

    # ─────────────────────────────────────────────────────────────────────────
    # GUARDAR MANIFEST
    # ─────────────────────────────────────────────────────────────────────────
    with open('manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n====================================")
    print("      DESCARGA FINALIZADA           ")
    print("====================================")
    print(f"Archivos subidos    : {exitos}")
    print(f"Sin órbitas (aviso) : {advertencias}")
    print(f"Errores             : {errores}")
    print(f"Manifest entries    : {len(manifest)}")
    print("====================================")