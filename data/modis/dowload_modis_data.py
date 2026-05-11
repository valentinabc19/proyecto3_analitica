import os
import ee
import requests
import boto3
import dask
from dask.distributed import Client
from datetime import date, timedelta
from dotenv import load_dotenv

# ─── CARGAR VARIABLES DE ENTORNO ─────────────────────────────────────────────
# Esto lee el archivo .env y carga las variables en memoria
load_dotenv()

WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
WASABI_REGION     = os.getenv('WASABI_REGION')
WASABI_BUCKET     = os.getenv('WASABI_BUCKET')
GEE_PROJECT       = os.getenv('GEE_PROJECT_ID')

# Validación rápida para evitar errores
if not WASABI_ACCESS_KEY or not WASABI_BUCKET:
    raise ValueError("Faltan credenciales en el archivo .env")

# ─── 1. INICIALIZACIÓN DE GOOGLE EARTH ENGINE ────────────────────────────────
ee.Authenticate() # Solo la primera vez
ee.Initialize(project=GEE_PROJECT)

# ─── 3. PARÁMETROS GEOGRÁFICOS Y DE DATASET ──────────────────────────────────
cali = ee.Geometry.Rectangle([-76.75, 3.20, -76.30, 3.75])

def apply_scale_factor(image):
    aod_green = image.select('Optical_Depth_055').multiply(0.001)
    return aod_green.rename('AOD_550nm').copyProperties(image, ['system:time_start'])

modis_raw = (ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
             .map(apply_scale_factor))

# ─── 4. FUNCIÓN WORKER (TAREA INDIVIDUAL PARA DASK) ──────────────────────────
def export_daily_to_wasabi(target_date_str):
    try:
        # Usamos las variables cargadas desde el .env
        s3_client = boto3.client('s3',
            endpoint_url=f'https://s3.{WASABI_REGION}.wasabisys.com',
            aws_access_key_id=WASABI_ACCESS_KEY,
            aws_secret_access_key=WASABI_SECRET_KEY
        )

        start = ee.Date(target_date_str)
        end   = start.advance(1, 'day')
        year  = target_date_str.split('-')[0] 

        img_daily = (modis_raw
                     .filterDate(start, end)
                     .median()              
                     .clip(cali))
        
        label = f'MODIS_AOD_Cali_{target_date_str}.tif'

        url = img_daily.getDownloadURL({
            'scale': 1000,           
            'crs': 'EPSG:4326',      
            'region': cali,
            'format': 'GEO_TIFF'
        })

        response = requests.get(url)
        if response.status_code != 200:
            return f"[{target_date_str}] Error HTTP de GEE."

        # SUBIDA A WASABI
        s3_client.put_object(
            Bucket=WASABI_BUCKET,
            Key=f'MODIS/{year}/{label}', 
            Body=response.content
        )
        
        return f'[{target_date_str}] Éxito.'

    except Exception as e:
        return f"[{target_date_str}] Omitido (Sin datos o error de red)"

def generate_date_list(start_date, end_date):
    delta = end_date - start_date
    return [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(delta.days + 1)]

# ─── 5. EJECUCIÓN PRINCIPAL ──────────────────────────────────────────────────
if __name__ == '__main__':
    
    # --- CREAR LA CARPETA 'MODIS' EN WASABI ---
    print("Conectando con Wasabi para configurar el bucket...")
    s3_setup = boto3.client('s3',
        endpoint_url=f'https://s3.{WASABI_REGION}.wasabisys.com',
        aws_access_key_id=WASABI_ACCESS_KEY,
        aws_secret_access_key=WASABI_SECRET_KEY
    )
    
    s3_setup.put_object(Bucket=WASABI_BUCKET, Key='MODIS/')
    print("Carpeta principal 'MODIS/' creada/verificada.")

    print("\nIniciando cluster de Dask...")
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
    print(f"Panel de monitoreo Dask: {client.dashboard_link}")

    fecha_inicio = date(2020, 1, 1)
    fecha_fin    = date(2024, 12, 31)
    lista_fechas = generate_date_list(fecha_inicio, fecha_fin)

    print(f'\nConstruyendo el grafo para {len(lista_fechas)} días...')
    futures = [dask.delayed(export_daily_to_wasabi)(fecha) for fecha in lista_fechas]

    print(f"Lanzando tareas en paralelo...")
    resultados = dask.compute(*futures)

    exitosos = sum(1 for r in resultados if 'Éxito' in r)
    omitidos = len(resultados) - exitosos

    print('\n Proceso de extracción finalizado')
    print(f" - Días descargados con éxito: {exitosos}")
    print(f" - Días sin datos (nubes/errores): {omitidos}")