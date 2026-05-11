import os
import ee
import requests
import boto3
import dask
from dask.distributed import Client
from datetime import date, timedelta
from dotenv import load_dotenv

# ─── CARGAR VARIABLES DE ENTORNO ─────────────────────────────────────────────
load_dotenv()
WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
WASABI_REGION     = os.getenv('WASABI_REGION')
WASABI_BUCKET     = os.getenv('WASABI_BUCKET')
GEE_PROJECT       = os.getenv('GEE_PROJECT_ID')

# ─── INICIALIZACIÓN DE GEE ───────────────────────────────────────────────────
ee.Initialize(project=GEE_PROJECT)

# ─── PARÁMETROS DEL DATASET ──────────────────────────────────────────────────
CALI_BBOX = ee.Geometry.Rectangle([-76.75, 3.20, -76.30, 3.75])
SCALE     = 1000 # GEE lo remuestreará a 1km

CONTAMINANTES = {
    'NO2': {'collection': 'COPERNICUS/S5P/OFFL/L3_NO2', 'band': 'tropospheric_NO2_column_number_density'},
    'SO2': {'collection': 'COPERNICUS/S5P/OFFL/L3_SO2', 'band': 'SO2_column_number_density'},
    'O3':  {'collection': 'COPERNICUS/S5P/OFFL/L3_O3',  'band': 'O3_column_number_density'},
}

# ─── FUNCIÓN WORKER (TAREA DASK) ─────────────────────────────────────────────
def procesar_s5p(fecha_str, contaminante):
    """
    Descarga 1 gas para 1 día específico y lo envía a Wasabi.
    """
    try:
        # 1. Cliente Boto3 dentro del worker
        s3_client = boto3.client('s3',
            endpoint_url=f'https://s3.{WASABI_REGION}.wasabisys.com',
            aws_access_key_id=WASABI_ACCESS_KEY,
            aws_secret_access_key=WASABI_SECRET_KEY
        )

        # 2. Manejo de fechas en GEE
        start = ee.Date(fecha_str)
        end   = start.advance(1, 'day')
        año   = fecha_str.split('-')[0]
        config = CONTAMINANTES[contaminante]

        # 3. Filtrar colección
        coleccion = (ee.ImageCollection(config['collection'])
                     .filterBounds(CALI_BBOX)
                     .filterDate(start, end)
                     .select(config['band']))

        # Verificar si el satélite pasó ese día sobre Cali
        if coleccion.size().getInfo() == 0:
            return f"[{fecha_str} | {contaminante}] Sin pasada del satélite."

        # Promediar si hay más de 1 órbita y recortar a Cali
        imagen_diaria = coleccion.mean().clip(CALI_BBOX)
        nombre_archivo = f'S5P_{contaminante}_{fecha_str}.tif'

        # 4. Generar URL de descarga directa a RAM
        url = imagen_diaria.getDownloadURL({
            'scale': SCALE,
            'crs': 'EPSG:4326',
            'region': CALI_BBOX,
            'format': 'GEO_TIFF'
        })

        # 5. Descargar a memoria
        response = requests.get(url)
        if response.status_code != 200:
            return f"[{fecha_str} | {contaminante}] Error HTTP de GEE."

        # 6. Subir directamente a Wasabi
        # Ruta: GeoVision/Sentinel5P/NO2/2020/archivo.tif
        s3_client.put_object(
            Bucket=WASABI_BUCKET,
            Key=f'GeoVision/Sentinel5P/{contaminante}/{año}/{nombre_archivo}',
            Body=response.content
        )

        return f"[{fecha_str} | {contaminante}] Éxito."

    except Exception as e:
        return f"[{fecha_str} | {contaminante}] Error: {str(e)}"

# ─── UTILIDADES ──────────────────────────────────────────────────────────────
def generar_fechas(start_date, end_date):
    delta = end_date - start_date
    return [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(delta.days + 1)]

# ─── EJECUCIÓN DISTRIBUIDA CON DASK ──────────────────────────────────────────
if __name__ == '__main__':
    print("Iniciando cluster local de Dask...")
    
    # 4 workers está perfecto porque la descarga va a la RAM y es muy ligera
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
    print(f"Panel Dask: {client.dashboard_link}")

    # Generar los 5 años (1827 días)
    fecha_inicio = date(2020, 1, 1)
    fecha_fin    = date(2024, 12, 31)
    lista_fechas = generar_fechas(fecha_inicio, fecha_fin)

    print("\nConstruyendo el grafo de tareas...")
    futures = []
    
    # Creamos una tarea por cada día y por cada contaminante
    for fecha in lista_fechas:
        for contaminante in CONTAMINANTES.keys(): # NO2, SO2, O3
            futures.append(dask.delayed(procesar_s5p)(fecha, contaminante))

    total_tareas = len(futures)
    print(f"Lanzando {total_tareas} tareas en paralelo... (Aprox 10-15 min)")
    
    # Ejecución paralela
    resultados = dask.compute(*futures)

    # Métricas Finales
    exitosos = sum(1 for r in resultados if '✅' in r)
    vacios   = sum(1 for r in resultados if '⚠️' in r)
    errores  = sum(1 for r in resultados if '❌' in r)

    print("\n" + "="*45)
    print("REPORTE FINAL SENTINEL-5P (2020-2024)")
    print("="*45)
    print(f"Archivos subidos: {exitosos}")
    print(f"Días sin órbitas: {vacios} (Normal en los polos/trópicos)")
    print(f"Errores GEE/Red : {errores}")
    print("=============================================")