import os
import ee
import boto3
import dask
import geemap
from dask.distributed import Client
from dotenv import load_dotenv
import nest_asyncio
import botocore

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
CALI_METRO = ee.Geometry.Rectangle([-76.75, 3.20, -76.30, 3.75])
BANDAS_ESPECTRALES = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
BANDA_SCL = ['SCL']

# Filtrar la colección
s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(CALI_METRO)
                 .filterDate('2020-01-01', '2024-12-31')
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)))

# Extraer a lista para poder iterar con Dask
total_images = s2_collection.size().getInfo()
lista_imagenes = s2_collection.toList(total_images)


# ─── FUNCIÓN WORKER PARA DASK ────────────────────────────────────────────────
def procesar_sentinel_a_wasabi(index):
    img_id_short = "Desconocido"
    temp_dir = "temp_s2_downloads"
    os.makedirs(temp_dir, exist_ok=True) 

    ruta_local_bands = None
    ruta_local_scl = None

    try:
        nest_asyncio.apply()
        
        load_dotenv()
        ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))
        
        s3_client = boto3.client('s3',
            endpoint_url=f"https://s3.{os.getenv('WASABI_REGION')}.wasabisys.com",
            aws_access_key_id=os.getenv('WASABI_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('WASABI_SECRET_KEY')
        )

        img = ee.Image(lista_imagenes.get(index))
        fecha = img.date().format('YYYY-MM-dd').getInfo()
        
        img_id_full = img.get('system:index').getInfo()
        img_id_short = img_id_full.split('_')[-1] 
        año = fecha.split('-')[0]

        nombre_bands = f'S2_{fecha}_{img_id_short}_bands.tif'
        nombre_scl   = f'S2_{fecha}_{img_id_short}_scl.tif'
        
        ruta_wasabi_bands = f'GeoVision/Sentinel2/Bands/{año}/{nombre_bands}'
        ruta_wasabi_scl   = f'GeoVision/Sentinel2/SCL/{año}/{nombre_scl}'

        # ─── 3.5. CHECKPOINT: VERIFICAR SI YA EXISTE EN WASABI ───
        try:
            # Si head_object no lanza error, el archivo existe
            s3_client.head_object(Bucket=os.getenv('WASABI_BUCKET'), Key=ruta_wasabi_bands)
            s3_client.head_object(Bucket=os.getenv('WASABI_BUCKET'), Key=ruta_wasabi_scl)
             # AGREGA ESTE PRINT PARA VERLO EN LA CONSOLA EN TIEMPO REAL:
            print(f"[⏭️] Omitido (Ya existe): {fecha} | {img_id_short}") 
            
            return f"[⏭️] Omitido (Ya existe): {fecha} | {img_id_short}"
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                # El archivo no existe, procedemos con la descarga normal
                pass
            else:
                raise # Otro tipo de error (ej. credenciales malas)
        # ──────────────────────────────────────────────────────────

        ruta_local_bands = os.path.join(temp_dir, nombre_bands)
        ruta_local_scl   = os.path.join(temp_dir, nombre_scl)

        # 4. Descargar Bandas Espectrales (Escala 10m) 
        img_bands = img.select(BANDAS_ESPECTRALES).clip(CALI_METRO)
        geemap.download_ee_image(
            image=img_bands, filename=ruta_local_bands, scale=10, region=CALI_METRO, crs='EPSG:4326'
        )

        # 5. Descargar Máscara SCL (Escala 20m)
        img_scl = img.select(BANDA_SCL).clip(CALI_METRO)
        geemap.download_ee_image(
            image=img_scl, filename=ruta_local_scl, scale=20, region=CALI_METRO, crs='EPSG:4326'
        )

        # 6. Subir archivos a Wasabi
        s3_client.upload_file(Filename=ruta_local_bands, Bucket=os.getenv('WASABI_BUCKET'), Key=ruta_wasabi_bands)
        s3_client.upload_file(Filename=ruta_local_scl, Bucket=os.getenv('WASABI_BUCKET'), Key=ruta_wasabi_scl)

        # 7. Limpieza
        os.remove(ruta_local_bands)
        os.remove(ruta_local_scl)

        return f"[✅] Éxito: {fecha} | {img_id_short}"

    except Exception as e:
        if ruta_local_bands and os.path.exists(ruta_local_bands): os.remove(ruta_local_bands)
        if ruta_local_scl and os.path.exists(ruta_local_scl): os.remove(ruta_local_scl)
        return f"[❌] Error en índice {index} ({img_id_short}): {str(e)}"

# ─── EJECUCIÓN DISTRIBUIDA ───────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Total de imágenes Sentinel-2 a procesar: {total_images}")
    
    # IMPORTANTE: Cambiamos a threads_per_worker=1
    # Ponemos 3 workers para compensar y mantener la velocidad
    client = Client(n_workers=3, threads_per_worker=1, memory_limit='5GB')
    print(f"Panel de Dask: {client.dashboard_link}")

    print("\nConstruyendo grafo de tareas...")
    # Creamos las tareas (del 0 al total de imágenes encontradas)
    futures = [dask.delayed(procesar_sentinel_a_wasabi)(i) for i in range(total_images)]

    print("Iniciando descargas. ¡Esto tomará varias horas debido al peso de Sentinel-2!")
    resultados = dask.compute(*futures)

    # Resumen
    exitosos = sum(1 for r in resultados if '[✅]' in r)
    fallidos = total_images - exitosos

    print("\n" + "="*40)
    print("🚀 REPORTE FINAL DE SENTINEL-2")
    print(f"✅ Descargados y subidos: {exitosos}")
    print(f"❌ Fallidos: {fallidos}")
    print("="*40)

    # Imprimir los primeros 5 errores para poder diagnosticarlos
    if fallidos > 0:
        print("\n🔍 DETALLE DE LOS ERRORES (Primeros 5):")
        errores = [r for r in resultados if '[❌]' in r]
        for e in errores[:5]:
            print(e)