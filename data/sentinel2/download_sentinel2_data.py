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
import time # Asegúrate de que esto esté al inicio del script

# ─── FUNCIÓN WORKER PARA DASK (CON REINTENTOS) ───────────────────────────────
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

        # ─── 3.5. CHECKPOINT ───
        try:
            s3_client.head_object(Bucket=os.getenv('WASABI_BUCKET'), Key=ruta_wasabi_bands)
            s3_client.head_object(Bucket=os.getenv('WASABI_BUCKET'), Key=ruta_wasabi_scl)
            
            # ¡AGREGA ESTE PRINT PARA VERLO EN VIVO!
            print(f"[⏭️] Omitido (Ya existe): {fecha} | {img_id_short}")
            
            return f"[⏭️] Omitido (Ya existe): {fecha} | {img_id_short}"
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                pass # No existe, procedemos
            else:
                raise 

        ruta_local_bands = os.path.join(temp_dir, nombre_bands)
        ruta_local_scl   = os.path.join(temp_dir, nombre_scl)

        # ─── BUCLE DE REINTENTOS PARA ESQUIVAR EL "TOO MANY REQUESTS" ───
        max_reintentos = 3
        for intento in range(max_reintentos):
            try:
                # Descargar Bandas Espectrales (10m)
                img_bands = img.select(BANDAS_ESPECTRALES).clip(CALI_METRO)
                geemap.download_ee_image(
                    image=img_bands, filename=ruta_local_bands, scale=10, region=CALI_METRO, crs='EPSG:4326'
                )

                # Descargar Máscara SCL (20m)
                img_scl = img.select(BANDA_SCL).clip(CALI_METRO)
                geemap.download_ee_image(
                    image=img_scl, filename=ruta_local_scl, scale=20, region=CALI_METRO, crs='EPSG:4326'
                )
                
                # Si llega aquí, descargó con éxito, rompemos el bucle de reintentos
                break 

            except Exception as e:
                error_msg = str(e)
                if "Too Many Requests" in error_msg and intento < max_reintentos - 1:
                    print(f"  [⏳] GEE saturado en {fecha}. Esperando 15s (Intento {intento+1}/{max_reintentos})...")
                    time.sleep(15) # Esperar 15 segundos para que Google nos perdone
                else:
                    raise e # Si es otro error o se acabaron los reintentos, que falle normalmente

        # Subir archivos a Wasabi
        s3_client.upload_file(Filename=ruta_local_bands, Bucket=os.getenv('WASABI_BUCKET'), Key=ruta_wasabi_bands)
        s3_client.upload_file(Filename=ruta_local_scl, Bucket=os.getenv('WASABI_BUCKET'), Key=ruta_wasabi_scl)

        # Limpieza
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
    
    client = Client(n_workers=2, threads_per_worker=1, memory_limit='8GB')
    print(f"Panel de Dask: {client.dashboard_link}")

    print("\nConstruyendo grafo de tareas...")
    futures = [dask.delayed(procesar_sentinel_a_wasabi)(i) for i in range(total_images)]

    print("Iniciando descargas...")
    resultados = dask.compute(*futures)

    # Resumen final
    exitosos = sum(1 for r in resultados if '[✅]' in r) + sum(1 for r in resultados if '[⏭️]' in r)
    fallidos = total_images - exitosos

    print("\n" + "="*40)
    print("🚀 REPORTE FINAL DE SENTINEL-2")
    print(f"✅ Procesados (Nuevos + Existentes): {exitosos}")
    print(f"❌ Fallidos: {fallidos}")
    print("="*40)

    if fallidos > 0:
        print("\n🔍 DETALLE DE LOS ERRORES (Primeros 5):")
        errores = [r for r in resultados if '[❌]' in r]
        for e in errores[:5]:
            print(e)