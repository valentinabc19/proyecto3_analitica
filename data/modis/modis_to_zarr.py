import os
import json
import hashlib
import pandas as pd
import xarray as xr
import rioxarray 
import s3fs
from dask.distributed import Client
from dask import delayed, compute
from dotenv import load_dotenv

# ─── CARGAR VARIABLES ────────────────────────────────────────────────────────
load_dotenv()
ACCESS_KEY  = os.getenv('WASABI_ACCESS_KEY')
SECRET_KEY  = os.getenv('WASABI_SECRET_KEY')
REGION      = os.getenv('WASABI_REGION')
SRC_BUCKET  = os.getenv('WASABI_SOURCE_BUCKET')
DEST_BUCKET = os.getenv('WASABI_DEST_BUCKET')
ENDPOINT    = f'https://s3.{REGION}.wasabisys.com'

os.environ['AWS_ACCESS_KEY_ID'] = ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = SECRET_KEY
os.environ['AWS_S3_ENDPOINT'] = ENDPOINT.replace('https://', '')
os.environ['AWS_VIRTUAL_HOSTING'] = 'FALSE'
os.environ['AWS_HTTPS'] = 'YES'

# ─── BBOX OBLIGATORIO DEL PROYECTO (Pág. 4) ──────────────────────────────────
BBOX_CALI = [-76.75, 3.20, -76.30, 3.75]

# ─── FUNCIÓN PARALELIZADA PARA PROCESAR UN ARCHIVO MODIS ─────────────────────
@delayed
def procesar_modis_paralelo(ruta_s3, fs, bbox):
    """Procesa un archivo MODIS individual en paralelo"""
    try:
        # Extraer metadatos
        info = fs.info(ruta_s3)
        fecha_str = ruta_s3.split('_')[-1].replace('.tif', '')
        fecha = pd.to_datetime(fecha_str)
        
        # Cargar y recortar a BBOX de Cali
        da = rioxarray.open_rasterio(f"s3://{ruta_s3}")
        da = da.rio.clip_box(*bbox)
        da = da.squeeze(dim='band', drop=True)
        da = da.expand_dims(time=[fecha])
        da.name = "AOD_550nm"
        
        # Obtener o calcular MD5
        etag = info.get('ETag', '').replace('"', '')
        if len(etag) != 32:  # Si ETag no es MD5 (por multipart), calcularlo
            etag = calcular_md5_desde_s3(ruta_s3, fs)
        
        return {
            'dataarray': da,
            'metadata': {
                'ruta': f"s3://{ruta_s3}",
                'fecha': fecha_str,
                'tamaño_bytes': info.get('size', 0),
                'hash_md5': etag,
                'fuente': 'MODIS_MCD19A2'
            }
        }
    except Exception as e:
        print(f"Error procesando {ruta_s3}: {e}")
        return None

def main():
    print("🚀 Iniciando procesamiento paralelo de MODIS con Dask...")
    
    # Configurar cluster Dask
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
    print(f"📊 Dashboard: {client.dashboard_link}")
    
    # Conectar a S3
    fs = s3fs.S3FileSystem(
        key=ACCESS_KEY, 
        secret=SECRET_KEY, 
        client_kwargs={'endpoint_url': ENDPOINT}
    )
    
    # Buscar archivos MODIS
    archivos_modis = fs.glob(f'{SRC_BUCKET}/MODIS/*/*.tif')
    print(f"✅ Encontrados {len(archivos_modis)} archivos MODIS")
    
    if not archivos_modis:
        raise ValueError("No se encontraron archivos MODIS en el bucket")
    
    # ─── 1. PROCESAMIENTO PARALELO ──────────────────────────────────────────
    print("🔄 Procesando archivos en paralelo...")
    tareas = [procesar_modis_paralelo(arch, fs, BBOX_CALI) for arch in archivos_modis]
    resultados = compute(*tareas, scheduler='distributed')
    
    # Filtrar resultados válidos
    resultados_validos = [r for r in resultados if r is not None]
    print(f"✅ Procesados exitosamente: {len(resultados_validos)}/{len(archivos_modis)}")
    
    if not resultados_validos:
        raise ValueError("No se pudo procesar ningún archivo")
    
    # ─── 2. CREACIÓN DEL MANIFEST JSON (Pág. 5) ─────────────────────────────
    print("\n📝 Generando manifest JSON...")
    
    manifest = {
        "dataset": "MODIS_AOD_Cali_Metropolitano",
        "version": "1.0",
        "fecha_creacion": pd.Timestamp.now().isoformat(),
        "total_archivos": len(resultados_validos),
        "bounding_box": {
            "west": BBOX_CALI[0],
            "south": BBOX_CALI[1],
            "east": BBOX_CALI[2],
            "north": BBOX_CALI[3]
        },
        "archivos": [r['metadata'] for r in resultados_validos]
    }
    
    # Calcular tamaño total
    total_bytes = sum(r['metadata']['tamaño_bytes'] for r in resultados_validos)
    total_gb = total_bytes / (1024**3)
    manifest["tamaño_total_gb"] = round(total_gb, 2)
    
    # Guardar manifest
    with open('manifest_modis.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Manifest guardado: manifest_modis.json")
    print(f"📊 Tamaño total: {total_gb:.2f} GB")
    
    # Verificar umbral del proyecto (mínimo 50GB combinado con otras fuentes)
    if total_gb < 10:
        print(f"⚠️ MODIS aporta {total_gb:.2f} GB - Se necesita Sentinel-5P y Sentinel-2 para alcanzar 50GB")
    
    # ─── 3. COMBINAR DATASETS ───────────────────────────────────────────────
    print("\n🔗 Combinando datasets...")
    datasets = [r['dataarray'] for r in resultados_validos]
    ds = xr.concat(datasets, dim='time').sortby('time')
    
    # Configurar chunks óptimos para memoria
    ds = ds.chunk({'time': 30, 'y': 512, 'x': 512})
    
    print(f"📐 Dimensiones finales: {ds.dims}")
    print(f"📅 Rango temporal: {ds.time.min().values} → {ds.time.max().values}")
    
    # ─── 4. EXPORTAR A ZARR (Pág. 5) ────────────────────────────────────────
    ruta_zarr = f"{DEST_BUCKET}/GeoVision_Panel/MODIS_AOD.zarr"
    print(f"\n💾 Exportando a Zarr: s3://{ruta_zarr}")
    
    s3_store = s3fs.S3Map(root=ruta_zarr, s3=fs, check=False)
    
    ds.to_zarr(
        store=s3_store,
        consolidated=True,
        mode='w',
        compute=True
    )
    
    print("✅ Exportación completada exitosamente")
    
    # ─── 5. ESTADÍSTICAS BÁSICAS PARA EDA ───────────────────────────────────
    print("\n📊 Estadísticas del panel MODIS:")
    print(f"   - Total de archivos: {len(resultados_validos)}")
    print(f"   - Tamaño total: {total_gb:.2f} GB")
    print(f"   - Shape del cubo: {ds.dims}")
    print(f"   - AOD promedio: {float(ds.AOD_550nm.mean().values):.4f}")
    print(f"   - AOD máximo: {float(ds.AOD_550nm.max().values):.4f}")
    print(f"   - AOD mínimo: {float(ds.AOD_550nm.min().values):.4f}")
    
    # Resumen para el informe
    print("\n" + "="*60)
    print("✅ PROCESAMIENTO MODIS COMPLETADO")
    print("="*60)
    print(f"📦 Dataset almacenado en: s3://{ruta_zarr}")
    print(f"📄 Manifest: manifest_modis.json")
    print(f"💾 Tamaño: {total_gb:.2f} GB")
    print("="*60)
    
    # Cerrar cliente
    client.close()

if __name__ == '__main__':
    main()