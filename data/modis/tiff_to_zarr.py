import os
import json
import pandas as pd
import xarray as xr
import rioxarray 
import s3fs
import dask
from dask.distributed import Client
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
BBOX_CALI = [-76.60, 3.30, -76.40, 3.55]

def main():
    print("Iniciando Dask Cluster...")
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
    
    print(f"\nExplorando bucket de origen: {SRC_BUCKET}")
    fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={'endpoint_url': ENDPOINT})
    archivos_tif = fs.glob(f'{SRC_BUCKET}/MODIS/*/*.tif')
    
    if not archivos_tif:
        raise ValueError("No se encontraron archivos.")
    
    print(f"Se encontraron {len(archivos_tif)} archivos. Iniciando extracción de metadatos...")

    # ─── 1. CREACIÓN DEL MANIFEST JSON (Pág. 5 del Proyecto) ─────────────────
    
    # Leemos solo el primer archivo para obtener las dimensiones espaciales exactas (ahorra horas de cómputo)
    primer_raster = rioxarray.open_rasterio(f"s3://{archivos_tif[0]}")
    dimensiones_espaciales = {'x': primer_raster.rio.width, 'y': primer_raster.rio.height}
    primer_raster.close()

    manifest_entries = []

    for ruta in archivos_tif:
        # Extraer metadatos ultrarrápidos de Wasabi
        info = fs.info(ruta)
        
        # En Wasabi/S3, el ETag es el Hash MD5 en formato Hex (quitamos las comillas)
        hash_md5 = info.get('ETag', '').replace('"', '') 
        
        # Extraer fecha del nombre del archivo
        fecha_str = ruta.split('_')[-1].replace('.tif', '')

        # Estructura exigida por la Tarea 6
        manifest_entries.append({
            "ruta": f"s3://{ruta}",
            "hash_md5": hash_md5,
            "tamaño_bytes": info.get('size'),
            "dimensiones": dimensiones_espaciales,
            "fecha_adquisicion": fecha_str,
            "fuente": "MODIS/061/MCD19A2_GRANULES",
            "bounding_box": BBOX_CALI
        })

    # Guardar el JSON localmente (Este es el archivo que debes subir a Git)
    ruta_manifest = 'manifest_modis.json'
    with open(ruta_manifest, 'w', encoding='utf-8') as f:
        json.dump({"dataset": "MODIS_AOD_Cali", "total_archivos": len(manifest_entries), "archivos": manifest_entries}, f, indent=4)
    
    print(f"Manifest JSON creado exitosamente: {ruta_manifest}")

    # ─── 2. CONSTRUCCIÓN DEL CUBO MULTIDIMENSIONAL ───────────────────────────
    print("\nConstruyendo Data Cube para la exportación a Zarr...")
    
    def cargar_tif_s3(ruta_s3):
        fecha_str = ruta_s3.split('_')[-1].replace('.tif', '')
        fecha = pd.to_datetime(fecha_str)
        da = rioxarray.open_rasterio(f"s3://{ruta_s3}", chunks='auto')
        da = da.squeeze(dim='band').drop_vars('band')
        da = da.expand_dims(time=[fecha])
        da.name = "AOD_550nm"
        return da

    # Mapeo perezoso (lazy)
    datasets = [cargar_tif_s3(archivo) for archivo in archivos_tif]
    ds = xr.concat(datasets, dim='time').sortby('time')
    ds = ds.chunk({'time': 30, 'y': -1, 'x': -1})

    # ─── 3. EXPORTACIÓN A ZARR ───────────────────────────────────────────────
    ruta_zarr = f"{DEST_BUCKET}/GeoVision_Panel/MODIS.zarr"
    print(f"\nIniciando conversión distribuida a Zarr en: s3://{ruta_zarr}")
    
    s3_store = s3fs.S3Map(root=ruta_zarr, s3=fs, check=False)
    
    ds.to_zarr(
        store=s3_store, 
        consolidated=True, 
        mode='w'
    )

    print("Conversión a Zarr completada con éxito")

if __name__ == '__main__':
    main()