import os
import json
import hashlib
import tempfile
import boto3
import xarray as xr
from dotenv import load_dotenv

# Cargar credenciales del archivo .env
load_dotenv()

WASABI_ACCESS_KEY = os.getenv("WASABI_ACCESS_KEY")
WASABI_SECRET_KEY = os.getenv("WASABI_SECRET_KEY")
WASABI_REGION = os.getenv("WASABI_REGION", "us-east-1") 
BUCKET_NAME = "data-geo-raw"

# Bounding box solicitado
BBOX = [-76.75, 3.20, -76.30, 3.75]

# Inicializar cliente S3 para Wasabi
s3 = boto3.client(
    's3',
    endpoint_url=f'https://s3.{WASABI_REGION}.wasabisys.com',
    aws_access_key_id=WASABI_ACCESS_KEY,
    aws_secret_access_key=WASABI_SECRET_KEY
)

def calculate_md5(file_path):
    """Calcula el hash MD5 real de un archivo local."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def process_era5():
    print("Iniciando procesamiento de dataset ERA5...")
    manifest_entries = []
    
    era5_prefix = "GeoVision/ERA5prueba/"
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=era5_prefix)
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if not key.endswith('.nc'):
                continue
                
            filename = key.split('/')[-1]
            print(f"Procesando: {filename}")
            
            parts = filename.replace('.nc', '').split('_')
            try:
                year, month = parts[-2], parts[-1]
                fecha_adquisicion = f"{year}-{month}"
            except Exception:
                fecha_adquisicion = "Desconocida"
            
            tmp_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
            tmp_path = tmp_file.name
            tmp_file.close() 
            
            try:
                s3.download_file(BUCKET_NAME, key, tmp_path)
                
                md5_hash = calculate_md5(tmp_path)
                
                try:
                    with xr.open_dataset(tmp_path) as ds:
                        dimensiones = dict(ds.dims)
                except Exception as e:
                    dimensiones = f"Error al leer dimensiones NetCDF: {str(e)}"
                    
            finally:
                # Asegura que el archivo temporal se borre siempre, incluso si hay errores
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            manifest_entries.append({
                "ruta": f"s3://{BUCKET_NAME}/{key}",
                "hash_MD5": md5_hash,
                "dimensiones": dimensiones,
                "fecha_de_adquisicion": fecha_adquisicion,
                "fuente": "ERA5",
                "bounding_box": BBOX
            })

    output_file = "manifest_era5.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_entries, f, indent=4, ensure_ascii=False)
    
    print(f"\n¡Manifest de ERA5 generado con éxito en {output_file}!")

if __name__ == "__main__":
    process_era5()