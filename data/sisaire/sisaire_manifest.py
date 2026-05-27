import os
import json
import hashlib
import tempfile
import boto3
import pandas as pd
from dotenv import load_dotenv

# Cargar credenciales del archivo .env
load_dotenv()

WASABI_ACCESS_KEY = os.getenv("WASABI_ACCESS_KEY")
WASABI_SECRET_KEY = os.getenv("WASABI_SECRET_KEY")
WASABI_REGION = os.getenv("WASABI_REGION", "us-east-1") 
BUCKET_NAME = "data-geo-raw"

BBOX = [-76.75, 3.20, -76.30, 3.75]

s3 = boto3.client(
    's3',
    endpoint_url=f'https://s3.{WASABI_REGION}.wasabisys.com',
    aws_access_key_id=WASABI_ACCESS_KEY,
    aws_secret_access_key=WASABI_SECRET_KEY
)

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def process_estaciones():
    print("Iniciando procesamiento de dataset DAGMA/SISAIRE...")
    manifest_entries = []
    
    estaciones_prefix = "GeoVision/estaciones_raw_data/"
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=estaciones_prefix)
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if not key.endswith('.csv'):
                continue
            
            print(f"Procesando: {key}")
            
            path_parts = key.split('/')
            try:
                year_part = [p for p in path_parts if "year=" in p][0]
                month_part = [p for p in path_parts if "month=" in p][0]
                year = year_part.split('=')[1]
                month = month_part.split('=')[1]
                fecha_adquisicion = f"{year}-{month}"
            except Exception:
                fecha_adquisicion = "Desconocida"
            
            # SOLUCIÓN PARA WINDOWS:
            tmp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            tmp_path = tmp_file.name
            tmp_file.close() # Libera el bloqueo del archivo inmediato
            
            try:
                s3.download_file(BUCKET_NAME, key, tmp_path)
                md5_hash = calculate_md5(tmp_path)
                
                try:
                    df = pd.read_csv(tmp_path)
                    dimensiones = {"filas": df.shape[0], "columnas": df.shape[1]}
                except Exception as e:
                    dimensiones = f"Error al leer dimensiones CSV: {str(e)}"
                    
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                
            manifest_entries.append({
                "ruta": f"s3://{BUCKET_NAME}/{key}",
                "hash_MD5": md5_hash,
                "dimensiones": dimensiones,
                "fecha_de_adquisicion": fecha_adquisicion,
                "fuente": "DAGMA / SISAIRE",
                "bounding_box": BBOX
            })

    output_file = "manifest_estaciones.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_entries, f, indent=4, ensure_ascii=False)
    
    print(f"\n¡Manifest de Estaciones generado con éxito en {output_file}!")

if __name__ == "_main_":
    process_estaciones()