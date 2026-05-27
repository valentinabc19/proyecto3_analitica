import os
import boto3
from dotenv import load_dotenv

load_dotenv()

s3_client = boto3.client('s3',
    endpoint_url=f"https://s3.{os.getenv('WASABI_REGION')}.wasabisys.com",
    aws_access_key_id=os.getenv('WASABI_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('WASABI_SECRET_KEY')
)

bucket = os.getenv('WASABI_BUCKET')

print("Contando archivos reales en Wasabi...\n")

total_general = 0

# Contar por cada año
for anio in ['2020', '2021', '2022', '2023', '2024']:
    respuesta = s3_client.list_objects_v2(Bucket=bucket, Prefix=f'MODIS/{anio}/')
    
    # KeyCount nos da el número exacto de archivos que coinciden con esa ruta
    cantidad = respuesta.get('KeyCount', 0)
    print(f"Año {anio}: {cantidad} archivos")
    total_general += cantidad

print(f"\nTOTAL DESCARGADO: {total_general} archivos")