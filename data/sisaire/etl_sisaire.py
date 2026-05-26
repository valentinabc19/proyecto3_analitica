import dask.bag as db
from dask.distributed import Client
import urllib.parse
import boto3
import calendar
import os
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURACIÓN WASABI ───
WASABI_ACCESS_KEY = os.getenv("WASABI_ACCESS_KEY")
WASABI_SECRET_KEY = os.getenv("WASABI_SECRET_KEY")
WASABI_ENDPOINT = "https://s3.wasabisys.com"
RAW_BUCKET = os.getenv("WASABI_BUCKET")

# Validación rápida para evitar errores
if not WASABI_ACCESS_KEY or not RAW_BUCKET:
    raise ValueError("Faltan credenciales en el archivo .env")

def descargar_mes_raw(mes_año):
    year, month = mes_año
    last_day = calendar.monthrange(year, month)[1]
    
    # Query basada en tu código pero segmentada por mes para Dask
    consulta_sql = f"""
    SELECT * 
    WHERE (municipio = 'Cali' OR municipio = 'Santiago de Cali' OR municipio = 'Yumbo' 
           OR nombre_fgda = 'CVC' OR nombre_fgda = 'SISAIRE' OR nombre_fgda = 'SISAIRE - CVC' OR nombre_fgda = 'IDEAM' OR nombre_fgda = 'DAGMA') 
      AND med_fecha_inicio >= '{year}-{month:02d}-01T00:00:00.000' 
      AND med_fecha_inicio <= '{year}-{month:02d}-{last_day:02d}T23:59:59.000'
    LIMIT 32788084
    """
    
    consulta_codificada = urllib.parse.quote(consulta_sql.strip())
    url = f"https://www.datos.gov.co/resource/g4t8-zkc3.csv?$query={consulta_codificada}"
    
    try:
        # Descarga directa del stream de bytes
        import requests
        response = requests.get(url)
        if response.status_code != 200:
            return f"Error {response.status_code} en {year}-{month}"
        
        # Subir a Wasabi (Raw)
        s3 = boto3.client('s3', endpoint_url=WASABI_ENDPOINT, 
                          aws_access_key_id=WASABI_ACCESS_KEY, 
                          aws_secret_access_key=WASABI_SECRET_KEY)
        
        path_wasabi = f"raw/year={year}/month={month:02d}/datos_brutos.csv"
        s3.put_object(Body=response.content, Bucket=RAW_BUCKET, Key=path_wasabi)
        
        return f"Descargado: {year}-{month}"
    except Exception as e:
        return f"Error en {year}-{month}: {e}"

if __name__ == '__main__':
    client = Client(n_workers=4) # Pipeline distribuido
    print(f"Dashboard de Dask: {client.dashboard_link}")
    
    # Generar lista de meses 2020-2024
    meses = [(y, m) for y in range(2020, 2025) for m in range(1, 13)]
    
    # Ejecución distribuida
    b = db.from_sequence(meses)
    resultados = b.map(descargar_mes_raw).compute()
    
    for res in resultados:
        print(res)