import os
import calendar
import tempfile
import cdsapi
import s3fs
from dask.distributed import Client, as_completed
from dotenv import load_dotenv

# ─── CARGAR VARIABLES DE ENTORNO ─────────────────────────────────────────────
load_dotenv()
WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
WASABI_REGION     = os.getenv('WASABI_REGION')
WASABI_BUCKET     = os.getenv('WASABI_BUCKET')
CDS_TOKEN         = os.getenv('CDS_TOKEN') # Tu token de Copernicus
CDS_URL           = os.getenv('CDS_URL')

# ─── CONFIGURACIÓN GEOGRÁFICA Y TEMPORAL (Pág 4 del PDF) ─────────────────────
# Bounding Box para CDS [Norte, Oeste, Sur, Este]
AREA = [3.75, -76.75, 3.20, -76.30]

AÑOS  = ['2020', '2021', '2022', '2023', '2024']
MESES = [str(m).zfill(2) for m in range(1, 13)]
HORAS = [f"{str(h).zfill(2)}:00" for h in range(24)]

# Variables meteorológicas solicitadas
VARIABLES = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'boundary_layer_height',   # BLH (Capa Límite)
    '2m_dewpoint_temperature', # Usada para calcular Humedad Relativa (RH)
]

# ─── FUNCIÓN DEL WORKER PARA DASK ────────────────────────────────────────────
# ─── FUNCIÓN DEL WORKER PARA DASK ────────────────────────────────────────────
def descargar_mes_y_subir(year, month):
    nombre_archivo = f'ERA5_Cali_{year}_{month}.nc'
    ruta_s3 = f'{WASABI_BUCKET}/GeoVision/ERA5/{nombre_archivo}'

    try:
        fs = s3fs.S3FileSystem(
            key=WASABI_ACCESS_KEY,
            secret=WASABI_SECRET_KEY,
            client_kwargs={'endpoint_url': f'https://s3.{WASABI_REGION}.wasabisys.com'}
        )

        if fs.exists(ruta_s3):
            return f"[⏭️] Saltado (Ya existe): {nombre_archivo}"

        _, num_dias = calendar.monthrange(int(year), int(month))
        dias_del_mes = [str(d).zfill(2) for d in range(1, num_dias + 1)]

        # ✅ URL y KEY pasados directamente: no depende de .cdsapirc ni os.environ
        c = cdsapi.Client(
            url='https://cds.climate.copernicus.eu/api',
            key=CDS_TOKEN,
            quiet=True,
            verify=False
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            ruta_local = os.path.join(temp_dir, nombre_archivo)

            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': VARIABLES,
                    'year': year,
                    'month': month,
                    'day': dias_del_mes,
                    'time': HORAS,
                    'area': AREA,
                    'data_format': 'netcdf',
                    'download_format': 'unarchived'
                },
                ruta_local
            )

            if not os.path.exists(ruta_local):
                return f"[❌] Error: Copernicus no devolvió {nombre_archivo}"

            tamaño_mb = round(os.path.getsize(ruta_local) / (1024**2), 2)
            fs.put(ruta_local, ruta_s3)
            return f"[✅] Éxito: {nombre_archivo} ({tamaño_mb} MB) subido a Wasabi."

    except Exception as e:
        return f"[❌] Error crítico en {year}-{month}: {str(e)}"

# ─── EJECUCIÓN DISTRIBUIDA ───────────────────────────────────────────────────
if __name__ == '__main__':
    
    # 3 workers máximo: Es la regla de oro de Copernicus para no banearte
    client = Client(n_workers=3, threads_per_worker=1, memory_limit='2GB')
    print("Dashboard de Dask:", client.dashboard_link)

    total_tareas = len(AÑOS) * len(MESES)
    print(f"\nIniciando petición de {total_tareas} meses a Copernicus...\n")

    # Enviar tareas a Dask
    futures = []
    for year in AÑOS:
        for month in MESES:
            future = client.submit(descargar_mes_y_subir, year, month)
            futures.append(future)

    # ── Tracking en tiempo real ──
    completadas = 0
    exitos = 0
    
    # as_completed nos permite imprimir cada tarea apenas termina, sin importar el orden
    for future in as_completed(futures):
        resultado = future.result()
        completadas += 1
        
        if "[✅]" in resultado or "[⏭️]" in resultado:
            exitos += 1
            
        print(f"[{completadas}/{total_tareas}] {resultado}")

    print("\n" + "="*40)
    print(f"REPORTE FINAL ERA5")
    print(f"Procesados correctamente: {exitos} de {total_tareas}")
    print("="*40)
    client.close()