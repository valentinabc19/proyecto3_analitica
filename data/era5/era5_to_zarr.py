import os
from pathlib import Path
import s3fs
import xarray as xr
from dask.distributed import Client
from dotenv import load_dotenv
import cdsapi

# ─── CARGAR VARIABLES DE ENTORNO ─────────────────────────────────────────────
load_dotenv()
WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
WASABI_REGION     = os.getenv('WASABI_REGION')
WASABI_BUCKET     = os.getenv('WASABI_BUCKET')
ENDPOINT          = f'https://s3.{WASABI_REGION}.wasabisys.com'

# ─── CONFIGURACIÓN CDS API PARA ERA5 ─────────────────────────────────────────
# Opción 1: Configurar mediante variables de entorno (recomendado para el proyecto)
os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api/v2'
os.environ['CDSAPI_KEY'] = 'c9066123-1988-47d5-872b-8679903a850a'

# Opción 2: Crear archivo de configuración programáticamente
cdsapirc_content = f"""
url: https://cds.climate.copernicus.eu/api/v2
key: c9066123-1988-47d5-872b-8679903a850a
"""

# Guardar archivo .cdsapirc en el directorio de usuario
cdsapirc_path = Path.home() / '.cdsapirc'
if not cdsapirc_path.exists():
    with open(cdsapirc_path, 'w') as f:
        f.write(cdsapirc_content.strip())
    print(f"✅ Archivo de configuración creado en: {cdsapirc_path}")
else:
    print(f"✅ Archivo de configuración ya existe en: {cdsapirc_path}")

# Verificar que la configuración funciona
try:
    c = cdsapi.Client()
    print("✅ Conexión con CDS API establecida correctamente")
except Exception as e:
    print(f"❌ Error conectando con CDS API: {e}")
    print("Verifica que tu API key sea correcta y que tu cuenta esté activada")

# ─── FUNCIÓN PARA DESCARGAR ERA5-LAND ────────────────────────────────────────
def descargar_era5_cali(anio_inicio=2020, anio_fin=2024):
    """Descarga datos ERA5-Land para el área metropolitana de Cali"""
    
    c = cdsapi.Client()
    
    # BBOX para Cali según especificación del proyecto [north, west, south, east]
    bbox_cali = [3.55, -76.60, 3.30, -76.40]
    
    for anio in range(anio_inicio, anio_fin + 1):
        for mes in range(1, 13):
            mes_str = f"{mes:02d}"
            
            # Nombre del archivo de salida temporal
            archivo_salida = f"era5_cali_{anio}_{mes_str}.nc"
            
            # Solicitud de datos ERA5-Land según especificación del proyecto (pág. 4)
            request = {
                'variable': [
                    '2m_temperature', '10m_u_component_of_wind',
                    '10m_v_component_of_wind', 'boundary_layer_height',
                    'surface_pressure', 'total_precipitation'
                ],
                'year': str(anio),
                'month': mes_str,
                'day': [f"{d:02d}" for d in range(1, 32)],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': bbox_cali,  # [north, west, south, east]
                'format': 'netcdf'
            }
            
            print(f"📥 Descargando ERA5-Land: {anio}-{mes_str}")
            
            try:
                c.retrieve('reanalysis-era5-land', request, archivo_salida)
                print(f"✅ Descargado: {archivo_salida}")
                
                # Subir a Wasabi inmediatamente
                fs = s3fs.S3FileSystem(
                    key=WASABI_ACCESS_KEY,
                    secret=WASABI_SECRET_KEY,
                    client_kwargs={'endpoint_url': ENDPOINT}
                )
                
                ruta_wasabi = f"{RUTA_ORIGEN_NC}/{archivo_salida}"
                fs.upload(archivo_salida, ruta_wasabi)
                print(f"📤 Subido a Wasabi: {ruta_wasabi}")
                
                # Eliminar archivo local para liberar espacio
                os.remove(archivo_salida)
                
            except Exception as e:
                print(f"❌ Error en {anio}-{mes_str}: {e}")
                continue

# Inyectar credenciales al entorno para que las librerías C/C++ subyacentes tengan acceso
os.environ['AWS_ACCESS_KEY_ID'] = WASABI_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = WASABI_SECRET_KEY
os.environ['AWS_S3_ENDPOINT'] = ENDPOINT.replace('https://', '')
os.environ['AWS_VIRTUAL_HOSTING'] = 'FALSE'
os.environ['AWS_HTTPS'] = 'YES'

# ─── RUTAS ───────────────────────────────────────────────────────────────────
RUTA_ORIGEN_NC = f'{WASABI_BUCKET}/GeoVision/ERA5'
RUTA_DESTINO_ZARR = f'{WASABI_BUCKET}/GeoVision_Panel/ERA5.zarr'
RUTA_MANIFEST_LOCAL = 'manifest_era5.json'
BBOX_CALI = [3.75, -76.75, 3.20, -76.30] # [N, W, S, E]

def main():
    # 1. Iniciar Dask 
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
    print("Dashboard de Dask disponible en:", client.dashboard_link)
    
    # 2. Descargar datos ERA5 si no existen
    print("\n🔄 Verificando/Descargando datos ERA5-Land...")
    descargar_era5_cali()

    # 3. Conectarse a Wasabi
    fs = s3fs.S3FileSystem(
        key=WASABI_ACCESS_KEY,
        secret=WASABI_SECRET_KEY,
        client_kwargs={'endpoint_url': ENDPOINT}
    )

    print("\nBuscando archivos NetCDF en Wasabi...")
    archivos_nc = fs.glob(f'{RUTA_ORIGEN_NC}/*.nc')
    
    if not archivos_nc:
        raise ValueError("No se encontraron archivos .nc. Verifica la ruta origen.")
        
    rutas_s3 = ['s3://' + archivo for archivo in archivos_nc]
    print(f"Se encontraron {len(rutas_s3)} archivos. Generando Manifest...")

    # ─── 4. ABRIR DATASET MASIVO ─────────────────────────────────────────────
    print("\nAbriendo cubo multidimensional distribuido...")
    
    ds_masivo = xr.open_mfdataset(
        rutas_s3,
        engine='h5netcdf', 
        combine='by_coords', 
        parallel=True,       # ¡Clave! Obliga a Dask a abrir los metadatos en paralelo
        storage_options={'key': WASABI_ACCESS_KEY, 'secret': WASABI_SECRET_KEY, 'client_kwargs': {'endpoint_url': ENDPOINT}}
    )

    if 'valid_time' in ds_masivo.dims:
        ds_masivo = ds_masivo.rename({'valid_time': 'time'})

    # ─── 5. CHUNKING ESPACIO-TEMPORAL ────────────────────────────────────────
    # time: 720 horas = ~30 días (1 mes por bloque)
    # lat/lon: -1 = Todo el espacio en un solo bloque (ideal para áreas pequeñas como Cali)
    ds_chunked = ds_masivo.chunk({
        'time': 720,       
        'latitude': -1,    
        'longitude': -1
    })

    print("\nDataset preparado con la siguiente estructura:")
    print(ds_chunked)

    # ─── 6. GUARDAR ZARR EN WASABI ───────────────────────────────────────────
    print(f"\nIniciando conversión y subida a Zarr en: s3://{RUTA_DESTINO_ZARR}")
    print("¡Revisa el panel de Dask para ver el progreso!")

    store = s3fs.S3Map(root=RUTA_DESTINO_ZARR, s3=fs, check=False)

    ds_chunked.to_zarr(
        store=store,
        consolidated=True, 
        mode='w'           
    )

    print(f"\n Panel Zarr de ERA5 guardado exitosamente en la nube.")

if __name__ == '__main__':
    main()