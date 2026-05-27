import os
import rasterio
import zarr
import numpy as np
import s3fs
from dotenv import load_dotenv

# ─── 1. CONFIGURACIÓN WASABI Y S3FS ──────────────────────────────────────────
load_dotenv()
WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
WASABI_REGION     = os.getenv('WASABI_REGION')
WASABI_BUCKET     = os.getenv('WASABI_BUCKET')
ENDPOINT          = f'https://s3.{WASABI_REGION}.wasabisys.com'

# Conexión al Filesystem de S3
fs = s3fs.S3FileSystem(
    key=WASABI_ACCESS_KEY,
    secret=WASABI_SECRET_KEY,
    client_kwargs={'endpoint_url': ENDPOINT}
)

# ─── 2. RUTAS Y LISTADO DE ARCHIVOS ──────────────────────────────────────────
RUTA_ORIGEN_BANDS = f'{WASABI_BUCKET}/GeoVision/Sentinel2/Bands'
ZARR_S3           = f'{WASABI_BUCKET}/GeoVision_Panel/sentinel2.zarr'

print("Buscando imágenes de Sentinel-2 en Wasabi...")
# Buscar y ordenar cronológicamente los archivos TIF
archivos_tif = fs.glob(f'{RUTA_ORIGEN_BANDS}/*/*.tif')
archivos_tif.sort()

if not archivos_tif:
    raise ValueError("No se encontraron archivos. Verifica la ruta origen.")

# ─── 3. DIMENSIONES Y ZARR MAPPER ────────────────────────────────────────────
H, W, N = 6123, 5011, 12  # (Alto, Ancho, Bandas)
T = len(archivos_tif)     # Tiempo (Total de imágenes)

store_map = s3fs.S3Map(root=ZARR_S3, s3=fs, check=False)
store = zarr.open_group(store_map, mode='w')

# Crear el cubo principal de reflectancia
data = store.zeros(
    name   = 'reflectance',       
    shape  = (T, N, H, W),
    chunks = (1, N, 256, 256), # Chunking perfecto para Deep Learning
    dtype  = 'uint16'
)

# Crear el array de fechas
fechas_arr = store.zeros(
    name  = 'fechas',
    shape = (T,),
    dtype = 'U10'
)

print(f"Shape Zarr: {data.shape}")
print(f"Iniciando conversión transaccional de {T} imágenes a la nube...")

# ─── 4. PROCESO DE CONVERSIÓN ────────────────────────────────────────────────
errores = []

for i, ruta_s3 in enumerate(archivos_tif):
    tmp_file = 'temp_s2_bandas.tif'
    
    try:
        # Extraer fecha del nombre del archivo (S2_YYYY-MM-DD_ID_bands.tif)
        nombre_archivo = ruta_s3.split('/')[-1]
        fecha = nombre_archivo.split('_')[1]
        
        # Tamaño para el log
        size_mb = fs.info(ruta_s3)['size'] / 1e6

        # Descargar temporalmente desde Wasabi
        fs.get(ruta_s3, tmp_file)

        # Leer con Rasterio y rellenar (padding) si es necesario
        with rasterio.open(tmp_file) as src:
            img = src.read()
            if img.shape[1:] != (H, W):
                padded = np.zeros((N, H, W), dtype='uint16')
                
                # Evitar errores si el TIF es un poco más grande que el H,W objetivo
                c_h = min(img.shape[1], H)
                c_w = min(img.shape[2], W)
                
                padded[:, :c_h, :c_w] = img[:, :c_h, :c_w]
                img = padded

        # ─── AQUI SUCEDE LA MAGIA ───
        # Al asignar el array numpy a `data[i]`, Zarr automáticamente lo corta
        # en cuadritos de 256x256 y los sube uno por uno a Wasabi
        data[i]       = img
        fechas_arr[i] = fecha
        
        os.remove(tmp_file)

        # Checkpoint de metadatos para seguir el progreso
        if i % 10 == 0:
            store.attrs['ultimo_indice'] = i
            print(f"Checkpoint de metadatos guardado ({i}/{T})")

        print(f"✅ {i+1}/{T} — {fecha} ({size_mb:.0f} MB)")

    except Exception as e:
        print(f"❌ Error en índice {i} ({ruta_s3}): {e}")
        errores.append(i)
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

# ─── 5. METADATA GLOBAL ──────────────────────────────────────────────────────
store.attrs['bandas']         = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
store.attrs['bbox']           = [-76.75, 3.20, -76.30, 3.75]
store.attrs['fuente']         = 'Sentinel-2 L2A Harmonized'
store.attrs['periodo']        = '2020-2024'
store.attrs['total_imagenes'] = T

print(f"\n✅ Conversión completada exitosamente.")
print(f"📁 Zarr alojado en: s3://{ZARR_S3}")
print(f"📦 Shape Final:     {data.shape}")
print(f"❌ Errores:         {len(errores)}")