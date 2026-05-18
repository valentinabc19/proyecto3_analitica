import os
import pandas as pd
from sodapy import Socrata

# ─── 1. RUTAS LOCALES ────────────────────────────────────────────────────────
CARPETA_DESTINO = './data/sisaire'
RUTA_LOCAL_CSV  = f'{CARPETA_DESTINO}/DAGMA_SISAIRE.csv'

def extraer_sisaire_local():
    print("🌐 Conectando a la API de Datos Abiertos de Colombia (Socrata)...")
    
    # Cliente de Socrata sin token
    cliente = Socrata("www.datos.gov.co", None)
    DATASET_ID = "g4t8-zkc3"
    
    # ─── 2. CONSULTA SQL A LA API (Con los nombres de columna reales) ────────
    # Usamos nombre_fgda para el DAGMA y med_fecha_inicio para la fecha
    query_where = "(municipio = 'Cali' OR municipio = 'Santiago de Cali' OR nombre_fgda = 'SISAIRE') AND med_fecha_inicio >= '2020-01-01T00:00:00.000'"
    
    print("⏳ Descargando registros históricos de Cali (2020-2024)...")
    resultados = cliente.get(DATASET_ID, where=query_where, limit=500000)
    
    df_raw = pd.DataFrame.from_records(resultados)
    
    if df_raw.empty:
        raise ValueError("❌ La API no devolvió datos. Verifica la consulta.")
        
    print(f"✅ Descarga completada: {len(df_raw)} registros obtenidos de la API.")
    
    # ─── 3. LIMPIEZA Y TRANSFORMACIÓN ───────────────────────────────────────
    print("🧹 Estructurando el Dataset...")
    
    # Mapeo usando los nombres reales de la imagen que enviaste
    columnas_utiles = {
        'med_fecha_inicio': 'fecha',
        'nombre_est': 'estacion', 
        'msfl_code': 'contaminante',
        'med_concentracion_estandar': 'concentracion',
        'latitud': 'lat',
        'longitud': 'lon'
    }
    
    # Renombrar
    columnas_presentes = {k: v for k, v in columnas_utiles.items() if k in df_raw.columns}
    df = df_raw[list(columnas_presentes.keys())].rename(columns=columnas_presentes)
    
    # Convertir a tipos numéricos y de fecha
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['concentracion'] = pd.to_numeric(df['concentracion'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    # Eliminar vacíos
    df = df.dropna(subset=['fecha', 'concentracion', 'lat', 'lon'])
    
    # ─── 4. FILTRO GEOESPACIAL (Bounding Box) ──────────────────────────────
    print("🗺️ Aplicando recorte espacial BBox: [-76.75, 3.20, -76.30, 3.75]")
    # Latitud entre 3.20 y 3.75 | Longitud entre -76.75 y -76.30
    df = df[
        (df['lat'] >= 3.20) & (df['lat'] <= 3.75) &
        (df['lon'] >= -76.75) & (df['lon'] <= -76.30)
    ]
    
    # ─── 5. HOMOLOGACIÓN DE CONTAMINANTES ──────────────────────────────────
    df['contaminante'] = df['contaminante'].str.upper().str.strip()
    # A veces en el SISAIRE vienen así, los estandarizamos:
    df['contaminante'] = df['contaminante'].replace({
        'PM 2.5': 'PM25',
        'PM2.5': 'PM25',
        'PM 10': 'PM10'
    })
    
    contaminantes_validos = ['NO2', 'SO2', 'O3', 'PM10', 'PM25']
    df = df[df['contaminante'].isin(contaminantes_validos)]
    
    df['año'] = df['fecha'].dt.year.astype(str)
    df['mes'] = df['fecha'].dt.month.astype(str).str.zfill(2)
    
    print(f"Estructura final: {len(df)} mediciones limpias dentro del área de Cali.")
    
    # ─── 6. GUARDAR LOCALMENTE ─────────────────────────────────────────────
    os.makedirs(CARPETA_DESTINO, exist_ok=True)

    print(f"📄 Guardando datos limpios en: {RUTA_LOCAL_CSV} ...")
    df.to_csv(RUTA_LOCAL_CSV, index=False, encoding='utf-8')

    print("\n✅ Descarga y estructuración local completada con éxito")

if __name__ == '__main__':
    extraer_sisaire_local()