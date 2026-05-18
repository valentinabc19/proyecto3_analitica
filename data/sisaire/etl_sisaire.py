import os
import pandas as pd
import urllib.parse

# ─── 1. RUTAS LOCALES ────────────────────────────────────────────────────────
CARPETA_DESTINO = './data/sisaire'
# Cambiamos el nombre para que refleje el dataset completo
RUTA_LOCAL_CSV  = f'{CARPETA_DESTINO}/data_estaciones.csv' 

def extraer_datos_terrestres():
    print("🌐 Conectando directamente al CSV de la API de Datos Abiertos...")
    
    # ─── 2. CREAR LA URL DE DESCARGA DIRECTA (SoQL) ────────────────────────
    # 🔥 AÑADIMOS 'SISAIRE' A LAS AUTORIDADES AMBIENTALES
    consulta_sql = """
    SELECT * 
    WHERE (municipio = 'Cali' 
           OR municipio = 'Santiago de Cali' 
           OR municipio = 'Yumbo' 
           OR nombre_fgda = 'CVC'
           OR nombre_fgda = 'SISAIRE') 
      AND med_fecha_inicio >= '2020-01-01T00:00:00.000' 
      AND med_fecha_inicio <= '2024-12-31T23:59:59.000'
    LIMIT 32788084
    """
    
    consulta_codificada = urllib.parse.quote(consulta_sql.strip())
    url_descarga = f"https://www.datos.gov.co/resource/g4t8-zkc3.csv?$query={consulta_codificada}"
    
    print("⏳ Descargando registros históricos (DAGMA, CVC, SISAIRE)...")
    
    try:
        df_raw = pd.read_csv(url_descarga, low_memory=False)
    except Exception as e:
        raise ValueError(f"❌ Error al descargar de la API: {e}")
    
    if df_raw.empty:
        raise ValueError("❌ El CSV descargado está vacío. Verifica los parámetros.")
        
    print(f"✅ Descarga completada: {len(df_raw)} registros crudos obtenidos.")
    
    # ─── 3. LIMPIEZA Y TRANSFORMACIÓN ───────────────────────────────────────
    print("🧹 Estructurando el Dataset...")
    
    columnas_utiles = {
        'med_fecha_inicio': 'fecha',
        'nombre_est': 'estacion', 
        'nombre_fgda': 'autoridad', # Guardará DAGMA, CVC o SISAIRE
        'msfl_code': 'contaminante',
        'med_concentracion_estandar': 'concentracion',
        'latitud': 'lat',
        'longitud': 'lon'
    }
    
    columnas_presentes = {k: v for k, v in columnas_utiles.items() if k in df_raw.columns}
    df = df_raw[list(columnas_presentes.keys())].rename(columns=columnas_presentes)
    
    # Limpiar y convertir formatos
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['concentracion'] = pd.to_numeric(df['concentracion'], errors='coerce')
    
    # Arreglar coordenadas que vienen con coma
    if df['lat'].dtype == object:
        df['lat'] = df['lat'].str.replace(',', '.')
    if df['lon'].dtype == object:
        df['lon'] = df['lon'].str.replace(',', '.')
        
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    # Eliminar valores corruptos o vacíos
    df = df.dropna(subset=['fecha', 'concentracion', 'lat', 'lon'])
    
    # ─── 4. FILTRO GEOESPACIAL (Bounding Box) ──────────────────────────────
    print("🗺️ Aplicando recorte espacial estricto (BBox: -76.75, 3.20 a -76.30, 3.75)...")
    # Este paso es VITAL: Descargamos toda la CVC y todo el SISAIRE de Colombia,
    # pero aquí eliminamos todo lo que no esté en Cali y Yumbo.
    df = df[
        (df['lat'] >= 3.20) & (df['lat'] <= 3.75) &
        (df['lon'] >= -76.75) & (df['lon'] <= -76.30)
    ]
    
    # ─── 5. HOMOLOGACIÓN DE CONTAMINANTES ──────────────────────────────────
    df['contaminante'] = df['contaminante'].astype(str).str.upper().str.strip()
    df['contaminante'] = df['contaminante'].replace({
        'PM 2.5': 'PM25',
        'PM2.5': 'PM25',
        'PM 10': 'PM10'
    })
    
    contaminantes_validos = ['NO2', 'SO2', 'O3']
    df = df[df['contaminante'].isin(contaminantes_validos)]
    
    # Variables de partición y análisis
    df['año'] = df['fecha'].dt.year.astype(str)
    df['mes'] = df['fecha'].dt.month.astype(str).str.zfill(2)
    
    print(f"✅ Estructura final: {len(df)} mediciones limpias.")
    
    # 🔥 REPORTE DE AUTORIDADES Y ESTACIONES
    print("\n📍 Autoridades y Estaciones encontradas en la zona:")
    resumen = df.groupby(['autoridad', 'estacion']).size().reset_index(name='mediciones')
    for index, row in resumen.iterrows():
        print(f"  - [{row['autoridad']}] {row['estacion']} ({row['mediciones']} registros)")
    
    # ─── 6. GUARDAR LOCALMENTE ─────────────────────────────────────────────
    os.makedirs(CARPETA_DESTINO, exist_ok=True)

    print(f"\n📄 Guardando CSV final en: {RUTA_LOCAL_CSV} ...")
    df.to_csv(RUTA_LOCAL_CSV, index=False, encoding='utf-8')

    print("🚀 ¡Proceso completado con éxito! Tus datos terrestres (Ground Truth) están listos.")

if __name__ == '__main__':
    extraer_datos_terrestres()