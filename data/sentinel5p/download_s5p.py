import ee
import time
from datetime import datetime, timedelta

ee.Authenticate() 
ee.Initialize(project='proyecto-analitica-3')

CALI_BBOX = ee.Geometry.Rectangle([-76.60, 3.30, -76.40, 3.55])
YEAR      = '2020'
DRIVE_DIR = 'geovision_s5p_daily'
SCALE     = 1000

CONTAMINANTES = {
    'NO2': {
        'collection': 'COPERNICUS/S5P/OFFL/L3_NO2',
        'band':       'tropospheric_NO2_column_number_density',
    },
    'SO2': {
        'collection': 'COPERNICUS/S5P/OFFL/L3_SO2',
        'band':       'SO2_column_number_density',
    },
    'O3': {
        'collection': 'COPERNICUS/S5P/OFFL/L3_O3',
        'band':       'O3_column_number_density',
    },
}

def generar_fechas(year):
    """Genera lista de fechas diarias para el año."""
    inicio = datetime(int(year), 1, 1)
    fin    = datetime(int(year), 12, 31)
    fechas = []
    actual = inicio
    while actual <= fin:
        fechas.append(actual.strftime('%Y-%m-%d'))
        actual += timedelta(days=1)
    return fechas

def exportar_dia(nombre_contaminante, config, fecha):
    """Exporta todas las órbitas disponibles de un día dado."""
    fecha_siguiente = (datetime.strptime(fecha, '%Y-%m-%d') 
                       + timedelta(days=1)).strftime('%Y-%m-%d')
    
    coleccion = (
        ee.ImageCollection(config['collection'])
        .filterBounds(CALI_BBOX)
        .filterDate(fecha, fecha_siguiente)
        .select(config['band'])
    )

    n = coleccion.size().getInfo()
    if n == 0:
        return None  

    imagen = coleccion.first().clip(CALI_BBOX)
    nombre_archivo = f'S5P_{nombre_contaminante}_{fecha}'

    tarea = ee.batch.Export.image.toDrive(
        image          = imagen,
        description    = nombre_archivo,
        folder         = DRIVE_DIR,
        fileNamePrefix = nombre_archivo,
        region         = CALI_BBOX,
        scale          = SCALE,
        crs            = 'EPSG:4326',
        fileFormat     = 'GeoTIFF',
        maxPixels      = 1e9,
    )
    tarea.start()
    return tarea

# ── Ejecutar ───────────────────────────────────────────────────
fechas = generar_fechas(YEAR)
tareas = []
tareas_fallidas = 0

for nombre, config in CONTAMINANTES.items():
    print(f'\n── {nombre} ({len(fechas)} días) ──')
    for fecha in fechas:
        tarea = exportar_dia(nombre, config, fecha)
        if tarea:
            tareas.append(tarea)
            print(f'  ✓ {fecha}')
        else:
            tareas_fallidas += 1
        time.sleep(0.5)  

print(f'\nTareas enviadas : {len(tareas)}')
print(f'Días sin cobertura: {tareas_fallidas}')
print(f'Progreso en: https://code.earthengine.google.com/tasks')