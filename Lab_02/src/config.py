feature_cols = {
  'DIA': 'Día',
  'MES': 'Mes',
  'AÑO': 'Año',
  'ESTACION': 'Estación',
  'PAIS': 'País',
  'CIUDAD': 'Ciudad',
  'CALLELUGAR': 'CalleLugar',
  'NUMEROPISO': 'NúmeroPiso',
  'MIGUEL2': 'Miguel2',
  'GONZALEZ2': 'Gonzalez2',
  'AVENIDA2': 'Avenida2',
  'IMPERIAL2': 'Imperial2',
  'A682': 'A682',
  'CALDERA2': 'Caldera2',
  'COPIAPO2': 'Copiapo2',
}


GDS_CLASSES = ['Sin_deterioro', 'muy_leve', 'leve', 'moderado', 'moderado_alto', 'severo', 'muy_severo']
GDS_R1_CLASSES = ['leve', 'moderado', 'muy_severo']
GDS_R2_CLASSES = ['muy_leve', 'leve', 'severo']
GDS_R3_CLASSES = ['muy_leve', 'severo']
GDS_R4_CLASSES = ['sin_deterioro', 'moderado_alto', 'muy_severo']
GDS_R5_CLASSES = ['sin_deterioro', 'moderado', 'muy_severo']


TARGETS = {
  'GDS': GDS_CLASSES,
  'GDS_R1': GDS_R1_CLASSES,
  'GDS_R2': GDS_R2_CLASSES,
  'GDS_R3': GDS_R3_CLASSES,
  'GDS_R4': GDS_R4_CLASSES,
  'GDS_R5': GDS_R5_CLASSES,
}


SAV_DATA_PATH = 'data/raw/dataset_deterioro.sav'
CSV_DATA_PATH = 'data/raw/dataset_deterioro.csv'










ejemplo_init = "¡Hola! Este es un ejemplo de variable de configuración."
