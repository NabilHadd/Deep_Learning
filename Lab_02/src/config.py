from sklearn.metrics import hamming_loss, make_scorer
from sklearn.model_selection import KFold

hamming_scorer = make_scorer(
    hamming_loss,
    greater_is_better=False
)


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

RANDOM_SEED = 42


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
SCORES_CSV_PATH = 'data/output/scores.csv'


#Cross validation anidado:
#5 folds ecternos
#3 folds internos para la busqueda de hiperparametros

#se podria probar con multilabelstratifiedkfold

outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)














ejemplo_init = "¡Hola! Este es un ejemplo de variable de configuración."
