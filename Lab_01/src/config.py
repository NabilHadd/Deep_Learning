from pathlib import Path
from stacking import stacking
from bagging import bagging
from boosting import boosting
from enum import Enum

GDS_CLASSES = ['Sin_deterioro', 'muy_leve', 'leve', 'moderado', 'moderado-alto', 'severo', 'muy severo']
GDS_R2_CLASSES = ['muy-leve', 'leve', 'severo']

# BASE PATH
BASE_PATH = Path(".")

RAW_DATA_PATH       = BASE_PATH / "data/raw/15 atributos R0-R5.sav"
DATA_PATH           = BASE_PATH / "data/processed/data.csv"
GDS_RESULTS_PATH    = BASE_PATH / "data/outputs/gds_output.csv"
GDS_R2_RESULTS_PATH = BASE_PATH / "data/outputs/gds_r2_output.csv"

SAVE_IMAGES_PATH    = BASE_PATH / "data/outputs/images"

# ENUM
class Model_names(Enum):
    STACKING_NAME   = "Stacking_model"
    BOOSTING_NAME   = "Boosting_model"
    BAGGING_NAME    = "Bagging_model"
    OWN_NAIVE_NAYES = "Own_naive_Bayes"

# MODELOS
MODELS = {
    Model_names.STACKING_NAME: stacking,
    Model_names.BOOSTING_NAME: boosting,
    Model_names.BAGGING_NAME: bagging
}