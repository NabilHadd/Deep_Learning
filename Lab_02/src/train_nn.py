
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .preprocessing import preprocessor
from .models import nn_config
from .config import RANDOM_SEED, inner_cv, hamming_scorer


def train_shallow_nn(X, Y):


  full_pipeline = Pipeline([
    ('preprocessor', preprocessor['pipeline']),
    ('classifier', nn_config['classifier'])
  ])

    # Combinar parámetros
  param_grid = {}
  for key, value in preprocessor['params'].items():
    param_grid['preprocessor__' + key] = value
  for key, value in nn_config['params'].items():
    param_grid['classifier__' + key] = value


    #busqueda
  final_model = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_grid,
    n_iter=10,
    scoring=hamming_scorer,
    n_jobs=-1,
    cv= inner_cv,
    verbose=0,
    random_state=RANDOM_SEED,
  )

  return final_model.fit(X, Y)
