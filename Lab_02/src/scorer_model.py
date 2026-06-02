
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from .preprocessing import preprocessor
from sklearn.base import clone
from .models import ShallowMultiLabelNet, nn_config
from .config import RANDOM_SEED, inner_cv, outer_cv, hamming_scorer



#Hay que pensar en como aplicar el cross alidation en un esquema anidado.
def scorer_shallow_nn(X, Y):

    

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor['pipeline']),   # único
        ('classifier', nn_config['classifier'])
    ])

    param_grid = {}

    for key, value in nn_config['params'].items():
        param_grid['classifier__' + key] = value
    for key, value in preprocessor['params'].items():
        param_grid['preprocessor__' + key] = value

    random_search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring= hamming_scorer,
        n_jobs=-1,
        cv=inner_cv,
        verbose=0,
        random_state=RANDOM_SEED,
        error_score='raise'
    )

    scores = cross_val_score(
        random_search,
        X,
        Y,
        cv=outer_cv
    )

    return scores