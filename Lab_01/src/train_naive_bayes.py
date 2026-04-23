import pandas as pd
from scipy.stats import randint, uniform

import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from data_loader import data_loader

from preprocessing import preprocessors
from sklearn.base import clone
from naive_bayes import NaiveBayesClassifier


naive_bayes_config = {
        'classifier': NaiveBayesClassifier(fit_prior=True),
        'params': {
            'classifier__alpha': uniform(1, 0.5)
        }
    }


def train_own_naive_bayes(X_train, Y_train):


    for _, prep_config in preprocessors.items():

        full_pipeline = Pipeline([
            ('preprocessor', prep_config['pipeline']),
            ('classifier', naive_bayes_config['classifier'])
        ])

        # Combinar parámetros
        param_grid = {}
        for key, value in prep_config['params'].items():
            param_grid['preprocessor__' + key] = value
        for key, value in naive_bayes_config['params'].items():
            param_grid[key] = value


            #busqueda
        random_search = RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=param_grid,
            n_iter=10,
            scoring='accuracy',
            n_jobs=-1,
            cv= LeaveOneOut(),
            verbose=0,
            random_state=10,
            error_score='raise'
        )

        Y_train = Y_train.ravel()

        random_search.fit(X_train, Y_train)

    return random_search.best_estimator_
