from scipy.stats import randint

import pandas as pd
import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from data_loader import data_loader

from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier

from preprocessing import preprocessors
from classifiers import classifiers
from sklearn.base import clone


def bagging(PATH, X_train, Y_train, X_test, Y_test):
    df = data_loader(PATH)
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()

    # 1. Elegir mejor modelo (ej: mayor accuracy)
    row = df.sort_values(by="recall", ascending=False).iloc[0]

    params = json.loads(row['best_params'])

    # 2. Construir modelo base
    base_model = Pipeline([
        ('preprocessor', clone(preprocessors[row['preprocessor']]['pipeline'])),
        ('classifier', clone(classifiers[row['classifier']]['classifier']))
    ])

    base_model.set_params(**params)

    full_pipeline = Pipeline([
        ('classifier', BaggingClassifier(estimator=base_model, n_estimators=10, random_state=42, n_jobs=-1))
    ])

    param_grid = {
        'classifier__n_estimators' : randint(100, 500)
    }

    random_search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='accuracy',
        n_jobs=-1,
        cv= 5,
        verbose=0,
        random_state=10,
        error_score='raise'
    )

    # 4. Entrenar
    random_search.fit(X_train, Y_train)

    # 5. Evaluar
    y_pred = random_search.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(Y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(Y_test, y_pred, average='macro', zero_division=0)

    return random_search.best_estimator_, accuracy, precision, recall, f1