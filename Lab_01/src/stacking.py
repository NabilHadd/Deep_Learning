import pandas as pd
import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_loader import data_loader

from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocessors
from classifiers import classifiers
from sklearn.base import clone


def stacking(PATH, X_train, Y_train, X_test, Y_test):
    df = data_loader(PATH)
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()

    estimators = []

    for i, row in df.iterrows():

        params = json.loads(row['best_params'])

        model = Pipeline([
            ('preprocessor', clone(preprocessors[row['preprocessor']]['pipeline'])),
            ('classifier', clone(classifiers[row['classifier']]['classifier']))
        ])

        model.set_params(**params)

        name = f"{row['preprocessor']}_{row['classifier']}_{i}"
        estimators.append((name, model))


    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        stack_method='predict_proba'
    )

    stacking_model.fit(X_train, Y_train)

    y_pred = stacking_model.predict(X_test)
    # Métricas
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(Y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(Y_test, y_pred, average='macro', zero_division=0)

    return stacking_model, accuracy, precision, recall, f1
