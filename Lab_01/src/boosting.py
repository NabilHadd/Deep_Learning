from scipy.stats import randint, uniform

import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from data_loader import data_loader

from sklearn.ensemble import AdaBoostClassifier

from preprocessing import preprocessors
from classifiers import classifiers
from sklearn.base import clone


def boosting(PATH, X_train, Y_train, X_test, Y_test):
    df = data_loader(PATH)
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()


    # 1. elegir mejor modelo
    row = df.sort_values(by="accuracy", ascending=False).iloc[0]

    params = json.loads(row['best_params'])


    preprocess = clone(preprocessors[row['preprocessor']]['pipeline'])
    X_train_prep = preprocess.fit_transform(X_train, Y_train)
    X_test_prep = preprocess.transform(X_test)

    base_clf = clone(classifiers[row['classifier']]['classifier'])
    base_clf.set_params(**{
        k.replace("classifier__", ""): v
        for k, v in params.items() if "classifier__" in k
    })


    # 3. boosting
    boosting_model = AdaBoostClassifier(
        estimator=base_clf,
        random_state=42
    )
    param_grid = {
        'n_estimators' : randint(50, 250),
        'learning_rate': uniform(0.01, 0.5),
    }

    random_search = RandomizedSearchCV(
        estimator=boosting_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring='accuracy',
        n_jobs=-1,
        cv=5,
        verbose=0,
        random_state=10,
        error_score='raise'
    )

    # 4. entrenar
    random_search.fit(X_train_prep, Y_train)

    # 5. evaluar
    y_pred = random_search.predict(X_test_prep)

    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(Y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(Y_test, y_pred, average='macro', zero_division=0)

    return random_search.best_estimator_, accuracy, precision, recall, f1