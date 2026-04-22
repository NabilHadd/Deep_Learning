
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import randint, uniform


# NOTESE QUE:
# classifiers es un diccionario con la siguiente configuracion:
#  Classifiers:
#       -"clasificador":
#              -Objeto del modelo.
#       -"parametros":
#              -"Hiperparametros del clasificador":
#                   -lista de hiperparametros
#              -"otro":
#                   -otras especificaciones sobre
#                    los hiperparametros como kernel.
#



classifiers = {

    'NaiveBayes': {
        'classifier': GaussianNB(),
        'params': {
            'classifier__var_smoothing': uniform(1e-10, 1e-5)
        }
    },

    'RandomForest': {
        'classifier': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': randint(50, 200),
            'classifier__max_depth': randint(3, 30),
            'classifier__min_samples_split': randint(2, 15)
        }
    },

    'LogisticRegression': {
        'classifier': LogisticRegression(random_state=42),
        'params': {
            'classifier__max_iter': randint(200, 1000),
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': uniform(0.01, 5),
            'classifier__solver': ['saga']
        }
    },

    'SVM': {
        'classifier': SVC(random_state=42, class_weight='balanced', probability=True),
        'params': {
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__C': uniform(0.1, 10),
            'classifier__gamma': ['scale', 'auto']
        }
    }
}