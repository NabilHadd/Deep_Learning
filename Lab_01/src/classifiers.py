from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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

    # Naive Bayes: modelo probabilístico, no tiene muchos hiperparámetros
    # GaussianNB asume distribución normal de los datos
    'NaiveBayes': {
        'classifier': GaussianNB(),
        'params': {
            'var_smoothing': [1e-7, 1e-6] #estabiliza la varianza de variables con poca. vuelve el modelo mas suave.
        }
    },

    # Random Forest: conjunto de árboles, reduce overfitting
    'RandomForest': {
        'classifier': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, None],
            'classifier__min_samples_split': [2, 10]
        }
    },

    # Logistic Regression: regularización controla complejidad del modelo
    'LogisticRegression': {
        'classifier': LogisticRegression(random_state=42, max_iter=200),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.1, 1], #inverso a la regularización.
            'classifier__solver': ['liblinear']
        }
    },

    # SVM: margen máximo entre clases
    'SVM': {
        'classifier': SVC(random_state=42),
        'params': {
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__C': [0.1, 1], #C mas grande, mas penalizacion.
            'classifier__gamma': ['scale']  # solo afecta rbf, no se usara auto debido a que son clases desbalanceadas y scale considera la varianza para calcular el gamma.
        }
    }
}