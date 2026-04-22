from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from preprocessing import preprocessors
from classifiers import classifiers
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import json


path = "data/outputs/models"


def view_config():
    for prep_name, prep_config in preprocessors.items():
        print(f"\n\033[31m--- Preprocesamiento: {prep_name} ---\033[0m\n")

        for clf_name, clf_config in classifiers.items():
            print(f"   Clasificador: \033[33m{clf_name}\033[0m")
            print("")


#se considera un hold out
def random_train(X_train, Y_train, X_val, Y_val):

    print("\n\033[33m=== EJECUCIÓN DE EXPERIMENTOS ===\033[0m")
    results = []
    best_models = []

    for prep_name, prep_config in preprocessors.items():
        print(f"\033[31m\n--- Preprocesamiento: {prep_name} ---\033[0m")

        for clf_name, clf_config in classifiers.items():
            print(f"\033[31m\n  Clasificador: {clf_name}\033[0m")

            # Crear pipeline combinado
            full_pipeline = Pipeline([
                ('preprocessor', prep_config['pipeline']),
                ('classifier', clf_config['classifier'])
            ])


            # Combinar parámetros
            param_grid = {}
            for key, value in prep_config['params'].items():
                param_grid['preprocessor__' + key] = value
            for key, value in clf_config['params'].items():
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
            Y_val = Y_val.ravel()
            # Entrenar y evaluar
            random_search.fit(X_train, Y_train)

            # Predecir en validación
            Y_val_pred = random_search.predict(X_val)

            # Métricas
            accuracy = accuracy_score(Y_val, Y_val_pred)
            precision = precision_score(Y_val, Y_val_pred, average='macro', zero_division=0)
            recall = recall_score(Y_val, Y_val_pred, average='macro', zero_division=0)
            f1 = f1_score(Y_val, Y_val_pred, average='macro', zero_division=0)

            # Guardar resultados
            results.append({
                "preprocessor": prep_name,
                "classifier": clf_name,
                "best_params": json.dumps(random_search.best_params_),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
            
            best_models.append({
                "name": f"{prep_name}_{clf_name}",
                "model": random_search.best_estimator_
            })
            

    return pd.DataFrame(results)


