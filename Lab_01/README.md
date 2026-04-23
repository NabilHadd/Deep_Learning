# Laboratorio 01 - Clasificacion GDS

Este repositorio contiene el codigo del Laboratorio 01 de Deep Learning. El objetivo es entrenar, evaluar y comparar modelos clasicos de Machine Learning para clasificar deterioro cognitivo usando las etiquetas **GDS** y **GDS_R2**.

El flujo principal esta implementado en `src/main.py` e incluye:

- carga de datos desde un archivo `.sav`;
- analisis exploratorio basico;
- generacion o reutilizacion del dataset procesado;
- entrenamiento con busqueda aleatoria de hiperparametros;
- construccion de modelos de ensamble;
- generacion de curvas ROC y matrices de confusion.

## Requisitos

El proyecto usa **Conda**. Las librerias necesarias estan declaradas en `environment.yml`.

Para crear y activar el entorno:

```bash
conda env create -f environment.yml
conda activate lab01_dl
```

El entorno instala Python 3.10 y las dependencias principales del proyecto, incluyendo `pandas`, `matplotlib`, `pyreadstat` y `scikit-learn`.

## Estructura del proyecto

```text
.
├── data/
│   ├── raw/
│   │   └── 15 atributos R0-R5.sav
│   ├── processed/
│   │   └── data.csv
│   └── outputs/
│       ├── gds_output.csv
│       ├── gds_r2_output.csv
│       └── images/
├── notebooks/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── classifiers.py
│   ├── train.py
│   ├── stacking.py
│   ├── bagging.py
│   ├── boosting.py
│   └── visualization.py
├── environment.yml
└── README.md
```

## Como ejecutar

Desde la raiz del proyecto, con el entorno Conda activado:

```bash
python src/main.py
```

Las rutas usadas por el codigo estan definidas en `src/config.py`:

- datos originales: `data/raw/15 atributos R0-R5.sav`;
- datos procesados: `data/processed/data.csv`;
- resultados GDS: `data/outputs/gds_output.csv`;
- resultados GDS_R2: `data/outputs/gds_r2_output.csv`;
- imagenes generadas: `data/outputs/images/`.

## Importante: secciones que se pueden comentar

Algunas partes del codigo estan pensadas para ejecutarse solo la primera vez o cuando se quieran regenerar resultados. Esto evita volver a correr procesos costosos, especialmente el entrenamiento con `RandomizedSearchCV`.

### 1. Regenerar el CSV procesado

En `src/main.py` aparece esta linea:

```python
# eda.make_csv(df_15_raw, DATA_PATH)
```

Esta linea genera `data/processed/data.csv` desde el archivo original `.sav`. Como el CSV procesado ya esta guardado, normalmente debe quedar comentada.

Solo se debe descomentar si:

- se modifica el archivo original en `data/raw/`;
- se quiere reconstruir `data/processed/data.csv`;
- se elimina el CSV procesado.

### 2. Reentrenar modelos base

Estas secciones ejecutan la busqueda aleatoria de hiperparametros para GDS y GDS_R2:

```python
# PARA GDS
results_dataframe = tr.random_train(X_TRAIN, Y_TRAIN_GDS, X_TEST, Y_TEST_GDS)
eda.make_csv(results_dataframe[0], GDS_RESULTS_PATH, idx=True)

# PARA GDS_R2
results_dataframe = tr.random_train(X_TRAIN, Y_TRAIN_GDS_R2, X_TEST, Y_TEST_GDS_R2)
eda.make_csv(results_dataframe[0], GDS_R2_RESULTS_PATH, idx=True)
```

Este bloque puede tardar bastante porque usa `RandomizedSearchCV` con validacion cruzada `LeaveOneOut`.

Si ya existen estos archivos:

- `data/outputs/gds_output.csv`;
- `data/outputs/gds_r2_output.csv`;

entonces no es necesario volver a ejecutar el entrenamiento base. Se puede comentar ese bloque y reutilizar los resultados guardados.

El codigo posterior usa esos CSV para reconstruir y entrenar los ensambles:

```python
gds_ensamble_models = {
    model_name: MODELS[model_name](GDS_RESULTS_PATH, X_train=X_TRAIN, Y_train=Y_TRAIN_GDS, X_test=X_TEST, Y_test=Y_TEST_GDS)
    for model_name in Model_names
}
```

Por eso, antes de comentar el entrenamiento base, debe verificarse que los archivos `gds_output.csv` y `gds_r2_output.csv` existan.

## Modelos y preprocesamientos

Los preprocesamientos estan definidos en `src/preprocessing.py`:

- filtro por correlacion de Pearson;
- PCA con estandarizacion.

Los clasificadores base estan definidos en `src/classifiers.py`:

- Naive Bayes;
- Random Forest;
- Regresion Logistica;
- SVM.

Los ensambles estan definidos en:

- `src/stacking.py`;
- `src/bagging.py`;
- `src/boosting.py`.

## Salidas esperadas

Al ejecutar el pipeline se generan o reutilizan:

- `data/processed/data.csv`;
- `data/outputs/gds_output.csv`;
- `data/outputs/gds_r2_output.csv`;
- imagenes en `data/outputs/images/GDS/`;
- imagenes en `data/outputs/images/GDS_R2/`.

Las imagenes incluyen curvas ROC multiclase y matrices de confusion, dependiendo del modelo y etiqueta evaluada.

## Notas de uso

- Ejecutar siempre desde la raiz del repositorio para que las rutas relativas funcionen correctamente.
- Mantener activado el entorno `lab01_dl` antes de correr el codigo.
- Si se quieren resultados completamente nuevos, descomentar las secciones de generacion de CSV y entrenamiento.
- Si solo se quieren reutilizar resultados ya generados, dejar comentada la generacion del CSV procesado y comentar el reentrenamiento base.
