# Laboratorio 01 - Clasificacion GDS

Este repositorio contiene el codigo del Laboratorio 01 de Deep Learning. El objetivo es entrenar, evaluar y comparar modelos clasicos de Machine Learning para clasificar deterioro cognitivo usando las etiquetas **GDS** y **GDS_R2**.

El flujo principal vive en [src/main.py](/Users/Tomas/Documents/Universidad/Deep%20learning/Deep_Learning/Lab_01/src/main.py) y esta organizado por secciones. La idea del archivo es simple: se dejan comentadas las partes costosas o que solo se ejecutan una vez, y se descomenta un bloque solo cuando realmente se necesita correrlo.

## Requisitos

El proyecto usa **Conda**. Las dependencias estan definidas en `environment.yml`.

```bash
conda env create -f environment.yml
conda activate lab01_dl
```

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
│   ├── naive_bayes.py
│   ├── train.py
│   ├── train_naive_bayes.py
│   ├── stacking.py
│   ├── bagging.py
│   ├── boosting.py
│   └── visualization.py
├── environment.yml
└── README.md
```

## Como se ejecuta

Desde la raiz del proyecto:

```bash
python src/main.py
```

La ejecucion esta pensada para controlarse directamente desde las secciones comentadas de `main.py`.

## Flujo por secciones en `main.py`

`src/main.py` esta dividido en cuatro partes. La dinamica esperada es ir descomentando solo lo que se quiere ejecutar.

### 1. Primera parte: carga inicial, analisis y generacion del CSV procesado

Esta seccion:

- lee el archivo original `.sav`;
- revisa outliers y valores faltantes;
- analiza distribuciones basicas;
- genera `data/processed/data.csv`.

Normalmente queda comentada, porque ese CSV ya viene generado. Solo conviene descomentarla si se modifico el archivo original o si se quiere reconstruir el dataset procesado.

### 2. Segunda parte: lectura del CSV procesado y particion de datos

Esta es la unica seccion que debe permanecer sin comentar. Su funcion es:

- cargar `data/processed/data.csv`;
- definir los labels a partir de `GDS`;
- graficar frecuencias y entropia;
- separar atributos y etiquetas;
- hacer el `train_test_split` para `GDS` y `GDS_R2`.

El resto del pipeline depende de esta etapa, porque aqui se construyen `X_TRAIN`, `X_TEST`, `Y_TRAIN_GDS`, `Y_TEST_GDS`, `Y_TRAIN_GDS_R2` y `Y_TEST_GDS_R2`.

### 3. Tercera parte: entrenamiento de modelos base

Esta seccion ejecuta `RandomizedSearchCV` sobre las combinaciones de preprocesamiento y clasificadores base, y luego guarda los resultados en:

- `data/outputs/gds_output.csv`
- `data/outputs/gds_r2_output.csv`

Este bloque suele dejarse comentado porque es costoso. Solo debe descomentarse cuando se quiera volver a entrenar los modelos base y regenerar esos CSV.

### 4. Cuarta parte: ensambles, Own Naive Bayes y visualizaciones

Esta seccion:

- reconstruye y entrena los modelos de ensamble a partir de los CSV de resultados;
- entrena el modelo `Own_naive_Bayes` para `GDS_R2`;
- genera curvas ROC;
- genera matrices de confusion para `GDS_R2`.

Tambien suele permanecer comentada y se descomenta cuando se quiere regenerar figuras o volver a construir los modelos finales.

## Orden recomendado de uso

1. Dejar activa la segunda parte.
2. Descomentar la primera parte solo si hace falta regenerar `data/processed/data.csv`.
3. Descomentar la tercera parte solo si hace falta recalcular `gds_output.csv` y `gds_r2_output.csv`.
4. Descomentar la cuarta parte cuando ya existan esos CSV y se quieran construir ensambles, entrenar el `Own Naive Bayes` y generar las visualizaciones.

## Own Naive Bayes

Se agrego un nuevo modelo propio en [src/naive_bayes.py](/Users/Tomas/Documents/Universidad/Deep%20learning/Deep_Learning/Lab_01/src/naive_bayes.py): `NaiveBayesClassifier`.

Esta implementacion hereda de `ClassifierMixin` y `BaseEstimator` de `sklearn`, lo que permite integrarla naturalmente con herramientas del ecosistema de scikit-learn, por ejemplo:

- uso dentro de `Pipeline`;
- compatibilidad con `RandomizedSearchCV`;
- soporte de `predict`;
- soporte de `predict_proba`, necesario para curvas ROC;
- manejo consistente de `fit` y validaciones internas.

El entrenamiento de este modelo se realiza en [src/train_naive_bayes.py](/Users/Tomas/Documents/Universidad/Deep%20learning/Deep_Learning/Lab_01/src/train_naive_bayes.py).

### Parametros principales

- `alpha`: controla el suavizado de Laplace para evitar probabilidades nulas en valores no observados.
- `fit_prior`: permite estimar la probabilidad a priori de cada clase segun su frecuencia en el dataset. Esto es especialmente util cuando las clases estan desbalanceadas.

## Rol de `config.py`

El archivo [src/config.py](/Users/Tomas/Documents/Universidad/Deep%20learning/Deep_Learning/Lab_01/src/config.py) concentra configuracion central del proyecto:

- rutas de entrada y salida;
- nombres de clases para `GDS` y `GDS_R2`;
- un `Enum` llamado `Model_names`;
- un diccionario `MODELS` que asocia cada valor del enum con su constructor real.

Esto permite reconocer de forma consistente los distintos modelos y evitar a toda costa los "magic strings" repartidos por el codigo. En vez de depender de nombres escritos manualmente en varios lugares, se usa una fuente unica de verdad para referenciar stacking, bagging, boosting y el `Own Naive Bayes`.

## Modelos y preprocesamientos

Los preprocesamientos estan definidos en [src/preprocessing.py](/Users/Tomas/Documents/Universidad/Deep%20learning/Deep_Learning/Lab_01/src/preprocessing.py):

- filtro por correlacion de Pearson;
- PCA con estandarizacion.

Los clasificadores base estan definidos en [src/classifiers.py](/Users/Tomas/Documents/Universidad/Deep%20learning/Deep_Learning/Lab_01/src/classifiers.py):

- Naive Bayes de `sklearn`;
- Random Forest;
- Regresion Logistica;
- SVM.

Los ensambles estan definidos en:

- [src/stacking.py](/Users/Tomas/Documents/Universidad/Deep%20learning/Deep_Learning/Lab_01/src/stacking.py)
- [src/bagging.py](/Users/Tomas/Documents/Universidad/Deep%20learning/Deep_Learning/Lab_01/src/bagging.py)
- [src/boosting.py](/Users/Tomas/Documents/Universidad/Deep%20learning/Deep_Learning/Lab_01/src/boosting.py)

## Salidas esperadas

Dependiendo de las secciones activadas, el proyecto genera o reutiliza:

- `data/processed/data.csv`
- `data/outputs/gds_output.csv`
- `data/outputs/gds_r2_output.csv`
- imagenes en `data/outputs/images/GDS/`
- imagenes en `data/outputs/images/GDS_R2/`

## Notas de uso

- Ejecutar siempre desde la raiz del repositorio.
- Mantener activa la segunda seccion de `main.py`, ya que las demas dependen de ella.
- No descomentar bloques por defecto: la idea del archivo es habilitar solo la parte que se necesita ejecutar.
- Si ya existen los CSV de salida, no hace falta repetir el entrenamiento base.
