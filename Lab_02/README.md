# Laboratorio 02 - Redes neuronales poco profundas

Este repositorio contiene el codigo del Laboratorio 02 de Deep Learning. El objetivo es entrenar y evaluar un modelo de clasificacion multietiqueta para deterioro cognitivo, donde la red neuronal fue construida con **PyTorch** y luego integrada como clasificador compatible con **scikit-learn** para aprovechar sus herramientas de experimentacion y validacion.

La orquestacion principal vive en [main.py](main.py). El flujo convierte datos desde formato `.sav`, prepara atributos, ejecuta validacion cruzada anidada para evaluar generalizacion y finalmente entrena el modelo con busqueda de hiperparametros.

## Requisitos

El proyecto usa **Conda**. Las dependencias base estan en [enviroment.yml](enviroment.yml).

```bash
conda env create -f enviroment.yml
conda activate lab02_dl
```

## Estructura del proyecto

```text
.
├── data
│   ├── output
│   │   └── scores.csv
│   └── raw
│       ├── dataset_deterioro.csv
│       └── dataset_deterioro.sav
├── enviroment.yml
├── main.py
├── README.md
└── src
    ├── config.py
    ├── data_loader.py
    ├── eda.py
    ├── evaluation.py
    ├── __init__.py
    ├── models.py
    ├── preprocessing.py
    ├── __pycache__
    │   ├── config.cpython-314.pyc
    │   ├── data_loader.cpython-314.pyc
    │   ├── evaluation.cpython-314.pyc
    │   ├── __init__.cpython-314.pyc
    │   ├── models.cpython-314.pyc
    │   ├── preprocessing.cpython-314.pyc
    │   ├── scorer_model.cpython-314.pyc
    │   └── train_nn.cpython-314.pyc
    ├── scorer_model.py
    ├── train_nn.py
    ├── uncertainly.py
    └── visualization.py

```

## Como se ejecuta

Desde la raiz del proyecto:

```bash
python main.py
```

## Flujo principal en `main.py`

La ejecucion en [main.py](main.py) sigue este orden:

### 1. Conversion y carga de datos

En [src/data_loader.py](src/data_loader.py):

- `sav_to_csv()` convierte `data/raw/dataset_deterioro.sav` a CSV.
- `load_csv()` carga el CSV para el resto del pipeline.

### 2. Evaluacion con validacion cruzada anidada

En [src/evaluation.py](src/evaluation.py) se llama `evaluate_model(...)` para evaluar los targets:

- `GDS`
- `GDS_R1`
- `GDS_R2`
- `GDS_R3`
- `GDS_R4`
- `GDS_R5`

El scoring se guarda en `data/output/scores.csv`.

### 3. Entrenamiento final

Despues de la evaluacion, [main.py](main.py) vuelve a entrenar el modelo con todos los datos usando [src/train_nn.py](src/train_nn.py).

## PyTorch para la red neuronal

La arquitectura neuronal esta en [src/models.py](src/models.py) como `MyNN`, una red poco profunda implementada con `torch.nn.Module`:

- capa lineal de entrada (`nn.Linear`),
- activacion configurable (`ReLU` o `LeakyReLU`),
- `Dropout` para regularizacion,
- capa de salida lineal para logits.

Durante entrenamiento se usa:

- `BCEWithLogitsLoss` para el problema multietiqueta,
- optimizador `Adam`,
- mini-batches y multiples epocas configurables.

Esto asegura que el modelo central del laboratorio sea una red neuronal real construida y entrenada en PyTorch.

## Integrar la red de torch con scikit-learn

La clave del laboratorio es que la red de PyTorch no se usa de forma aislada. En [src/models.py](src/models.py), `ShallowMultiLabelNet` hereda de `BaseEstimator` y `ClassifierMixin`, exponiendo interfaz tipo sklearn:

- `fit(X, y)`
- `predict_proba(X)`
- `predict(X)`

Gracias a eso, se puede usar todo el ecosistema de scikit-learn:

- `Pipeline` de preprocesamiento + clasificador ([src/preprocessing.py](src/preprocessing.py), [src/train_nn.py](src/train_nn.py));
- `RandomizedSearchCV` para sintonizar hiperparametros de preprocessing y red en una sola busqueda ([src/train_nn.py](src/train_nn.py));
- `cross_val_score` con esquema anidado (`inner_cv` + `outer_cv`) para medir generalizacion de forma robusta ([src/scorer_model.py](src/scorer_model.py), [src/config.py](src/config.py)).

En otras palabras: **la red la construye torch, pero la estrategia experimental y de validacion se apalanca en sklearn**.

## Preprocesamiento y targets

El preprocesamiento en [src/preprocessing.py](src/preprocessing.py) incluye:

- imputacion de faltantes con `SimpleImputer`;
- seleccion de atributos con `SelectKBest(chi2)`.

Para las salidas, `one_hot_encode(...)` transforma cada target a formato multietiqueta compatible con la funcion de perdida usada en PyTorch.

## Configuracion central

En [src/config.py](src/config.py) se concentran:

- semilla aleatoria (`RANDOM_SEED`),
- estrategia de scoring (`hamming_loss`),
- particiones de validacion cruzada:
	- `inner_cv`: busqueda de hiperparametros,
	- `outer_cv`: evaluacion externa.

Esto permite mantener consistencia y reproducibilidad en todo el laboratorio.

## Salidas esperadas

Dependiendo de la ejecucion, se generan o actualizan:

- `data/raw/dataset_deterioro.csv` (conversion desde SAV),
- `data/output/scores.csv` (resultados por fold y por target).

`scores.csv` resume el desempeno del pipeline completo (preprocesamiento + red en torch envuelta en sklearn) para cada variable objetivo.

## Notas de uso

- Ejecutar siempre desde la raiz del repositorio.
- Verificar que `torch` este disponible en el entorno antes de correr `main.py`.
- El tiempo de ejecucion puede ser alto por la validacion cruzada anidada y la busqueda de hiperparametros.
- Si ya existe `scores.csv`, puedes reutilizarlo para analisis sin volver a correr toda la evaluacion.

