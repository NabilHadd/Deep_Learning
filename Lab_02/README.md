# Laboratorio 02 — Redes Neuronales Poco Profundas

Clasificación multilabel de deterioro cognitivo usando una red neuronal poco profunda construida en **PyTorch** e integrada con **scikit-learn**.

---

## ¿Qué hace este proyecto?

Dado un dataset de pacientes que respondieron preguntas de orientación cognitiva (¿qué día es?, ¿en qué país estás?, etc.), el objetivo es predecir el nivel de deterioro cognitivo según 6 codificaciones distintas llamadas GDS:

| Variable | Clases |
|---|---|
| `GDS` | Sin_deterioro, muy_leve, leve, moderado, moderado_alto, severo, muy_severo |
| `GDS_R1` | leve, moderado, muy_severo |
| `GDS_R2` | muy_leve, leve, severo |
| `GDS_R3` | muy_leve, severo |
| `GDS_R4` | sin_deterioro, moderado_alto, muy_severo |
| `GDS_R5` | sin_deterioro, moderado, muy_severo |

El flujo completo determina cuál de los 6 GDS es más predecible, entrena el modelo final para ese GDS y genera curvas ROC para comparar todos.

---

## Requisitos

```bash
conda env create -f enviroment.yml
conda activate lab_02
```

## Cómo ejecutar

```bash
python main.py
```

Siempre desde la raíz del repositorio.

---

## Estructura del proyecto

```
.
├── data/
│   ├── raw/
│   │   ├── dataset_deterioro.sav     # Dataset original (SPSS)
│   │   └── dataset_deterioro.csv     # Generado automáticamente
│   └── output/
│       └── scores.csv                # Resultados de la evaluación por GDS
├── src/
│   ├── config.py          # Semilla, targets, rutas, configuración de CV
│   ├── data_loader.py     # Conversión SAV→CSV y carga del DataFrame
│   ├── eda.py             # Funciones de análisis exploratorio
│   ├── evaluation.py      # Evaluación con CV anidada para todos los GDS
│   ├── models.py          # Red neuronal (PyTorch) + wrapper sklearn
│   ├── preprocessing.py   # Imputación + selección de features
│   ├── scorer_model.py    # CV anidada: inner RandomizedSearchCV + outer cross_val_score
│   ├── train_nn.py        # Entrenamiento final con RandomizedSearchCV
│   └── visualization.py  # Gráficos: histogramas, heatmap, entropía, ROC
└── main.py                # Punto de entrada
```

---

## Flujo de ejecución

### 1. Carga de datos
Convierte el archivo `.sav` a CSV y lo carga en un DataFrame.

### 2. EDA
Antes de entrenar, se realiza un análisis exploratorio:
- **Histogramas de features:** muestra la distribución de respuestas por pregunta. Una feature apilada en un solo valor tiene poca utilidad predictiva.
- **Heatmap de correlación:** detecta features redundantes. Rojo intenso = alta correlación = información repetida.
- **Entropía por GDS:** mide qué tan balanceadas están las clases en cada variable objetivo. Mayor entropía = clases más equilibradas = mejor para entrenar.

### 3. Evaluación con validación cruzada anidada
Se evalúan los 6 GDS usando validación cruzada anidada para obtener estimaciones de rendimiento **no sesgadas**:

- **Inner CV** (3 folds, 10 iteraciones): busca los mejores hiperparámetros de la red y el preprocesador.
- **Outer CV** (5 folds): evalúa qué tan bien generaliza el proceso completo a datos no vistos.

Resultado: `data/output/scores.csv` con 5 scores de Hamming Loss por cada GDS.

> Esta etapa es costosa computacionalmente. Si `scores.csv` ya existe, se puede comentar `evaluate_model(...)` en `main.py` para reutilizarlo.

El GDS con **menor Hamming Loss promedio** es el más predecible.

### 4. Curvas ROC para los 6 GDS
Se entrena un modelo por cada GDS con todos los datos y se generan curvas ROC (por clase, micro-average y macro-average). El **macro-average** es la métrica más relevante dado el desbalance de clases.

### 5. Entrenamiento final
Se entrena el modelo definitivo usando todos los datos con el mejor GDS encontrado en el paso 3.

---

## Arquitectura de la red neuronal

```
Input (n_features)
      ↓
   Linear
      ↓
 ReLU / LeakyReLU
      ↓
   Dropout
      ↓
   Linear
      ↓
Output (n_clases del GDS)
```

- Pérdida: `BCEWithLogitsLoss` (multilabel)
- Optimizador: `Adam`
- Hiperparámetros tuneables: `hidden_dim`, `dropout`, `learning_rate`, `epochs`, `batch_size`, `activation`, `weight_decay`, `threshold`

La red está envuelta en `ShallowMultiLabelNet` (hereda de `BaseEstimator` y `ClassifierMixin`) para ser compatible con pipelines y búsquedas de hiperparámetros de sklearn.

---

## Salidas

| Archivo | Descripción |
|---|---|
| `data/raw/dataset_deterioro.csv` | Dataset convertido desde SAV |
| `data/output/scores.csv` | Hamming Loss por fold y por GDS (CV anidada) |
