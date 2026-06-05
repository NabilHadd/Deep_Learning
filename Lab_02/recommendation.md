# Recomendaciones del repositorio

## Observaciones generales

### 1. Bug de importación en `visualization.py`
**Problema:** La línea `from eda import entropy_table` fallaba al ejecutar desde la raíz del proyecto porque Python no encontraba el módulo `eda` como módulo de nivel superior.  
**Corrección aplicada:** Cambiado a `from src.eda import entropy_table`, consistente con el resto del proyecto.

---

### 2. `plot_roc_curves` no existía
**Problema:** El código comentado en `main.py` sugería construir curvas ROC, y las instrucciones del laboratorio describen una función `plot_roc_curves(y_true, y_score, class_names, gds_name, save_path=None)`. Sin embargo, solo existía `plot_roc_multiclass`, que tiene una firma incompatible: recibe el modelo y etiquetas 1D (multiclase), no arrays multilabel directamente.  
**Corrección aplicada:** Se agregó `plot_roc_curves` en `visualization.py` con la firma correcta para clasificación multilabel (curvas por clase, micro-average y macro-average).

---

### 3. Sin partición train/test
**Problema:** Todo el dataset se usa para entrenar (`train_shallow_nn(X=X, Y=Y)`) y luego se evalúa sobre los mismos datos (`y_pred = final_model.predict(X)`). Las métricas obtenidas (Hamming Loss, curvas ROC) reflejan el ajuste sobre entrenamiento, no la capacidad de generalización.  
**Recomendación:** Agregar una partición `train_test_split` antes del entrenamiento final y evaluar sobre el conjunto de test. Esto haría las curvas ROC representativas del rendimiento real.

---

### 4. Columna `ID` incluida en features
**Problema:** `X = df.drop(columns=list(TARGETS.keys()))` elimina los 6 targets GDS pero conserva la columna `ID`, que es un índice de fila (1, 2, 3, …) sin valor predictivo. El modelo podría aprender correlaciones espurias con ese campo.  
**Recomendación:** Excluir `ID` explícitamente: `X = df.drop(columns=list(TARGETS.keys()) + ['ID'])`.

---

### 5. `evaluate_model` comentada
**Observación:** La evaluación con validación cruzada anidada (`evaluate_model`) está comentada en `main.py`. Cada ejecución reutiliza el archivo `data/output/scores.csv` existente. Esto es intencional para ahorrar tiempo de cómputo, pero si el dataset cambia, hay que recordar descomentar esa línea para regenerar los scores.

---

### 6. `frec_table.dropna()` no modifica in-place
**Problema menor:** En `eda.py`, la línea `frec_table.dropna()` no tiene efecto porque pandas no modifica el DataFrame en su lugar por defecto.  
**Corrección sugerida:** Cambiar a `frec_table = frec_table.dropna()` o `frec_table.dropna(inplace=True)`.

---

### 7. Nombre del entorno conda no coincide
**Observación:** El archivo `enviroment.yml` define `name: lab_02`, pero el README indica `conda activate lab02_dl`. El comando correcto para activar el entorno es `conda activate lab_02`.

---

## Compatibilidad `visualization.py` → `main.py`

| Función | Estado | Descripción |
|---|---|---|
| `frec_plot(df)` | Compatible | Funciona directamente con el DataFrame cargado |
| `entropy_plot(df)` | Compatible (tras fix de import) | Requería corregir `from eda import` → `from src.eda import` |
| `plot_roc_curves(y_true, y_score, ...)` | Agregada | Nueva función multilabel; compatible con la salida de `predict_proba` de `ShallowMultiLabelNet` |
| `plot_roc_multiclass(model, ...)` | No usada | Firma incompatible para multilabel; queda disponible para uso futuro con un solo target 1D |
