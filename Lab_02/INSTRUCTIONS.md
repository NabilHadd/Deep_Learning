# Instructions for Claude Code

Este documento describe las tareas a realizar sobre el repositorio de laboratorio de deep learning.
El proyecto implementa un clasificador multilabel basado en redes neuronales poco profundas,
donde se entrena una red neuronal independiente por cada grupo de datos (GDS).

---

## Contexto del proyecto

- El archivo de entrada principal es `main.py`. Ahí comienza la ejecución.
- Existe un módulo `visualization` con funciones ya implementadas para:
  - Visualizar la distribución de clases por conjunto GDS.
  - Visualizar la entropía asociada a cada conjunto GDS.
- Se entrena una red neuronal por cada GDS (clasificación multilabel).
- El README del repositorio contiene el resumen de la arquitectura y el flujo actual ES MUY IMPORTANTE QUE LO REVISES ANTES DE HACER CUALQUIER COSA.

---

## Tarea 1 — EDA al comienzo de `main`

Al inicio de la función `main`, agregar un bloque de Análisis Exploratorio de Datos (EDA) **antes** de
cualquier paso de entrenamiento. El EDA debe incluir como mínimo las siguientes visualizaciones y métricas:

### Distribución general del dataset
- Cantidad total de muestras y cómo se distribuyen entre los conjuntos GDS.
- Proporción de cada clase (label) a nivel global y por GDS.
- Identificar si existe desbalance de clases y cuán severo es.

### Estadísticas descriptivas de los features
- Media, desviación estándar, mínimo y máximo por feature.
- Histogramas de distribución de cada feature.
- Heatmap de correlación entre features para detectar redundancias.

### Llamadas a funciones ya existentes en `visualization`
Invocar las funciones ya implementadas dentro del bloque EDA para consolidar toda la exploración en un solo lugar:
- La función que visualiza la distribución de clases por GDS.
- La función que visualiza la entropía por GDS.

Existe un archivo en src llamado eda.py para que lo revises y completes en caso de ser necesario. EL eda no debe ser muy complejo.

---

## Tarea 2 — Curva ROC por GDS

### Diagnóstico previo
Antes de implementar, revisar la configuración actual del entrenamiento de cada red neuronal por GDS y verificar:

```python
def plot_roc_curves(y_true, y_score, class_names, gds_name, save_path=None):
    """
    Genera curvas ROC micro/macro y por clase para un GDS dado.
    y_true: array (n_samples, n_classes) con etiquetas reales.
    y_score: array (n_samples, n_classes) con probabilidades predichas.
    class_names: lista de strings con el nombre de cada clase.
    gds_name: string identificador del GDS (para el título).
    save_path: ruta opcional para guardar la figura.
    """
```

- Llamar a esta función desde `main` después del entrenamiento de cada red, pasando los datos
  del conjunto de test/validación.

---

## Resumen de cambios esperados

| Archivo | Cambio |
|---|---|
| `main.py` | Agregar bloque EDA al inicio de `main` con todas las visualizaciones descritas en Tarea 1. Agregar llamadas a `plot_roc_curves` tras el entrenamiento de cada GDS. |
| `visualization.py` | revisar funcion `plot_roc_curves`. |
| *(diagnóstico)* | Revisar de que manera se entrena cada red neuronal y evaluar la compatibilidad con la funcion plot_roc_curves (Tarea 2). |


Principalmente se espera que generes un documento recomendation.md donde puedas describir tus recomendaciones de manera muy simple y comprensible con respecto a lo que hayas observado en el repositorio, ademas de la implementación de un EDA simple y la compatibilización de visualization.py en main.py siempre y cuando sea compatible.
---