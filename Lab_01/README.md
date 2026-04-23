# Proyecto de Clasificación GDS (Laboratorio 01 - Deep Learning)

Este proyecto tiene como objetivo entrenar, evaluar y comparar múltiples modelos clásicos de Machine Learning para la clasificación del grado de deterioro cognitivo, representado a través de las escalas **GDS** y **GDS_R2**.

El flujo de trabajo incluye limpieza de datos, análisis exploratorio (EDA), reducción de dimensionalidad, optimización de hiperparámetros y la generación automática de reportes visuales de rendimiento.

## 🚀 Características Principales

- **Análisis Exploratorio de Datos (EDA):** Detección de *outliers*, manejo de valores nulos, conteo de características/etiquetas, y cálculo de la entropía para medir el balance de las clases.
- **Pipelines de Preprocesamiento:** Incluye técnicas de reducción de dimensionalidad usando Análisis de Componentes Principales (PCA) y descarte de variables altamente correlacionadas mediante el coeficiente de Pearson.
- **Modelos Evaluados:**
  - Naive Bayes (Gaussian)
  - Random Forest
  - Regresión Logística
  - Máquinas de Soporte Vectorial (SVM)
- **Entrenamiento Robusto:** Búsqueda aleatoria de hiperparámetros (`RandomizedSearchCV`) utilizando validación cruzada estricta (`LeaveOneOut`).
- **Visualización Avanzada:** Generación de Curvas ROC Multiclase (OvR con macro-average), Matrices de Confusión, y gráficos de distribución de frecuencias y entropía.

## 📂 Estructura del Código (`src/`)

- `main.py`: Punto de entrada principal. Coordina la carga de datos, división de conjuntos (entrenamiento/prueba), llamadas a EDA, entrenamiento de modelos, ensambles y graficado final.
- `eda.py`: Contiene funciones para el análisis estadístico, manipulación de índices, guardado de CSVs limpios y reportes de la estructura del dataset.
- `preprocessing.py`: Define los transformadores personalizados y los flujos (pipelines) de preparación de los datos antes de pasar a los estimadores.
- `classifiers.py`: Diccionario de configuración donde se instancian los algoritmos de clasificación y sus espacios de búsqueda (rangos y distribuciones de *Scipy* para los hiperparámetros).
- `train.py`: Lógica principal de iteración sobre preprocesamientos y clasificadores, cálculo de métricas (Accuracy, Precision, Recall, F1-Score) y registro de resultados en formato JSON/DataFrames.
- `visualization.py`: Módulo basado en `matplotlib` para plasmar los resultados del rendimiento y características de los datos en formato gráfico.

## 🛠️ Requisitos

El proyecto fue desarrollado utilizando Python y gestionado a través de **Conda**. Todas las dependencias (como `numpy`, `pandas`, `scikit-learn`, `scipy` y `matplotlib`) están definidas en el archivo `.yml` del entorno.

Para configurar el entorno virtual e instalar todas las dependencias automáticamente, ejecuta desde tu terminal:
```bash
# Crear el entorno a partir del archivo de configuración (asegúrate de colocar el nombre exacto de tu archivo .yml)
conda env create -f environment.yml

# Activar el entorno (reemplaza 'nombre_del_entorno' por el nombre definido dentro de tu archivo .yml)
conda activate nombre_del_entorno
```

## ⚙️ Uso

Para ejecutar el pipeline completo de experimentación, asegúrate de tener los datos originales en la ruta especificada en tus configuraciones (`config.py`) y simplemente ejecuta el script principal:

```bash
python src/main.py
```

Una vez finalizada la ejecución, los modelos se guardarán en las rutas correspondientes y los gráficos (curvas ROC, matrices de confusión) estarán disponibles en el directorio `SAVE_IMAGES_PATH`.
