# Laboratorio 03: Aprendizaje Multitarea — Clasificación de Género y Estimación de Edad

**Asignatura:** Deep Learning (2026-01) — Universidad Católica del Norte  
**Profesor:** Dr. Juan Bekios Calfa  
**Autores:** Nabil Haddad Pomar · Tomás Méndez Díaz  
**Repositorio:** https://github.com/NabilHadd/Deep_Learning/tree/master/Lab_03

## Descripción

Sistema de aprendizaje multitarea para predecir simultáneamente **género** (clasificación binaria) y **edad** (regresión) a partir de imágenes faciales del dataset [UTKFace](https://susanqq.github.io/UTKFace/). Se comparan seis estrategias de complejidad creciente y se cuantifica el *domain gap* entre datos pre-alineados y un pipeline de producción basado en cascadas de Haar.

La función de pérdida es:

```
L = CrossEntropyLoss(género) + λ · SmoothL1Loss(edad)
```

## Resultados principales

| Experimento | Modelo | Acc género | MAE edad | R² |
|---|---|:---:|:---:|:---:|
| E5 | `resnet_finetuning_lambda_high` | **0.9185** | **5.60** | **0.829** |
| E5 | `resnet_finetuning_base` | 0.9193 | 6.46 | 0.782 |
| E3 | `cnn_base` | 0.8746 | 9.66 | 0.538 |
| E1 | `classical_base` | 0.8698 | 10.33 | 0.525 |
| E6 | `resnet_aligned_base` † | 0.8856 | 8.39 | 0.631 |
| E2 | `mlp_no_dropout` | 0.8572 | 11.87 | 0.213 |

† Evaluado sobre imágenes del mundo real (raw + FaceAligner). Mejor modelo para despliegue en producción.

**Domain gap medido:** la CNN entrenada con datos alineados pierde 18.4 pp de accuracy y su R² se vuelve negativo (−0.279) al evaluar sobre imágenes reales. Entrenar en el dominio de producción cierra completamente la brecha.

---

## Estructura

```
.
├── main.py                        # Orquestador principal
├── streamlit_main.py              # Lanzador de la app de inferencia
├── src/
│   ├── config.py                  # Hiperparámetros y variables de entorno
│   ├── baselines/classical_todo.py
│   ├── data/
│   │   ├── parser.py
│   │   ├── dataset.py
│   │   ├── datamodule.py          # Datos alineados manualmente (E1–E5)
│   │   ├── aligned_datamodule.py  # Datos raw + FaceAligner (E6)
│   │   ├── face_aligner.py        # Pipeline de alineación automática
│   │   └── transforms.py
│   ├── models/
│   │   ├── cnn.py
│   │   ├── mlp_todo.py
│   │   └── resnet_todo.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── experiment_runner.py   # Catálogo completo de experimentos
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── reporter.py
│   └── inference/
│       ├── predictor.py           # Carga cualquier modelo (CNN/MLP/ResNet)
│       └── streamlit_app.py
├── artifacts/
│   ├── checkpoints/<nombre>/best_model.pt
│   ├── plots/<nombre>/
│   └── reports/
└── informe/main.pdf
```

---

## Instalación

```bash
conda env create -f environment.yml
conda activate lab03-dl-2026-01
cp .env.example .env
```

Configura la ruta al dataset en `.env`:

```dotenv
UTKFACE_DIR=/ruta/a/UTKFace
CNN_CHECKPOINT=artifacts/checkpoints/resnet_aligned_base/best_model.pt
```

El dataset debe contener archivos con nombres como `25_1_2_20170116174525125.jpg` (edad_género_raza_timestamp) basicamente el mismo formato que sigue UTKFace.

---

## Ejecución de experimentos

```bash
# Ver catálogo completo
python main.py --list

# Ejecutar un experimento
python main.py --experiment resnet_finetuning_lambda_high

# Ejecutar todos los experimentos
python main.py --all
```

Los experimentos completados se saltan automáticamente si `artifacts/checkpoints/<nombre>/best_model.pt` ya existe. Para forzar el reentrenamiento debes eliminar el respectivo checkpoint.

---

## Aplicación de inferencia (Streamlit)

```bash
streamlit run streamlit_main.py
```
La app permite subir una foto, la pasa por el mismo pipeline FaceAligner usado en E6 y predice género y edad con el modelo configurado en el .env. Soporta modelos CNN, MLP y ResNet.
Ademas hay una versión desplegada que utiliza el modelo resultante del experimento E6 "resnet_aligned_base". Para probarla accede al siguiente [link](deep-learning-nabil.streamlit.app)

---


## Experimentos

| ID | Estrategia | Ablaciones principales |
|---|---|---|
| E1 | PCA + scikit-learn (SVM/RF) | `pca_low` |
| E2 | MLP multitarea | `no_dropout`, `lambda_low/high` |
| E3 | CNN simple | `no_dropout`, `lambda_low/high`, `age_normalized` |
| E4 | ResNet18 congelada | `lambda_low/high`, `no_augmentation` |
| E5 | ResNet18 fine-tuning (layer4) | `lambda_high`, `unfreeze_more`, `lr_low` |
| E6 | Domain gap con FaceAligner | `cnn_aligned_base`, `resnet_aligned_base` |

Partición: 70/15/15 (entrenamiento/validación/prueba), semilla 42 fija para todos los experimentos.

