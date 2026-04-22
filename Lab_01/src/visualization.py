import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from eda import entropy_table
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve,  average_precision_score
import numpy as np
from sklearn.preprocessing import label_binarize



#Función que grafica la frecuencia de las diferentes clases para cada uno de las agrupaciones GDS
def frec_plot(df):
    # Lista de columnas codificadas
    experiments = ['GDS', 'GDS_R1', 'GDS_R2', 'GDS_R3', 'GDS_R4', 'GDS_R5']
    width = 0.15
    sum_width = 0

    plt.figure(figsize=(8, 6))

    for exp in experiments:

        # Obtener conteo de valores únicos ordenados por índice
        counts = df[exp].value_counts().sort_index()

        # Crear lista completa de clases (del 1 al 7), completando con 0 si falta alguna
        classes = list(range(1, 8))
        data = [counts.get(c, 0) for c in classes]

        # Graficar
        plt.bar(np.array(classes) + sum_width, data, width=width, label=exp)

        sum_width += width

        # Mostrar conteo en consola

    # Etiquetas y leyenda
    plt.title('Frecuencias de valores de clases vs tipos de codificación')
    plt.xlabel('Etiqueta de la clase (GDS)')
    plt.ylabel('Frecuencia')
    plt.xticks(np.arange(1, 8), labels=[str(i) for i in range(1, 8)])
    plt.legend()
    plt.show()



#Función para calcular la entropia de las frecuencias de cada conjunto de etiquetas.
def entropy_plot(df):
    entropias = entropy_table(df)

    plt.bar(entropias.index, entropias["entropy"])
    plt.xlabel("Label")
    plt.ylabel("Entropia")
    plt.title("Entropia normalizada de frecuencias por label")

    plt.show()



#Función para graficar la curva roc con su correspondiente area bajo la curva
def plot_roc_multiclass(model, model_name, X_test, y_test, classes_names, save_path):
    classes = np.unique(y_test)
    n_classes = len(classes)

    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(8,6))

    # ===== OvR curves =====
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(
            fpr[i], tpr[i],
            label=f"Deterioro {classes_names[i]} (AUC = {roc_auc[i]:.2f})"
        )

    # ===== Macro-average =====
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    plt.plot(
        all_fpr, mean_tpr,
        linestyle="--",
        linewidth=2,
        label=f"Macro-average (AUC = {macro_auc:.2f})"
    )

    # línea aleatoria
    plt.plot([0, 1], [0, 1], "k--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Multiclase (OvR + Macro-average) of " + model_name)
    plt.legend()
    plt.grid()

    save_path.parent.mkdir(parents=True, exist_ok=True) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def confusion_matrix_plot(model, model_name, X_test, y_test, classes_names, save_path):
    labels = sorted(np.unique(y_test))
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, classes_names)
    plt.yticks(tick_marks, classes_names)

    thresh = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()