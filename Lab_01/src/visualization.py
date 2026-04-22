import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from eda import entropy_table
from sklearn.metrics import roc_curve, auc, precision_recall_curve,  average_precision_score
import numpy as np
from sklearn.preprocessing import label_binarize


def frec_plot(df):
    # Lista de columnas codificadas
    experiments = ['GDS', 'GDS_R1', 'GDS_R2', 'GDS_R3', 'GDS_R4', 'GDS_R5']
    width = 0.15
    sum_width = 0

    plt.figure(figsize=(8, 6))

    for exp in experiments:
        print(exp)

        # Obtener conteo de valores únicos ordenados por índice
        counts = df[exp].value_counts().sort_index()
        print(counts)

        # Crear lista completa de clases (del 1 al 7), completando con 0 si falta alguna
        classes = list(range(1, 8))
        data = [counts.get(c, 0) for c in classes]

        # Graficar
        plt.bar(np.array(classes) + sum_width, data, width=width, label=exp)

        sum_width += width

        # Mostrar conteo en consola
        print(counts)
        print('=' * 60)

    # Etiquetas y leyenda
    plt.title('Frecuencias de valores de clases vs tipos de codificación')
    plt.xlabel('Etiqueta de la clase (GDS)')
    plt.ylabel('Frecuencia')
    plt.xticks(np.arange(1, 8), labels=[str(i) for i in range(1, 8)])
    plt.legend()
    plt.show()


def entropy_plot(df):
    entropias = entropy_table(df)

    plt.bar(entropias.index, entropias["entropy"])
    plt.xlabel("Label")
    plt.ylabel("Entropia")
    plt.title("Entropia normalizada de frecuencias por label")

    plt.show()



def plot_roc_multilabel(models, X_test, y_test):
    n_labels = y_test.shape[1]

    for name, model in models:
        plt.figure(figsize=(6,5))

        y_score = model.predict_proba(X_test)

        # Caso multilabel típico de sklearn → lista
        if isinstance(y_score, list):
            y_score = np.column_stack([s[:, 1] for s in y_score])

        for i in range(n_labels):
            y_true = y_test[:, i]
            unique_vals = np.unique(y_true)

            # 🔹 Caso binario normal
            if len(unique_vals) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"Label {i} (AUC={roc_auc:.2f})")

            # 🔹 Caso multiclass → One-vs-Rest
            elif len(unique_vals) > 2:
                y_bin = label_binarize(y_true, classes=unique_vals)

                for c in range(y_bin.shape[1]):
                    fpr, tpr, _ = roc_curve(y_bin[:, c], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, linestyle='--',
                            label=f"Label {i} class {unique_vals[c]} (AUC={roc_auc:.2f})")

            else:
                continue  # caso degenerado

        plt.plot([0,1], [0,1], 'k--')
        plt.title(f"ROC - {name}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.show()



def plot_pr_multilabel(models, X_test, y_test):
    n_labels = y_test.shape[1]

    for name, model in models:
        plt.figure(figsize=(6,5))

        y_score = model.predict_proba(X_test)

        if isinstance(y_score, list):
            y_score = np.column_stack([s[:,1] for s in y_score])

        for i in range(n_labels):
            precision, recall, _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            ap = average_precision_score(y_test[:, i], y_score[:, i])

            plt.plot(recall, precision, label=f"Label {i} (AP={ap:.2f})")

        plt.title(f"PR - {name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()