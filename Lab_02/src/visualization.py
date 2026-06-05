import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.eda import entropy_table
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


#hacer este grafico mas bonito
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
def entropy_plot(df, target_cols, save_path=None):
    entropias = entropy_table(df[target_cols])

    plt.bar(entropias.index, entropias["entropy"])
    plt.xlabel("GDS")
    plt.ylabel("Entropía normalizada")
    plt.title("Entropía por agrupación GDS (mayor = clases más balanceadas)")
    plt.ylim(0, 1)

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_describe_table(describe_df, save_path=None):
    """Renderiza la tabla de .describe() como figura."""
    df_t = describe_df.T.round(3)

    fig, ax = plt.subplots(figsize=(14, len(df_t) * 0.5 + 2))
    ax.axis('off')

    table = ax.table(
        cellText=df_t.values,
        rowLabels=df_t.index,
        colLabels=df_t.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(df_t.columns))))

    plt.title("Estadísticas descriptivas de features", fontsize=12, pad=20)
    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curves(y_true, y_score, class_names, gds_name, save_path=None):
    """
    Genera curvas ROC micro/macro y por clase para un GDS dado.
    y_true: array (n_samples, n_classes) con etiquetas reales.
    y_score: array (n_samples, n_classes) con probabilidades predichas.
    class_names: lista de strings con el nombre de cada clase.
    gds_name: string identificador del GDS (para el título).
    save_path: ruta opcional para guardar la figura.
    """
    n_classes = y_true.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc['macro'] = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(10, 7))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")

    plt.plot(fpr['micro'], tpr['micro'], linestyle=':', linewidth=2,
            label=f"Micro-average (AUC={roc_auc['micro']:.2f})")
    plt.plot(all_fpr, mean_tpr, linestyle='--', linewidth=2,
            label=f"Macro-average (AUC={roc_auc['macro']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curvas ROC — {gds_name}")
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()





#todo
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
