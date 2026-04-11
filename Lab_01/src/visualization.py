import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

