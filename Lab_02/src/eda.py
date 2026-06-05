import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def missing_values(df):
    df = erase_index(df)
    new_df = df.dropna(axis='index', how='any')
    print(f'Hay {len(df)-len(new_df)} datos faltantes.')
 

def erase_index(df):
    """
    Elimina columnas que probablemente provienen de un índice exportado.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro debe ser un pandas DataFrame")

    # nombres típicos de índices exportados
    possible_index_cols = [
        "Unnamed: 0",
        "index",
        "level_0"
    ]

    cols_to_drop = []

    for col in df.columns:

        # Caso 1: nombres típicos
        if str(col) in possible_index_cols:
            cols_to_drop.append(col)

        # Caso 2: columnas completamente secuenciales 0..n
        elif pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna().astype(int)

            if len(values) == len(df):
                expected = list(range(len(df)))

                if values.tolist() == expected:
                    cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop)





def feauters_labels_count(df):
    df = erase_index(df)
    columns = df.columns.tolist()

    gds_index = columns.index('GDS')

    features = columns[:gds_index]
    labels = columns[gds_index:]

    print(f'Labels:\n-'
        + f'\n-'.join(labels)
        + '\n' +  f'{len(labels)} labels.\n\n'
        + 'Features:\n-'
        + '\n-'.join(features)
        + '\n' + f'{len(features)} features.' )
    

#crear tabla de frecuencias para cada etiqueta.


def frec_table(df, label):
    frec = df[label].value_counts()
    rel_frec = frec/len(df)
    frec_table = pd.concat([frec, rel_frec], axis="columns")
    frec_table.columns= ['frec', 'rel_frec']
    frec_table.dropna()

    return frec_table


def entropy_table(df):
    frecuencias = [frec_table(df, col) for col in df.columns]

    entropias = {
        x.index.name: float((-(x['rel_frec'] * np.log(x['rel_frec'])).sum())/np.log(len(x)))
        for x in frecuencias
    }

    entropias_df = pd.DataFrame.from_dict(entropias, orient='index')
    entropias_df.columns = ['entropy']
    entropias_df.index.name = 'label'
    entropias_df = entropias_df.sort_values(by="entropy")
    
    return entropias_df
        
#Se normalizan las entropias para hacer una comparación mas robusta. Debido a que a un label con mas clases le es mas dificil estar desordenado.


def plot_feature_histograms(df, exclude_cols):
    """Histogramas de distribución por feature."""
    feature_df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    n = len(feature_df.columns)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_df.columns):
        axes[i].hist(feature_df[col].dropna(), bins=10, edgecolor='black')
        axes[i].set_title(col, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribución de features")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, exclude_cols):
    """Heatmap de correlación entre features."""
    feature_df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    corr = feature_df.corr()
    labels = corr.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=6)

    plt.title("Correlación entre features")
    plt.tight_layout()
    plt.show()
