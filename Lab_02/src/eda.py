import numpy as np
import pandas as pd

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
