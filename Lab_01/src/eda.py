import numpy as np
import pandas as pd


def make_csv(df, output, idx=False):
    if idx :
        df.to_csv(path_or_buf=output, sep=';', na_rep='-', index=False)
    else:
        df = erase_index(df)
        df.to_csv(path_or_buf=output, sep=';', na_rep='-', index=False)
    
    


def missing_values(df):
    df = erase_index(df)
    new_df = df.dropna(axis='index', how='any')
    print(f'Hay {len(df)-len(new_df)} datos faltantes.')



def erase_index(df):

    labels = df.columns.tolist()
    df = df.drop(labels=labels[0], axis='columns')
    return df


def ouliers_seeker(df, low, high):
    df = erase_index(df)
    results = []
    
    for col in df.columns.tolist()[low: high - 1]: #se le resta 1 debido a que eliminamos el indice y eso hace que high avance 1
        if df[col].max() > 1:
            results.append(f"outlier superior en {col}\n")
        if df[col].min() < 0:
            results.append(f"outlier inferior en {col}\n")
        
    if len(results) < 1:
        print('No existen outliers')
    else:
        print('\n'.join(results))



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