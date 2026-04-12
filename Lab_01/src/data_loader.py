import pandas as pd
#Podría ser mejor


def data_loader(path):
    ext = path.split('.')[-1]
    if ext == "sav":
        df = pd.read_spss(path)
    else:
        df = pd.read_csv(path, sep=';')


    return df