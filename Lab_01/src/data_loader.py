import pandas as pd

def data_loader(path):
    df = pd.read_spss(path)
    return df