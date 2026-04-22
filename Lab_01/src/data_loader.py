

import pandas as pd
from pathlib import Path

def data_loader(path):
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".sav":
        df = pd.read_spss(path)
    elif ext == ".csv":
        df = pd.read_csv(path, sep=';')
    else:
        raise ValueError(f"Formato no soportado: {ext}")

    return df