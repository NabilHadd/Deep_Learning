from data_loader import data_loader
from visualization import frec_plot

DATA_PATH = "./data/raw/15 atributos R0-R5.sav"

df_15 = data_loader(DATA_PATH)

frec_plot(df_15)