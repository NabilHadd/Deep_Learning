from data_loader import data_loader
from visualization import frec_plot, entropy_plot
import eda

RAW_DATA_PATH = "./data/raw/15 atributos R0-R5.sav"
DATA_PATH = './data/processed/data.csv'

#Leemos los datos.
df_15_raw = data_loader(RAW_DATA_PATH)

#identificamos outliers y missing values:
#eda.ouliers_seeker(df_15_raw, 0, df_15_raw.columns.to_list().index('GDS'))
#eda.missing_values(df_15_raw)
#eda.feauters_labels_count(df_15_raw)





#aplicamos un procesamiento general en función del analisis
eda.make_csv(df_15_raw, DATA_PATH)

#Leemos los datos procesados
df_15 = data_loader(DATA_PATH)

#consideramos solo los labels
labels = df_15.columns.tolist()[df_15.columns.to_list().index('GDS'):]

print(eda.entropy_table(df_15[labels]))

entropy_plot(df_15[labels])
#debido a que la entropia define desorde, ejemplo (el desorden perfecto para 2 clases son frecuencias de 50/50)
#En este contexto la entropia representa balanceo, por lo que a mayor entropria mas balanceadas estan las clases.


#frec_plot(df_15)