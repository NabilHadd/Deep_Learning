
from src.config import SAV_DATA_PATH, CSV_DATA_PATH, SCORES_CSV_PATH, TARGETS
from src.data_loader import sav_to_csv, load_csv
from src.evaluation import evaluate_model
from src.train_nn import train_shallow_nn
from src.preprocessing import one_hot_encode


def main():



  #Esto solo son pruebas.
  sav_to_csv(SAV_DATA_PATH, CSV_DATA_PATH)
  df = load_csv(CSV_DATA_PATH)
  evaluate_model(df, list(TARGETS.keys()), SCORES_CSV_PATH)
  df_scores = load_csv(SCORES_CSV_PATH)
  print(df_scores)


  ##aca podrias calcular cual es el mejor gds y entrenar solo sobre ese, no es necesario ejecutar 
  #todo denuevo debido a que lo puedes leer desde el csv

  X = df.drop(columns=list(TARGETS.keys()))
  #Y = one_hot_encode(te falta el targen,df)  <--------------- !!!!!!

  train_shallow_nn(df.drop(columns=list(TARGETS.keys())), df[TARGETS.keys()])

  






main()