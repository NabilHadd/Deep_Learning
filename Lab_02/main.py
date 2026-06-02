
from src.config import SAV_DATA_PATH, CSV_DATA_PATH, SCORES_CSV_PATH, TARGETS
from src.data_loader import sav_to_csv, load_csv
from src.evaluation import evaluate_model
from src.train_nn import train_shallow_nn
from src.preprocessing import one_hot_encode
from sklearn.metrics import hamming_loss


def main():
  #agregar algo de EDA por ejemplo la distribución de las clases, correlación entre features, etc.
  """
    Se hace un poco de analisis exploratorio de los datos para entender mejor el dataset:
    - Distribución de las clases para cada target
    - Correlación entre features
    A partir de ello se construye una hipotesis del mejor gds
  """

  """
    Se convierte el archivo .sav a .csv y se carga el dataframe
  """
  sav_to_csv(SAV_DATA_PATH, CSV_DATA_PATH)
  df = load_csv(CSV_DATA_PATH)


  """
    Se evalúan los modelos para cada target y se guardan los scores en un csv
    No es necesario ejecutar esto cada vez, ya que los scores se guardan en un csv
  """
  #evaluate_model(df, list(TARGETS.keys()), SCORES_CSV_PATH)


  """
    Se lee el csv con los scores, se calcula el mejor GDS y se entrena un modelo solo para ese GDS
  """
  df_scores = load_csv(SCORES_CSV_PATH)
  df_mean_scores = df_scores.apply(lambda x: x * -1).mean().sort_values(ascending=True)

  best_gds = df_mean_scores.index[0]
  print(f"Best GDS: {best_gds} with mean score: {df_mean_scores[best_gds]}")
  print(TARGETS[best_gds])



  """
    Se prepara X e Y para entrenar el GDS con los mejores score
    Luego se reentrena un modelo solo para ese GDS y se calcula el hamming loss
    para ver que tan bien se ajusta el modelo a los datos,
    luego se puede comparar este resultado con el resultado del csv para ver que tanta varianza existe 
  """

  X = df.drop(columns=list(TARGETS.keys()))
  Y = one_hot_encode(best_gds,df)

  y_pred = train_shallow_nn(X=X, Y=Y).predict(X)
  print(f"Hamming Loss for {best_gds}: {hamming_loss(Y, y_pred)}")
  #luego puedes comparar este resultado con el resultado del csv para ver que tanta varianza existe
  #Construir las curvas roc correspondientes a cada gds y compararlas entre si



main()