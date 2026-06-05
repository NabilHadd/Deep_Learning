
from src.config import SAV_DATA_PATH, CSV_DATA_PATH, SCORES_CSV_PATH, TARGETS
from src.data_loader import sav_to_csv, load_csv
from src.evaluation import evaluate_model
from src.train_nn import train_shallow_nn
from src.preprocessing import one_hot_encode
from src.eda import missing_values, feauters_labels_count, plot_feature_histograms, plot_correlation_heatmap
from src.visualization import frec_plot, entropy_plot, plot_roc_curves
from sklearn.metrics import hamming_loss


def main():
  sav_to_csv(SAV_DATA_PATH, CSV_DATA_PATH)
  df = load_csv(CSV_DATA_PATH)


  # --- EDA ---
  target_keys = list(TARGETS.keys())
  non_feature_cols = target_keys + ['ID']

  print(f"\n=== Análisis Exploratorio de Datos ===")
  print(f"Total de muestras: {len(df)}")
  feauters_labels_count(df)
  missing_values(df)

  print("\n--- Estadísticas descriptivas de features ---")
  feature_cols = [c for c in df.columns if c not in non_feature_cols]
  print(df[feature_cols].describe())

  plot_feature_histograms(df, non_feature_cols)
  plot_correlation_heatmap(df, non_feature_cols)
  frec_plot(df)
  entropy_plot(df)
  # --- Fin EDA ---


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
  """
  X = df.drop(columns=list(TARGETS.keys()))
  Y = one_hot_encode(best_gds, df)

  final_model = train_shallow_nn(X=X, Y=Y)
  y_pred = final_model.predict(X)
  y_score = final_model.predict_proba(X)
  print(f"Hamming Loss for {best_gds}: {hamming_loss(Y, y_pred)}")

  plot_roc_curves(
      y_true=Y,
      y_score=y_score,
      class_names=TARGETS[best_gds],
      gds_name=best_gds
  )


main()
