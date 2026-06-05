import pandas as pd

from src.data_loader import make_csv
from src.preprocessing import one_hot_encode
from src.scorer_model import scorer_shallow_nn


def evaluate_model(df, targets, scores_csv_path):

  scores_dict = {}
  
  for target in targets:
    Y = one_hot_encode(target, df)

    #pensar en que preprossesors de pueden definir y como aplicar el cross validation anidado.
    scores = scorer_shallow_nn(df.drop(columns=targets), Y)
    scores_dict[target] = scores

    scores_df = pd.DataFrame(scores_dict)

    
  make_csv(scores_df, scores_csv_path)