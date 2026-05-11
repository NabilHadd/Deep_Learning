

from src.config import SAV_DATA_PATH, CSV_DATA_PATH, TARGETS
from src.preprocessing import one_hot_encode
from src.data_loader import sav_to_csv, load_csv

def main():


  sav_to_csv(SAV_DATA_PATH, CSV_DATA_PATH)
  df = load_csv(CSV_DATA_PATH)


  targets = TARGETS.keys()
  
  for target in targets:
    Y = one_hot_encode(target, df)
    print(f"One-hot encoded target '{target}':")
    print(Y[:5])  # Imprime las primeras 5 filas del resultado

    


main()