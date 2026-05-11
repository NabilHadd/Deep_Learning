import numpy as np


def one_hot_encode(target_name, df):
  """One-hot encode the target variable.
  Args:
    target_name: The name of the target variable.
    df: The DataFrame containing the data.
    Returns:  A one-hot encoded numpy array of shape (num_samples, num_classes).
  """

  y_raw = df[target_name].astype(int)

  classes = sorted(y_raw.unique())
  class_to_index = {cls: idx for idx, cls in enumerate(classes)}
  y_encoded = y_raw.map(class_to_index).values
  Y = np.zeros((len(y_encoded), len(classes)), dtype=np.float32)
  Y[np.arange(len(y_encoded)), y_encoded] = 1.0

  return Y