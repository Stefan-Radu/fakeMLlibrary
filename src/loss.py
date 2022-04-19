import numpy as np


class MSE:
  """
    Mean Squared Error loss function
  """
  def __call__(self, result: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.square(np.subtract(result, expected)).mean()


# TODO more loss functions
