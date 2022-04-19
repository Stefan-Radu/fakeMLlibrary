import numpy as np


class Step:
  """
    Step activation function
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    x[x >= 0] = 1
    x[x < 0] = 0
    return x


class Sigmoid:
  """
    Sigmoid activation function
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class ReLU:
  """
    ReLU activation function
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    # fastest implementation. actually modifies x in place
    # output = x.copy()
    # output[x < 0] = 0
    # return output
    x[x < 0] = 0
    return x


class LeakyReLU:
  """
    LeakyReLU activation function
    an improved version of ReLu which has a small linear component 
    applied to negative values
  """
  def __init__(self):
    self._linear_component = 1e-2


  def __call__(self, x: np.ndarray) -> np.ndarray:
    x[x < 0] *= self._linear_component
    return x


class Tanh:
  """
    Tanh activation function. Already implemented in numpy
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class Softmax:
  """
    Softmax activation function
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    e = np.exp(x)
    return x / np.sum(e)
