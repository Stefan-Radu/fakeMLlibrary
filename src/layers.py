import abc
import numpy as np


class Layer:
  @abc.abstractmethod
  def __call__(self, x: np.ndarray) -> np.ndarray:
    pass

  @abc.abstractmethod
  def backward(self, x: np.ndarray) -> np.ndarray:
    pass


class LinearLayer(Layer):
  """
    The classic linear layer
  """
  def __init__(self, size, random_init=True):
    self._size = size
    self.weights = np.array([[]])
    self.biases = np.array([])
    self._random_init = random_init
    self.delta_weights = np.array([[]])
    self.delta_biases = np.array([])
    self.delta_count = 0

  def _initialize_parameters(self, input_size):
    self.weights = np.random.randn(self._size, input_size)
    self.biases = np.random.randn(self._size)
    self.delta_weights = np.zeros_like(self.weights)
    self.delta_biases = np.zeros_like(self.biases)
    self.delta_count = 0

  def __call__(self, x: np.ndarray) -> np.ndarray:
    self.input = x # activation from last layer
    if self.weights.size == 0 or self.biases.size == 0:
      input_size = x.shape[-1]
      self._initialize_parameters(input_size)

    self.output = self.weights @ x + self.biases
    return self.output

  def backward(self, grad: np.ndarray) -> np.ndarray:
    self.delta_count += 1
    self.delta_weights += grad.reshape(1, -1).T @ self.input.reshape(1, -1)
    self.delta_biases += grad
    return self.weights.T @ grad

  def apply_delta(self, lr):
    self.weights -= self.delta_weights * lr / self.delta_count
    self.biases -= self.delta_biases * lr / self.delta_count
    self.delta_weights = np.zeros_like(self.weights)
    self.delta_biases = np.zeros_like(self.biases)
    self.delta_count = 0
