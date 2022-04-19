from __future__ import annotations

# import abc
import numpy as np



# class FakeLayer(abc.ABC):
#   @abc.abstractmethod
#   def __init__(self, size=None):
#     self.size = size
#
#   @abc.abstractmethod
#   def __call__(self, x: FakeLayer) -> FakeLayer:
#     pass



class LinearLayer:
  def __init__(self, size, random_init=True):
    self._size = size
    self._weights = np.array([[]])
    self._biases = np.array([])
    self._random_init = random_init

  def _initialize_parameters(self, input_size):
    if self._weights.size == 0:
      self._weights = np.random.randn(self._size, input_size)
    if self._biases.size == 0:
      self._biases = np.random.randn(self._size)

  def __call__(self, x: np.ndarray) -> np.ndarray:
    if self._weights.size == 0 or self._biases.size == 0:
      input_size = x.shape[0]
      self._initialize_parameters(input_size)

    return self._weights @ x + self._biases



x = np.array((0, 1))
layer = LinearLayer(5)
print(layer(x))
