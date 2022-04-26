import abc
import json
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
  def __init__(self, out_size, in_size=None, weights=[[]], biases=[]):
    self.out_size = out_size
    self.in_size = in_size
    self.weights: np.ndarray = np.array(weights)
    self.biases: np.ndarray = np.array(biases)
    self.delta_weights = np.zeros_like(self.weights)
    self.delta_biases = np.zeros_like(self.biases)
    self.delta_count = 0


  @classmethod
  def load(cls, data: dict):
    in_size = int(data['in_size'])
    out_size = int(data['out_size'])
    weights = data['weights']
    biases = data['biases']
    return cls(out_size, in_size, weights, biases)


  def _initialize_parameters(self, input_size):
    self.in_size = input_size
    self.weights = np.random.randn(self.out_size, input_size)
    self.biases = np.random.randn(self.out_size)
    self.delta_count = 0
    self.delta_weights = np.zeros_like(self.weights)
    self.delta_biases = np.zeros_like(self.biases)


  def __call__(self, x: np.ndarray) -> np.ndarray:
    self.input = x # activation from last layer
    if self.weights.size == 0 or self.biases.size == 0:
      self._initialize_parameters(x.shape[-1])

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


  def serialize(self):
    data = {
      'type': 'linear',
      'in_size': self.in_size,
      'out_size': self.out_size,
      'weights': self.weights.tolist(),
      'biases': self.biases.tolist(),
    }
    return data


  def __repr__(self):
    return json.dumps(self.serialize())


if __name__ == '__main__':
  l = LinearLayer(13)
  print(repr(l))
