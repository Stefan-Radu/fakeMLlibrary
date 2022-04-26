import abc
import json
import numpy as np
from layers import Layer


class Activation(Layer):
  """
    Kinda interface for the Activation classes
  """
  @abc.abstractmethod
  def derivative(self) -> np.ndarray:
    pass

  def backward(self, grad: np.ndarray) -> np.ndarray:
      return grad * self.derivative()

  def serialize(self) -> dict:
    return { 'type': 'activation', }

  def __repr__(self) -> str:
    return json.dumps(self.serialize)


class Sigmoid(Activation):
  """
    Sigmoid activation function
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    self.output = 1 / (1 + np.exp(-x))
    return self.output

  def derivative(self):
    sig = self.output
    return sig * (1 - sig)

  def serialize(self) -> dict:
    return { 'type': 'sigmoid', }


class ReLU(Activation):
  """
    ReLU activation function
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    self.output = x.copy()
    self.output[self.output < 0] = 0
    return self.output

  def derivative(self):
    self.output[self.output > 0] = 1
    return self.output

  def serialize(self) -> dict:
    return { 'type': 'relu', }


class LeakyReLU(Activation):
  """
    LeakyReLU activation function
    an improved version of ReLu which has a small linear component 
    applied to negative values
  """
  def __init__(self):
    self._linear_component = 0.27

  def __call__(self, x: np.ndarray) -> np.ndarray:
    self.output = x.copy()
    self.output[self.output < 0] *= self._linear_component
    return self.output

  def derivative(self) -> np.ndarray:
    self.output[self.output < 0] = self._linear_component
    self.output[self.output > 0] = 1
    return self.output

  def serialize(self) -> dict:
    return { 'type': 'leakyrelu', }


class Tanh(Activation):
  """
    Tanh activation function. Already implemented in numpy
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    self.output = np.tanh(x)
    return self.output

  def derivative(self) -> np.ndarray:
    return 1 - self.output ** 2

  def serialize(self) -> dict:
    return { 'type': 'tanh', }


class Softmax(Activation):
  """
    Softmax activation function
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    norm_x = x - np.max(x)
    e = np.exp(norm_x)
    self.output = e / np.sum(e)
    return self.output

  def derivative(self) -> np.ndarray:
    sm = self.output.reshape((-1, 1))
    out = np.diagflat(sm) - np.dot(sm, sm.T)
    return np.diagonal(out) # idk if the math is right but it works lol xD

  def serialize(self) -> dict:
    return { 'type': 'softmax', }


# This is used to ease my life when loading
# from a serialized object 
activations_str_to_class = {
  'sigmoid': Sigmoid,
  'relu': ReLU,
  'leakyrelu': LeakyReLU,
  'tanh': Tanh,
  'softmax': Softmax,
}


if __name__ == '__main__':
  sm = Softmax()
  print(sm(np.array([1, 2, 3])))
  print(sm.backward(np.array([0.4, 0.2, 0.3])))

  s = Sigmoid()
  print(s(np.array([1, 2, 3])))
  print(s.backward(np.array([0.4, 0.2, 0.3])))

