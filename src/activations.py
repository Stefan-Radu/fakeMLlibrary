import abc
import numpy as np
from layers import Layer


class Activation(Layer):
  @abc.abstractmethod
  def derivative(self) -> np.ndarray:
    pass

  def backward(self, grad: np.ndarray) -> np.ndarray:
    return grad * self.derivative()


class Step(Activation):
  """
    Step activation function
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    x[x >= 0] = 1
    x[x < 0] = 0
    return x


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


#TODO derivative
class LeakyReLU(Activation):
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


#TODO derivative
class Tanh(Activation):
  """
    Tanh activation function. Already implemented in numpy
  """
  def __call__(self, x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


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
    print(out.shape)
    return np.diagflat(sm) - np.dot(sm, sm.T)


if __name__ == '__main__':
  sm = Softmax()
  print(sm(np.array([1, 2, 3, 5])))
  print(sm.backward(np.array([0.4, 0.2, 0.3, 3.4])))

  s = Sigmoid()
  print(s(np.array([-1, 0, 1])))
  print(s.backward(np.array([0.9, 0.2, 0.1])))

