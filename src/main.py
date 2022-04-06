import numpy as np



class LinearLayer():
  def __init__(self, size, random_init=True):
    self._size = size
    self._weights = None
    self._biases = None
    self._random_init = random_init

  def _initialize_parameters(self, input_neurons):
    if self._weights == None:
      self._weights = np.random.rand(input_neurons, self._size)
    if self._biases == None:
      self._biases = np.random.rand(self._size)

  def __call__(self, x):
    if self._weights == None or self._biases == None:
      input_size = x.shape[0]
      self._initialize_parameters(input_size)
    return np.dot(self._weights, x) + self._biases
